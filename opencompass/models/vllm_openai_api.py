"""vLLM OpenAI-compatible API model with continuation scoring."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from opencompass.registry import MODELS

from .openai_api import OpenAISDK


@MODELS.register_module()
class VLLMOpenAIAPI(OpenAISDK):
    """OpenAI-compatible vLLM client with exact continuation likelihoods.

    Generation uses ``/v1/chat/completions`` by default through
    :class:`OpenAISDK`; setting ``generation_endpoint='completions'`` selects
    raw ``/v1/completions`` for lm-evaluation-harness protocols that do not
    apply a chat template. ``get_loglikelihood`` uses vLLM's
    ``/v1/completions`` extensions ``prompt_logprobs`` and
    ``return_token_ids``. Only tokens after the separately tokenized context
    are summed, matching lm-evaluation-harness' multiple-choice protocol.
    """

    def __init__(self,
                 *args,
                 generation_endpoint: str = 'chat',
                 completion_extra_body: Optional[Dict] = None,
                 stream_chat: bool = False,
                 stream_idle_timeout: Optional[float] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if generation_endpoint not in {'chat', 'completions'}:
            raise ValueError('generation_endpoint must be either "chat" or '
                             '"completions"')
        self.generation_endpoint = generation_endpoint
        self.completion_extra_body = completion_extra_body or {}
        self.stream_chat = stream_chat
        self.stream_idle_timeout = stream_idle_timeout
        self._prompt_token_ids_cache: Dict[str, Tuple[int, ...]] = {}
        self._prompt_token_ids_cache_lock = Lock()

    def generate(self,
                 inputs,
                 max_out_len: int = 512,
                 temperature: float = 0.7,
                 stopping_criteria: Optional[List[str]] = None,
                 **kwargs) -> List[Union[str, Dict[str, str]]]:
        """Generate while forwarding dataset-specific stop sequences."""
        if self.temperature is not None:
            temperature = self.temperature
        stops = stopping_criteria or None

        def failed_prediction(error: Exception) -> Dict[str, str]:
            self.logger.exception(
                'Generation failed after all retries; recording an empty '
                'prediction so the remaining batch can continue.')
            return {
                'reasoning_content': '',
                'content': '',
                'inference_error': f'{type(error).__name__}: {error}',
            }

        if len(inputs) == 1:
            try:
                result = self._generate(inputs[0], max_out_len, temperature,
                                        stops)
            except Exception as error:
                result = failed_prediction(error)
            return [result]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._generate, input, max_out_len,
                                temperature, stops): index
                for index, input in enumerate(inputs)
            }
            results = [None] * len(inputs)
            for future in tqdm(as_completed(futures),
                               total=len(inputs),
                               desc='Inferencing'):
                try:
                    result = future.result()
                except Exception as error:
                    result = failed_prediction(error)
                results[futures[future]] = result
            return results

    def _generate(
        self,
        input,
        max_out_len: int,
        temperature: float,
        stopping_criteria: Optional[List[str]] = None
    ) -> Union[str, Dict[str, str]]:
        """Generate with chat or raw completions as explicitly configured."""
        if self.generation_endpoint == 'chat':
            if getattr(self, 'stream_chat', False):
                return self._generate_chat_stream(input, max_out_len,
                                                  temperature,
                                                  stopping_criteria)
            return super()._generate(input, max_out_len, temperature,
                                     stopping_criteria)
        if isinstance(input, str):
            prompt = input
        elif isinstance(input, list) and all(
                isinstance(message, dict)
                and message.get('role') in {'system', 'user', 'assistant'}
                and isinstance(message.get('content'), str)
                for message in input):
            if getattr(self, 'meta_template', None) is not None:
                raise ValueError(
                    'Raw /v1/completions generation with structured messages '
                    'requires meta_template=None.')
            # lm-evaluation-harness renders its internal messages as plain
            # text by concatenating only their contents when chat templating
            # is disabled. Preserve the same behavior for benchmark configs
            # that also support /v1/chat/completions.
            prompt = ''.join(message['content'] for message in input)
        else:
            raise TypeError(
                'Raw /v1/completions generation requires either a string '
                'prompt or OpenAI-format messages with role/content fields.')

        query_data = dict(model=self.path,
                          prompt=prompt,
                          n=1,
                          temperature=temperature,
                          extra_body=self.extra_body)
        if max_out_len is not None:
            query_data['max_tokens'] = max_out_len
        if stopping_criteria:
            query_data['stop'] = stopping_criteria
        if self.openai_extra_kwargs:
            query_data.update(self.openai_extra_kwargs)

        for attempt in range(self.retry):
            self.acquire()
            try:
                response = self.openai_client.completions.create(
                    **query_data, timeout=self.timeout)
                choices = getattr(response, 'choices', None)
                if not choices:
                    raise RuntimeError(
                        'vLLM completions response has no choices')
                text = getattr(choices[0], 'text', None)
                if text is None:
                    payload = self._response_dict(response)
                    text = (payload.get('choices') or [{}])[0].get('text')
                if text is None:
                    raise RuntimeError(
                        'vLLM completions response has no text field')
                return text
            except Exception as error:
                self.logger.error('vLLM raw completions request failed '
                                  f'(attempt {attempt + 1}/{self.retry}): '
                                  f'{error}')
                if attempt + 1 == self.retry:
                    raise
            finally:
                self.release()
        raise RuntimeError('vLLM raw completions request failed')

    def _generate_chat_stream(
        self,
        input,
        max_out_len: int,
        temperature: float,
        stopping_criteria: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Stream a chat response while keeping reasoning and answer apart.

        Large thinking responses can finish in the vLLM engine before the
        non-streaming HTTP layer has serialized and buffered the complete JSON
        response. Reading chunks as they are produced avoids that response
        pile-up. It also preserves the same public result shape as
        :class:`OpenAISDK`, so evaluators only score the final ``content`` and
        never accidentally score private reasoning.
        """
        from openai import APITimeoutError, APIStatusError, BadRequestError

        messages, max_out_len = self._preprocess_messages(
            input, max_out_len, self.max_seq_len, self.mode,
            self.get_token_len)
        query_data = dict(model=self.path,
                          n=1,
                          temperature=temperature,
                          messages=messages,
                          extra_body=self.extra_body)
        if max_out_len is not None:
            query_data['max_tokens'] = max_out_len
        if stopping_criteria:
            query_data['stop'] = stopping_criteria
        if self.openai_extra_kwargs:
            query_data.update(self.openai_extra_kwargs)
        # This is a transport choice, not a sampling option. Do not allow an
        # overlapping extra kwarg to silently disable it.
        query_data['stream'] = True

        for attempt in range(self.retry):
            response_stream = None
            content_chunks = []
            reasoning_chunks = []
            finish_reason = None
            self.acquire()
            try:
                response_stream = self.openai_client.chat.completions.create(
                    **query_data,
                    timeout=(self.stream_idle_timeout
                             if self.stream_idle_timeout is not None else
                             self.timeout))
                for chunk in response_stream:
                    choices = getattr(chunk, 'choices', None)
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = getattr(choice, 'delta', None)
                    if delta is not None:
                        content = getattr(delta, 'content', '') or ''
                        reasoning_content = (
                            getattr(delta, 'reasoning_content', '') or '')
                        reasoning = getattr(delta, 'reasoning', '') or ''
                        if content:
                            content_chunks.append(content)
                        if reasoning_content or reasoning:
                            reasoning_chunks.append(reasoning_content
                                                    or reasoning)
                    if getattr(choice, 'finish_reason', None) is not None:
                        finish_reason = choice.finish_reason

                if finish_reason is None:
                    raise RuntimeError(
                        'vLLM streaming response ended without finish_reason')
                content = ''.join(content_chunks)
                reasoning_content = ''.join(reasoning_chunks)
                if not content and not reasoning_content:
                    if finish_reason in {'stop', 'content_filter'}:
                        return {'reasoning_content': '', 'content': ''}
                    raise RuntimeError(
                        'vLLM streaming response contains no text')
                return {
                    'reasoning_content': reasoning_content,
                    'content': content,
                }
            except APITimeoutError as error:
                # A proxy can occasionally forward every generated SSE chunk
                # but omit the terminal event, leaving the OpenAI iterator
                # blocked even though vLLM has already marked the request as
                # finished.  Once text has arrived, preserve it as the
                # length-limited response instead of discarding a costly
                # long-context generation and retrying the whole request.
                if content_chunks or reasoning_chunks:
                    self.logger.warning(
                        'vLLM streaming chat response became idle after '
                        'receiving text but before a terminal event; '
                        'preserving the received response: %s', error)
                    return {
                        'reasoning_content': ''.join(reasoning_chunks),
                        'content': ''.join(content_chunks),
                    }
                self.logger.error('vLLM streaming chat request timed out '
                                  f'(attempt {attempt + 1}/{self.retry}): '
                                  f'{error}')
                if attempt + 1 == self.retry:
                    raise
            except (BadRequestError, APIStatusError) as error:
                status_code = error.status_code
                if (status_code is not None
                        and status_code in self.status_code_mappings):
                    return {
                        'reasoning_content': '',
                        'content': self.status_code_mappings[status_code],
                    }
                self.logger.error('vLLM streaming chat request failed '
                                  f'(attempt {attempt + 1}/{self.retry}): '
                                  f'{error}')
                if attempt + 1 == self.retry:
                    raise
            except Exception as error:
                self.logger.error('vLLM streaming chat request failed '
                                  f'(attempt {attempt + 1}/{self.retry}): '
                                  f'{error}')
                if attempt + 1 == self.retry:
                    raise
            finally:
                if response_stream is not None:
                    try:
                        response_stream.close()
                    except Exception:
                        pass
                self.release()
        raise RuntimeError('vLLM streaming chat request failed')

    @staticmethod
    def _response_dict(response) -> Dict:
        if isinstance(response, dict):
            return response
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        raise TypeError('Unexpected completions response type: '
                        f'{type(response).__name__}')

    @staticmethod
    def _choice_field(payload: Dict, name: str):
        choices = payload.get('choices') or []
        if not choices:
            raise RuntimeError('vLLM completions response has no choices')
        choice = choices[0]
        if not isinstance(choice, dict) and hasattr(choice, 'model_dump'):
            choice = choice.model_dump()
        value = choice.get(name) if isinstance(choice, dict) else None
        return value if value is not None else payload.get(name)

    def _request_prompt_data(
            self, prompt: str,
            with_logprobs: bool) -> Tuple[List[int], Optional[List]]:
        """Return server-tokenized prompt IDs and optional token logprobs."""
        extra_body = dict(self.completion_extra_body)
        extra_body['return_token_ids'] = True
        if with_logprobs:
            # In vLLM, zero returns the actual prompt token only. The actual
            # token is sufficient and avoids transferring unused top-k data.
            extra_body['prompt_logprobs'] = 0

        query_data = dict(model=self.path,
                          prompt=prompt,
                          max_tokens=1,
                          temperature=0,
                          extra_body=extra_body)

        for attempt in range(self.retry):
            self.acquire()
            try:
                response = self.openai_client.completions.create(
                    **query_data, timeout=self.timeout)
                payload = self._response_dict(response)
                prompt_token_ids = self._choice_field(payload,
                                                      'prompt_token_ids')
                if prompt_token_ids is None:
                    raise RuntimeError(
                        'vLLM did not return prompt_token_ids. Confirm that '
                        'the server supports return_token_ids=true.')

                prompt_logprobs = None
                if with_logprobs:
                    prompt_logprobs = self._choice_field(
                        payload, 'prompt_logprobs')
                    if prompt_logprobs is None:
                        raise RuntimeError(
                            'vLLM did not return prompt_logprobs. Confirm '
                            'that prompt_logprobs=0 is enabled by the server.')
                    if len(prompt_logprobs) != len(prompt_token_ids):
                        raise RuntimeError(
                            'vLLM returned different prompt_token_ids and '
                            'prompt_logprobs lengths: '
                            f'{len(prompt_token_ids)} != '
                            f'{len(prompt_logprobs)}')
                return list(prompt_token_ids), prompt_logprobs
            except Exception as error:
                self.logger.error('vLLM completions request failed '
                                  f'(attempt {attempt + 1}/{self.retry}): '
                                  f'{error}')
                if attempt + 1 == self.retry:
                    raise
                # A reverse proxy can briefly close many concurrent raw
                # completions connections at once.  Retrying immediately
                # makes every worker hit the same outage window and can burn
                # through the full retry budget within one second.  Keep the
                # delay short for normal transient failures, but back off
                # enough for the endpoint to recover.
                sleep(min(0.25 * (2**attempt), 4.0))
            finally:
                self.release()
        raise RuntimeError('vLLM completions request failed')

    def _get_context_token_ids(self, context: str) -> Tuple[int, ...]:
        with self._prompt_token_ids_cache_lock:
            cached = self._prompt_token_ids_cache.get(context)
        if cached is not None:
            return cached

        token_ids, _ = self._request_prompt_data(context, with_logprobs=False)
        result = tuple(token_ids)
        with self._prompt_token_ids_cache_lock:
            self._prompt_token_ids_cache.setdefault(context, result)
            return self._prompt_token_ids_cache[context]

    @staticmethod
    def _actual_token_logprob(entry, token_id: int) -> float:
        if entry is None:
            raise RuntimeError(
                'The continuation includes a prompt token without a '
                'logprob. The first prompt token is normally null, but a '
                'continuation token must not be null.')
        if not isinstance(entry, dict) and hasattr(entry, 'model_dump'):
            entry = entry.model_dump()
        if not isinstance(entry, dict):
            raise TypeError('Unexpected prompt_logprobs entry type: '
                            f'{type(entry).__name__}')

        token_data = entry.get(str(token_id), entry.get(token_id))
        if token_data is None:
            raise RuntimeError(
                f'vLLM omitted the actual prompt token {token_id} from its '
                'prompt_logprobs entry')
        if not isinstance(token_data, dict) and hasattr(
                token_data, 'model_dump'):
            token_data = token_data.model_dump()
        if isinstance(token_data, dict):
            token_data = token_data.get('logprob')
        if token_data is None:
            raise RuntimeError(
                f'vLLM returned no logprob for prompt token {token_id}')
        return float(token_data)

    def _score_continuation(self, text: str, continuation: str) -> float:
        if not continuation:
            raise ValueError('continuation must not be empty')
        if not text.endswith(continuation):
            raise ValueError(
                f'Input does not end with continuation {continuation!r}')

        context = text[:-len(continuation)]
        context_token_ids = self._get_context_token_ids(context)
        full_token_ids, prompt_logprobs = self._request_prompt_data(
            text, with_logprobs=True)
        continuation_start = len(context_token_ids)
        if continuation_start >= len(full_token_ids):
            raise RuntimeError(
                'The continuation produced no separately scoreable token. '
                f'Context tokens: {len(context_token_ids)}, full prompt '
                f'tokens: {len(full_token_ids)}')

        # This split is intentionally based on the server's separately
        # tokenized context length. It is the same context/continuation split
        # used by lm-evaluation-harness' encode-pair path.
        return sum(
            self._actual_token_logprob(prompt_logprobs[index],
                                       full_token_ids[index])
            for index in range(continuation_start, len(full_token_ids)))

    def get_loglikelihood(self, inputs: List[str],
                          conts: List[str]) -> np.ndarray:
        """Sum conditional logprobs of each continuation, without averaging."""
        if len(inputs) != len(conts):
            raise ValueError('inputs and conts must have the same length')
        if not inputs:
            return np.array([], dtype=float)

        if len(inputs) == 1:
            scores = [self._score_continuation(inputs[0], conts[0])]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                scores = list(
                    executor.map(self._score_continuation, inputs, conts))
        return np.asarray(scores, dtype=float)
