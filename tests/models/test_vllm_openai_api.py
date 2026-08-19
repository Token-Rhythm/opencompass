"""Tests for vLLM's OpenAI-compatible continuation scorer."""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opencompass.models.openai_api import OpenAISDK
from opencompass.models.vllm_openai_api import VLLMOpenAIAPI


def _logprob_entry(token_id, logprob):
    return {str(token_id): {'logprob': logprob}}


def test_score_continuation_sums_only_suffix_tokens():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model._get_context_token_ids = Mock(return_value=(1, 2, 3))
    model._request_prompt_data = Mock(return_value=(
        [1, 2, 3, 40, 41],
        [
            None,
            _logprob_entry(2, -9),
            _logprob_entry(3, -8),
            _logprob_entry(40, -0.25),
            _logprob_entry(41, -0.75)
        ],
    ))

    score = model._score_continuation('Question\nAnswer: AB', ' AB')

    assert score == pytest.approx(-1.0)
    model._get_context_token_ids.assert_called_once_with('Question\nAnswer:')


def test_score_continuation_uses_actual_token_id_not_top_one():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model._get_context_token_ids = Mock(return_value=(1, ))
    model._request_prompt_data = Mock(return_value=(
        [1, 5594],
        [
            None, {
                '3710': {
                    'logprob': -2.0,
                    'rank': 1
                },
                '5594': {
                    'logprob': -11.5,
                    'rank': 2
                }
            }
        ],
    ))

    assert model._score_continuation('prompt A', ' A') == -11.5


def test_get_loglikelihood_returns_raw_scores_without_normalizing():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.max_workers = 2
    model._score_continuation = Mock(side_effect=[-0.1, -3.0])

    scores = model.get_loglikelihood(['p A', 'p BC'], [' A', ' BC'])

    np.testing.assert_array_equal(scores, [-0.1, -3.0])


def test_generate_records_failed_item_and_continues_batch():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.temperature = None
    model.max_workers = 2
    model.logger = Mock()

    def generate_one(value, *args):
        if value == 'bad':
            raise TimeoutError('stream timed out')
        return 'good answer'

    model._generate = Mock(side_effect=generate_one)

    results = model.generate(['bad', 'good'], max_out_len=32)

    assert results[0] == {
        'reasoning_content': '',
        'content': '',
        'inference_error': 'TimeoutError: stream timed out',
    }
    assert results[1] == 'good answer'
    model.logger.exception.assert_called_once()


def test_generate_records_failed_single_item():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.temperature = None
    model.logger = Mock()
    model._generate = Mock(side_effect=TimeoutError('stream timed out'))

    results = model.generate(['bad'], max_out_len=32)

    assert results[0]['content'] == ''
    assert results[0]['inference_error'] == 'TimeoutError: stream timed out'


def test_prompt_data_retries_with_backoff_after_transient_failure():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.path = 'served-model'
    model.completion_extra_body = {}
    model.retry = 2
    model.timeout = 30
    model.acquire = Mock()
    model.release = Mock()
    model.logger = Mock()
    response = Mock()
    response.model_dump.return_value = {
        'choices': [{
            'prompt_token_ids': [1, 2, 3],
        }]
    }
    create = Mock(side_effect=[RuntimeError('proxy disconnected'), response])
    model.openai_client = Mock(completions=Mock(create=create))

    with patch('opencompass.models.vllm_openai_api.sleep') as retry_sleep:
        token_ids, logprobs = model._request_prompt_data(
            'prompt', with_logprobs=False)

    assert token_ids == [1, 2, 3]
    assert logprobs is None
    assert create.call_count == 2
    retry_sleep.assert_called_once_with(0.25)
    assert model.acquire.call_count == 2
    assert model.release.call_count == 2


def test_score_continuation_rejects_wrong_boundary():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    with pytest.raises(ValueError, match='does not end with continuation'):
        model._score_continuation('prompt A', ' B')


def test_raw_generation_uses_completions_without_chat_messages():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.generation_endpoint = 'completions'
    model.path = 'served-model'
    model.extra_body = {'top_k': 20}
    model.openai_extra_kwargs = {'top_p': 0.95}
    model.retry = 1
    model.timeout = 30
    model.acquire = Mock()
    model.release = Mock()
    model.logger = Mock()
    choice = Mock(text='raw answer')
    create = Mock(return_value=Mock(choices=[choice]))
    model.openai_client = Mock(completions=Mock(create=create))

    result = model._generate('raw prompt', 128, 0.0)

    assert result == 'raw answer'
    create.assert_called_once_with(model='served-model',
                                   prompt='raw prompt',
                                   max_tokens=128,
                                   n=1,
                                   temperature=0.0,
                                   extra_body={'top_k': 20},
                                   top_p=0.95,
                                   timeout=30)
    assert 'messages' not in create.call_args.kwargs


def test_raw_generation_forwards_official_stop_sequences():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.generation_endpoint = 'completions'
    model.path = 'served-model'
    model.extra_body = {}
    model.openai_extra_kwargs = None
    model.retry = 1
    model.timeout = 30
    model.acquire = Mock()
    model.release = Mock()
    model.logger = Mock()
    create = Mock(return_value=Mock(choices=[Mock(text='B')]))
    model.openai_client = Mock(completions=Mock(create=create))

    stops = ['</s>', 'Q:', 'Question:', '<|im_end|>']
    assert model._generate('prompt', 32, 0.0, stops) == 'B'
    assert create.call_args.kwargs['stop'] == stops


def test_raw_generation_rejects_chat_prompt():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.generation_endpoint = 'completions'
    with pytest.raises(TypeError, match='role/content fields'):
        model._generate([{'role': 'user', 'prompt': 'hello'}], 10, 0.0)


def test_raw_generation_flattens_message_contents_without_chat_template():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.generation_endpoint = 'completions'
    model.meta_template = None
    model.path = 'served-model'
    model.extra_body = {}
    model.openai_extra_kwargs = None
    model.retry = 1
    model.timeout = 30
    model.acquire = Mock()
    model.release = Mock()
    model.logger = Mock()
    create = Mock(return_value=Mock(choices=[Mock(text='B')]))
    model.openai_client = Mock(completions=Mock(create=create))

    messages = [
        {
            'role': 'system',
            'content': 'Subject description.\n\n'
        },
        {
            'role': 'user',
            'content': 'Question and choices'
        },
    ]
    result = model._generate(messages, 8, 0.0)

    assert result == 'B'
    assert create.call_args.kwargs['prompt'] == (
        'Subject description.\n\nQuestion and choices')
    assert 'messages' not in create.call_args.kwargs


def test_raw_generation_rejects_messages_with_meta_template():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.generation_endpoint = 'completions'
    model.meta_template = {'round': []}

    with pytest.raises(ValueError, match='requires meta_template=None'):
        model._generate([{'role': 'user', 'content': 'hello'}], 10, 0.0)


def test_chat_generation_preserves_system_and_user_messages():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.generation_endpoint = 'chat'
    messages = [
        {
            'role': 'system',
            'content': 'Subject description.\n\n'
        },
        {
            'role': 'user',
            'content': 'Question and choices'
        },
    ]

    with patch('opencompass.models.vllm_openai_api.OpenAISDK._generate',
               return_value='B') as generate:
        result = model._generate(messages, 8, 0.0)

    assert result == 'B'
    generate.assert_called_once_with(messages, 8, 0.0, None)


def test_streaming_chat_keeps_reasoning_separate_from_final_content():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.generation_endpoint = 'chat'
    model.stream_chat = True
    model.path = 'served-model'
    model.max_seq_len = 4096
    model.mode = 'none'
    model.get_token_len = Mock(return_value=10)
    model.extra_body = {'top_k': 20}
    model.openai_extra_kwargs = {'top_p': 0.95}
    model.retry = 1
    model.timeout = 30
    model.stream_idle_timeout = 600
    model.status_code_mappings = {}
    model.acquire = Mock()
    model.release = Mock()
    model.logger = Mock()

    chunks = [
        Mock(choices=[Mock(delta=Mock(content='', reasoning_content=None,
                                     reasoning='Think'),
                          finish_reason=None)]),
        Mock(choices=[Mock(delta=Mock(content='', reasoning_content=None,
                                     reasoning=' first'),
                          finish_reason=None)]),
        Mock(choices=[Mock(delta=Mock(content='The answer ',
                                     reasoning_content=None, reasoning=None),
                          finish_reason=None)]),
        Mock(choices=[Mock(delta=Mock(content='is (B).',
                                     reasoning_content=None, reasoning=None),
                          finish_reason='stop')]),
    ]
    response_stream = Mock()
    response_stream.__iter__ = Mock(return_value=iter(chunks))
    create = Mock(return_value=response_stream)
    model.openai_client = Mock(chat=Mock(completions=Mock(create=create)))

    result = model._generate([{
        'role': 'user',
        'content': 'Question'
    }], 128, 1.0, ['Question:'])

    assert result == {
        'reasoning_content': 'Think first',
        'content': 'The answer is (B).',
    }
    assert create.call_args.kwargs['stream'] is True
    assert create.call_args.kwargs['timeout'] == 600
    assert create.call_args.kwargs['stop'] == ['Question:']
    assert create.call_args.kwargs['extra_body'] == {'top_k': 20}
    assert create.call_args.kwargs['top_p'] == 0.95
    response_stream.close.assert_called_once_with()
    model.acquire.assert_called_once_with()
    model.release.assert_called_once_with()


def test_streaming_chat_preserves_received_text_when_terminal_event_is_lost():
    import httpx
    from openai import APITimeoutError

    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.path = 'served-model'
    model.max_seq_len = 4096
    model.mode = 'none'
    model.get_token_len = Mock(return_value=10)
    model.extra_body = {}
    model.openai_extra_kwargs = None
    model.retry = 1
    model.timeout = 7200
    model.stream_idle_timeout = 600
    model.status_code_mappings = {}
    model.acquire = Mock()
    model.release = Mock()
    model.logger = Mock()

    chunks = [
        Mock(choices=[Mock(delta=Mock(content='',
                                     reasoning_content='partial reasoning',
                                     reasoning=None),
                          finish_reason=None)]),
        Mock(choices=[Mock(delta=Mock(content='partial final',
                                     reasoning_content=None,
                                     reasoning=None),
                          finish_reason=None)]),
    ]

    def interrupted_stream():
        yield from chunks
        raise APITimeoutError(request=httpx.Request(
            'POST', 'http://endpoint/v1/chat/completions'))

    response_stream = Mock()
    response_stream.__iter__ = Mock(side_effect=interrupted_stream)
    create = Mock(return_value=response_stream)
    model.openai_client = Mock(chat=Mock(completions=Mock(create=create)))

    result = model._generate_chat_stream([{
        'role': 'user',
        'content': 'Question'
    }], 128, 1.0)

    assert result == {
        'reasoning_content': 'partial reasoning',
        'content': 'partial final',
    }
    assert create.call_args.kwargs['timeout'] == 600
    model.logger.warning.assert_called_once()
    response_stream.close.assert_called_once_with()
    model.release.assert_called_once_with()


def test_chat_generation_forwards_official_stop_sequences():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.generation_endpoint = 'chat'
    messages = [{'role': 'user', 'content': 'Question'}]
    stops = ['</s>', 'Question:']

    with patch('opencompass.models.vllm_openai_api.OpenAISDK._generate',
               return_value='B') as generate:
        assert model._generate(messages, 8, 0.0, stops) == 'B'

    generate.assert_called_once_with(messages, 8, 0.0, stops)


def test_openai_sdk_chat_request_contains_stop_sequences():
    model = OpenAISDK.__new__(OpenAISDK)
    model.path = 'served-model'
    model.max_seq_len = 4096
    model.mode = 'none'
    model.get_token_len = Mock(return_value=10)
    model.retry = 1
    model.temperature = None
    model.extra_body = {}
    model.openai_extra_kwargs = None
    model.timeout = 30
    model.verbose = False
    model.status_code_mappings = {}
    model.think_tag = '</think>'
    model.acquire = Mock()
    model.release = Mock()
    model.logger = Mock()
    message = Mock(content='B', reasoning_content=None, reasoning=None)
    response = Mock(choices=[Mock(message=message, finish_reason='stop')])
    create = Mock(return_value=response)
    model.openai_client = Mock(chat=Mock(completions=Mock(create=create)))

    stops = ['</s>', 'Question:']
    assert model._generate([{
        'role': 'user',
        'content': 'Question'
    }], 8, 0.0, stops) == {
        'reasoning_content': '',
        'content': 'B'
    }
    assert create.call_args.kwargs['stop'] == stops
    assert create.call_args.kwargs['temperature'] == 0.0


def test_generate_exposes_stopping_criteria_to_gen_inferencer():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.temperature = 0.0
    model._generate = Mock(return_value='B')

    stops = ['</s>', 'Question:']
    assert model.generate(['prompt'], 16, stopping_criteria=stops) == ['B']
    model._generate.assert_called_once_with('prompt', 16, 0.0, stops)


def test_parallel_generate_collects_as_completed_but_preserves_input_order():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.temperature = 0.0
    model.max_workers = 2

    def generate(prompt, *_args):
        if prompt == 'slow':
            time.sleep(0.02)
        return prompt.upper()

    model._generate = generate

    assert model.generate(['slow', 'fast'], 16) == ['SLOW', 'FAST']
