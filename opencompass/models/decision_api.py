"""Decision benchmarks: preserve transport errors and per-task schemas."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from opencompass.datasets._decision_upstream.jevbench.adapters.openai_compat import OpenAICompatAdapter
from opencompass.registry import MODELS

from .vllm_openai_api import VLLMOpenAIAPI


@MODELS.register_module()
class DecisionOpenAIAPI(VLLMOpenAIAPI):
    """Do not turn failed HTTP requests into scored empty answers."""

    def generate(self, inputs, max_out_len=4096, temperature=0,
                 stopping_criteria=None, **kwargs):
        if self.temperature is not None:
            temperature = self.temperature
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._generate, item, max_out_len,
                                       temperature, stopping_criteria or None)
                       for item in inputs]
            return [future.result() for future in futures]


@MODELS.register_module()
class JevBenchOpenAIAPI(DecisionOpenAIAPI):
    """Apply the upstream strict schema without sharing per-request state."""

    @staticmethod
    def response_format(messages):
        # Native prompts end with a separate, canonical label-list line.
        line = messages[-1]['content'].splitlines()[-1]
        prefixes = ('Output probabilities over exactly these keys: ',
                    'Rate the state. Output probabilities over the level indices: ')
        prefix = next((p for p in prefixes if line.startswith(p)), None)
        if prefix is None or not line.endswith('.'):
            raise ValueError('Expected an unmodified upstream JevBench prompt')
        labels = json.loads(line[len(prefix):-1])
        if not isinstance(labels, list) or not labels or any(
                not isinstance(label, str) for label in labels):
            raise ValueError('Invalid JevBench label list')
        task = SimpleNamespace(labels=labels, state='',
                               question={'type': 'choice', 'instructions': ''})
        return OpenAICompatAdapter('', '').build_request(task)['response_format']

    def _generate(self, input, max_out_len, temperature, stopping_criteria=None):
        if self.generation_endpoint != 'chat':
            raise ValueError('JevBench native adapter requires chat completions')
        messages, _ = self._preprocess_messages(
            input, max_out_len, self.max_seq_len, self.mode, self.get_token_len)
        local = copy.copy(self)
        local.openai_extra_kwargs = dict(self.openai_extra_kwargs or {})
        local.openai_extra_kwargs['response_format'] = self.response_format(messages)
        return super(JevBenchOpenAIAPI, local)._generate(
            input, max_out_len, temperature, stopping_criteria)
