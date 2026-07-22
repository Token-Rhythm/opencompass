"""Tests for vLLM's OpenAI-compatible continuation scorer."""

from unittest.mock import Mock

import numpy as np
import pytest

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


def test_raw_generation_rejects_chat_prompt():
    model = VLLMOpenAIAPI.__new__(VLLMOpenAIAPI)
    model.generation_endpoint = 'completions'
    with pytest.raises(TypeError, match='requires a string prompt'):
        model._generate([{'role': 'user', 'prompt': 'hello'}], 10, 0.0)
