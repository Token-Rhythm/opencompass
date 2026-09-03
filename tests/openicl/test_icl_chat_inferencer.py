"""Tests for ChatInferencer parser isolation."""

from unittest.mock import MagicMock

import pytest

from opencompass.models.base_api import APITemplateParser
from opencompass.openicl.icl_inferencer.icl_base_inferencer import \
    ChatOutputHandler
from opencompass.openicl.icl_inferencer.icl_chat_inferencer import (
    ChatInferencer, )


def test_chat_inferencer_restores_shared_model_parser_after_success():
    model = MagicMock()
    model.is_api = True
    original = APITemplateParser(None)
    model.template_parser = original
    inferencer = ChatInferencer(model=model)

    def check_active_parser(*args):
        assert model.template_parser is not original
        return {'0': {'prediction': 'ok'}}

    inferencer._inference = MagicMock(side_effect=check_active_parser)
    inferencer.inference(MagicMock())

    assert model.template_parser is original


def test_chat_inferencer_restores_shared_model_parser_after_error():
    model = MagicMock()
    model.is_api = True
    original = APITemplateParser(None)
    model.template_parser = original
    inferencer = ChatInferencer(model=model)
    inferencer._inference = MagicMock(side_effect=RuntimeError('failed'))

    with pytest.raises(RuntimeError, match='failed'):
        inferencer.inference(MagicMock())

    assert model.template_parser is original


def test_infer_every_normalizes_structured_output_and_preserves_gold():
    model = MagicMock()
    model.is_api = True
    model.template_parser = APITemplateParser(None)
    model.generate_from_template.side_effect = [
        [{
            'reasoning_content': 'reason one',
            'content': 'answer one',
        }],
        [{
            'reasoning_content': 'reason two',
            'content': 'answer two',
        }],
    ]
    inferencer = ChatInferencer(model=model, infer_mode='every')
    handler = ChatOutputHandler()
    chat = [
        {'role': 'user', 'content': 'question one'},
        {'role': 'assistant', 'content': 'gold one'},
        {'role': 'user', 'content': 'question two'},
        {'role': 'assistant', 'content': 'gold two'},
    ]

    inferencer.infer_every(chat, 7, handler)

    second_history = model.generate_from_template.call_args_list[1].args[0][0]
    assert second_history[1]['content'] == 'answer one'
    assert isinstance(second_history[1]['content'], str)
    assert handler.results_dict['7']['prediction'] == [
        'answer one', 'answer two'
    ]
    assert handler.results_dict['7']['reasoning_content'] == [
        'reason one', 'reason two'
    ]
    assert handler.results_dict['7']['gold'] == ['gold one', 'gold two']


def test_infer_every_preserves_structured_inference_errors_by_round():
    model = MagicMock()
    model.is_api = True
    model.template_parser = APITemplateParser(None)
    model.generate_from_template.side_effect = [
        [{
            'reasoning_content': '',
            'content': '',
            'inference_error': 'first round failed',
        }],
        [{
            'reasoning_content': '',
            'content': 'recovered',
        }],
    ]
    inferencer = ChatInferencer(model=model, infer_mode='every')
    handler = ChatOutputHandler()
    chat = [
        {'role': 'user', 'content': 'question one'},
        {'role': 'assistant', 'content': 'gold one'},
        {'role': 'user', 'content': 'question two'},
        {'role': 'assistant', 'content': 'gold two'},
    ]

    inferencer.infer_every(chat, 3, handler)

    assert handler.results_dict['3']['prediction'] == ['', 'recovered']
    assert handler.results_dict['3']['inference_error'] == [
        'first round failed', None
    ]
