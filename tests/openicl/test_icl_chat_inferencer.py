"""Tests for ChatInferencer parser isolation."""

from unittest.mock import MagicMock

import pytest

from opencompass.models.base_api import APITemplateParser
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
