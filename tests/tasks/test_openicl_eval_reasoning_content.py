"""Tests for separating reasoning from answer evaluation."""

from opencompass.tasks.openicl_eval import extract_prediction_content


def test_evaluation_prefers_top_level_content():
    record = {
        'prediction': 'reasoning that says A',
        'reasoning_content': 'reasoning that says A',
        'content': 'B',
    }

    assert extract_prediction_content(record) == 'B'


def test_evaluation_supports_legacy_and_nested_predictions():
    assert extract_prediction_content({'prediction': 'C'}) == 'C'
    assert extract_prediction_content(
        {'prediction': {
            'reasoning_content': 'reasoning',
            'content': 'D',
        }}) == 'D'
