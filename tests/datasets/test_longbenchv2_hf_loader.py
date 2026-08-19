from datasets import Dataset

from opencompass.datasets import longbenchv2
from opencompass.configs.datasets.longbenchv2.longbenchv2_gen_bd9437 import (
    LongBenchv2_reader_cfg,
)


def test_longbenchv2_hf_loader_uses_pinned_split(monkeypatch):
    source = Dataset.from_list([{
        '_id': 'sample',
        'context': 'context',
        'question': 'question',
        'choice_A': 'a',
        'choice_B': 'b',
        'choice_C': 'c',
        'choice_D': 'd',
        'answer': 'A',
        'difficulty': 'easy',
        'length': 'short',
    }])
    called = {}

    def fake_load_dataset(path, split, revision):
        called.update(path=path, split=split, revision=revision)
        return source

    monkeypatch.setattr(longbenchv2, 'load_dataset', fake_load_dataset)
    loaded = longbenchv2.LongBenchv2Dataset.load(
        'zai-org/LongBench-v2', hf_revision='pinned')

    assert called == {
        'path': 'zai-org/LongBench-v2',
        'split': 'train',
        'revision': 'pinned',
    }
    assert len(loaded['test']) == 1
    assert loaded['test'][0]['answer'] == 'A'


def test_longbenchv2_reader_uses_test_for_both_splits():
    assert LongBenchv2_reader_cfg['train_split'] == 'test'
    assert LongBenchv2_reader_cfg['test_split'] == 'test'


def test_longbenchv2_uses_official_answer_patterns():
    extract = longbenchv2.longbenchv2_answer_postprocess
    assert extract('Reasoning. The correct answer is **(C)**') == 'C'
    assert extract('The correct answer is D') == 'D'
    assert extract('I considered A and B but did not give the final form.') == ''


def test_longbenchv2_evaluator_rejects_length_mismatch():
    result = longbenchv2.LongBenchv2Evaluator().score([], ['A'], [])
    assert result == {
        'error': 'predictions, references, and test_set have different lengths'
    }
