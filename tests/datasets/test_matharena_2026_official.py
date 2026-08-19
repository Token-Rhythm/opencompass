import pytest
from datasets import Dataset, DatasetDict
from mmengine.config import Config

import opencompass.datasets.hmmt_2025 as matharena_module
from opencompass.datasets.hmmt_2025 import (MATHARENA_2026_DATASETS,
                                             MathArena2026Dataset,
                                             MathArenaEvaluator)
from opencompass.openicl.icl_inferencer import ParallelGenInferencer


@pytest.mark.parametrize('competition', ['aime_2026', 'hmmt_feb_2026'])
def test_matharena_2026_loader_pins_official_revision(monkeypatch,
                                                       competition):
    calls = []

    def fake_load_dataset(path, revision):
        calls.append((path, revision))
        return DatasetDict({
            'train':
            Dataset.from_list([{
                'problem': 'Compute 1+1.',
                'answer': 2,
            }])
        })

    monkeypatch.setattr(matharena_module, 'load_dataset', fake_load_dataset)
    dataset = MathArena2026Dataset.load(competition)

    assert calls == [MATHARENA_2026_DATASETS[competition]]
    assert len(dataset['test']) == 1
    assert dataset['test'][0] == {
        'problem': 'Compute 1+1.',
        # datasets preserves the source column's integer schema for AIME.
        # MathArenaEvaluator normalizes references with str() before parsing.
        'answer': 2,
        'prompt': ('Put your final answer within \\boxed{}.\n\n'
                   'Compute 1+1.'),
    }


def test_matharena_2026_loader_rejects_unknown_competition():
    with pytest.raises(ValueError, match='aime_2026'):
        MathArena2026Dataset.load('unknown')


def test_matharena_2026_loader_repeats_each_official_run(monkeypatch):
    source = Dataset.from_list([
        {'problem': 'First problem.', 'answer': 1},
        {'problem': 'Second problem.', 'answer': 2},
    ])
    monkeypatch.setattr(
        matharena_module,
        'load_dataset',
        lambda path, revision: DatasetDict({'train': source}),
    )

    dataset = MathArena2026Dataset.load('aime_2026', num_repeats=4)

    assert len(dataset['test']) == 8
    assert dataset['test']['answer'] == [1, 2, 1, 2, 1, 2, 1, 2]


@pytest.mark.parametrize(
    'relative_path,competition',
    [
        ('aime2026/aime2026_gen.py', 'aime_2026'),
        ('hmmt2026/hmmt2026_gen.py', 'hmmt_feb_2026'),
    ],
)
def test_matharena_2026_configs_use_official_parser_and_chat_prompt(
        relative_path, competition):
    config = Config.fromfile('opencompass/configs/datasets/' + relative_path)
    dataset = (config.aime2026_datasets[0]
               if competition == 'aime_2026' else
               config.hmmt2026_datasets[0])

    assert dataset['type'] is MathArena2026Dataset
    assert dataset['competition'] == competition
    assert dataset['num_repeats'] == 4
    assert dataset['eval_cfg']['evaluator']['type'] is MathArenaEvaluator
    assert dataset['infer_cfg']['prompt_template']['messages'] == [
        dict(role='user', content='{prompt}')
    ]
    assert dataset['infer_cfg']['inferencer'] == {
        'type': ParallelGenInferencer,
        'save_every': 1,
    }
