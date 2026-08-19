from datasets import Dataset
from mmengine.config import Config

from opencompass.datasets.supergpqa import supergpqa


def test_supergpqa_uses_rolling_parallel_inference():
    config = Config.fromfile(
        'opencompass/configs/datasets/supergpqa/supergpqa_gen.py')
    inferencer = config.supergpqa_datasets[0]['infer_cfg']['inferencer']

    assert inferencer['type'].__name__ == 'ParallelGenInferencer'
    assert inferencer['save_every'] == 1


def test_supergpqa_loader_pins_hf_revision(monkeypatch):
    source = Dataset.from_list([{
        'question': 'Question?',
        'options': ['one', 'two'],
        'answer_letter': 'A',
        'answer': 'one',
        'discipline': 'Mathematics',
        'field': 'field',
        'subfield': 'subfield',
        'difficulty': 'easy',
    }])
    called = {}

    def fake_load_dataset(path, split, revision):
        called.update(path=path, split=split, revision=revision)
        return source

    monkeypatch.setattr(supergpqa, 'load_dataset', fake_load_dataset)
    loaded = supergpqa.SuperGPQADataset.load(
        path='m-a-p/SuperGPQA',
        prompt_mode='zero-shot',
        hf_revision='pinned',
    )

    assert called == {
        'path': 'm-a-p/SuperGPQA',
        'split': 'train',
        'revision': 'pinned',
    }
    assert len(loaded) == 1
    assert loaded[0]['infer_prompt'].endswith('A) one\nB) two\n')


def test_supergpqa_loader_samples_every_discipline_difficulty(monkeypatch):
    rows = []
    for discipline in ('Science', 'Law'):
        for difficulty in ('easy', 'hard'):
            for index in range(3):
                rows.append({
                    'question': f'{discipline}-{difficulty}-{index}',
                    'options': ['one', 'two'],
                    'answer_letter': 'A',
                    'answer': 'one',
                    'discipline': discipline,
                    'field': 'field',
                    'subfield': 'subfield',
                    'difficulty': difficulty,
                })
    source = Dataset.from_list(rows)
    monkeypatch.setattr(supergpqa, 'load_dataset',
                        lambda path, split, revision: source)

    loaded = supergpqa.SuperGPQADataset.load(
        path='m-a-p/SuperGPQA',
        prompt_mode='zero-shot',
        hf_revision='pinned',
        samples_per_discipline_difficulty=2,
    )

    assert len(loaded) == 8
    counts = {}
    for row in loaded:
        key = (row['discipline'], row['difficulty'])
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {2}


def test_supergpqa_metrics_are_percentages():
    samples = [{
        'prompt_mode': 'zero-shot',
        'options': ['one', 'two'],
        'answer_letter': 'A',
        'answer': 'one',
        'discipline': 'Mathematics',
        'field': 'field',
        'subfield': 'subfield',
        'difficulty': 'easy',
    }, {
        'prompt_mode': 'zero-shot',
        'options': ['one', 'two'],
        'answer_letter': 'B',
        'answer': 'two',
        'discipline': 'Mathematics',
        'field': 'field',
        'subfield': 'subfield',
        'difficulty': 'easy',
    }]
    scores = supergpqa.SuperGPQAEvaluator().score(
        ['Answer: A', 'no final answer'],
        ['A', 'B'],
        samples,
    )

    assert scores['accuracy'] == 50
    assert scores['easy_accuracy'] == 50
    assert scores['miss_rate'] == 50
    assert scores['error_rate'] == 0


def test_supergpqa_rejects_misaligned_inputs():
    assert supergpqa.SuperGPQAEvaluator().score([], ['A'], [{}]) == {
        'error': 'predictions, references, and test_set have different lengths'
    }
