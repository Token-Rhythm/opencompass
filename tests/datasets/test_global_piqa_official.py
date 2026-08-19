from datasets import Dataset

from opencompass.configs.datasets.global_piqa.global_piqa_generation import (
    global_piqa_datasets,
)
from opencompass.datasets.global_piqa import (
    GLOBAL_PIQA_SOURCES,
    GlobalPIQADataset,
    GlobalPIQAEvaluator,
    _format_global_piqa,
)
from opencompass.openicl.icl_inferencer import ParallelGenInferencer


def test_global_piqa_prompts_match_lm_eval_generation_templates():
    nonparallel = _format_global_piqa(
        dict(prompt='A situation', solution0='first', solution1='second',
             label=1), 'nonparallel', 'eng_latn')
    assert nonparallel['question_prompt'] == (
        'Given the following situation, which option is more likely to be '
        'correct?\n\nSituation:\nA situation ...\n\nOption A: first\n\n'
        'Option B: second\n\nYour response should end with "The best '
        'answer is: [answer_letter]" where [answer_letter] is one of A or B.')
    assert nonparallel['answer_letter'] == 'B'

    parallel = _format_global_piqa(
        dict(prompt='A situation', solution0='first', solution1='second',
             solution2='third', solution3='fourth', label=2), 'parallel',
        'eng_latn')
    assert parallel['question_prompt'] == (
        'A situation\n\nOption A: first\n\nOption B: second\n\nOption C: '
        'third\n\nOption D: fourth\n\nYour response should end with "The best '
        'answer is: [answer_letter]" where [answer_letter] is one of A, B, C, '
        'or D.')
    assert parallel['answer_letter'] == 'C'


def test_global_piqa_sources_are_revision_pinned():
    assert set(GLOBAL_PIQA_SOURCES) == {'nonparallel', 'parallel'}
    assert all(len(revision) == 40
               for _, revision in GLOBAL_PIQA_SOURCES.values())


def test_global_piqa_can_sample_every_language_config(monkeypatch):
    import opencompass.datasets.global_piqa as module

    monkeypatch.setattr(module, 'get_dataset_config_names',
                        lambda path, revision: ['lang_a', 'lang_b'])

    def fake_load_dataset(path, language, revision, split):
        del path, language, revision, split
        return Dataset.from_list([
            dict(prompt='p0', solution0='a', solution1='b', solution2='c',
                 solution3='d', label=0),
            dict(prompt='p1', solution0='a', solution1='b', solution2='c',
                 solution3='d', label=1),
        ])

    monkeypatch.setattr(module, 'load_dataset', fake_load_dataset)
    sources = {
        'nonparallel': ('nonparallel-path', 'revision-a'),
        'parallel': ('parallel-path', 'revision-b'),
    }
    dataset = GlobalPIQADataset.load(sources=sources, samples_per_config=1)

    assert len(dataset['test']) == 4
    assert set(dataset['test']['variant']) == {'nonparallel', 'parallel'}
    assert set(dataset['test']['language_config']) == {'lang_a', 'lang_b'}


def test_global_piqa_rejects_invalid_per_config_sample_count():
    for value in (0, -1, True, 1.5):
        try:
            GlobalPIQADataset.load(sources={}, samples_per_config=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{value!r} should be rejected')


def test_global_piqa_has_no_hidden_sampling_overrides():
    inferencer = global_piqa_datasets[0]['infer_cfg']['inferencer']
    assert 'generation_kwargs' not in inferencer
    assert inferencer['type'] is ParallelGenInferencer
    assert inferencer['save_every'] == 1


def test_global_piqa_evaluator_rejects_length_mismatch():
    result = GlobalPIQAEvaluator().score([], ['A'], [])
    assert result == {
        'error': 'predictions, references, and test_set have different lengths'
    }


def test_global_piqa_evaluator_matches_two_level_unweighted_macro():
    # nonparallel language 1: 100%; nonparallel language 2: 0% => 50%
    # parallel language 1: 100% => 100%; final variant macro => 75%.
    rows = [
        dict(variant='nonparallel', language_config='l1'),
        dict(variant='nonparallel', language_config='l2'),
        dict(variant='parallel', language_config='l3'),
    ]
    predictions = [
        'The best answer is: A',
        'The best answer is: A',
        'Reasoning says A. The final answer is: D',
    ]
    result = GlobalPIQAEvaluator().score(predictions, ['A', 'B', 'D'], rows)
    assert result['nonparallel_accuracy'] == 50
    assert result['parallel_accuracy'] == 100
    assert result['accuracy'] == 75
