"""Protocol-level tests for the phase-one language benchmark adapters."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from datasets import Dataset
from mmengine import Config

from opencompass.datasets.global_piqa import (GlobalPIQAEvaluator,
                                              _format_global_piqa)
from opencompass.datasets.hmmt_2025 import MathArenaEvaluator, _format_hmmt
from opencompass.datasets.include import _format_include
from opencompass.datasets.mmlu_prox import (MMLUProXEvaluator, _format_example)
from opencompass.datasets.mmlu_prox_lang_libs import LANG_LIBS
from opencompass.datasets.mmlu_redux import (MMLUReduxEvaluator,
                                             _format_mmlu_redux)
from opencompass.openicl.icl_inferencer import GenInferencer

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('relative_path,variable,count', [
    ('mmlu_redux/mmlu_redux_gen.py', 'mmlu_redux_datasets', 1),
    ('mmlu_prox/mmlu_prox_5shot_cot_gen.py', 'mmlu_prox_datasets', 406),
    ('global_piqa/global_piqa_generation.py', 'global_piqa_datasets', 1),
    ('include/include_base_44_0shot_ppl.py', 'include_datasets', 44),
    ('hmmt_2025/hmmt_2025_matharena_gen.py', 'hmmt_2025_datasets', 2),
])
def test_configs_load(relative_path, variable, count):
    config = Config.fromfile(ROOT / 'opencompass/configs/datasets' /
                             relative_path)
    assert len(config[variable]) == count


@pytest.mark.parametrize('relative_path,variable,expected_train_split', [
    ('mmlu_redux/mmlu_redux_gen.py', 'mmlu_redux_datasets', 'test'),
    ('mmlu_prox/mmlu_prox_5shot_cot_gen.py', 'mmlu_prox_datasets',
     'validation'),
    ('global_piqa/global_piqa_generation.py', 'global_piqa_datasets', 'test'),
    ('include/include_base_44_0shot_ppl.py', 'include_datasets', 'test'),
    ('hmmt_2025/hmmt_2025_matharena_gen.py', 'hmmt_2025_datasets', 'test'),
])
def test_phase_one_configs_reference_existing_train_splits(
        relative_path, variable, expected_train_split):
    config = Config.fromfile(ROOT / 'opencompass/configs/datasets' /
                             relative_path)
    assert all(dataset['reader_cfg']['train_split'] == expected_train_split
               for dataset in config[variable])


def test_mmlu_prox_summarizer_matches_official_weighted_language_groups():
    config = Config.fromfile(ROOT /
                             'opencompass/configs/summarizers/mmlu_prox.py')
    groups = config.summarizer.summary_groups
    assert len(groups) == 29
    assert all(sum(group['weights'].values()) == 11759 for group in groups)


def test_mmlu_redux_official_prompt_and_first_letter_extraction():
    row = _format_mmlu_redux({
        'question': '  Question?  ',
        'choices': ['one', 'two', 'three', 'four'],
        'answer': 1,
    })
    assert row['prompt'] == (
        'Question?\nA. one\nB. two\nC. three\nD. four\n'
        'Please respond with the correct letter (A, B, C or D) without any '
        'additional comments, only the correct letter:')
    assert row['answer_letter'] == 'B'
    test_set = Dataset.from_list([
        {
            'subject': 'abstract_algebra',
            'category': 'stem'
        },
        {
            'subject': 'formal_logic',
            'category': 'humanities'
        },
    ])
    result = MMLUReduxEvaluator().score(['B', 'First A, then B'], ['B', 'B'],
                                        test_set)
    assert result['accuracy'] == 50.0
    assert result['category/stem'] == 100.0
    assert result['category/humanities'] == 0.0
    assert result['details'][1]['parsed'] == 'A'


def test_mmlu_prox_localized_prompt_and_extraction():
    source = {
        'question': '测试题',
        'option_0': '甲',
        'option_1': '乙',
        **{
            f'option_{index}': None
            for index in range(2, 10)
        },
        'cot_content': 'A: 让我们一步一步地思考。推理。答案是 (B)',
        'answer': 'B',
    }
    row = _format_example(source, 'zh', 'biology')
    assert row['question_prompt'].startswith('问题：\n测试题\n选项：\nA. 甲')
    assert row['question_prompt'].endswith('答案：让我们一步一步地思考。')
    assert '答案是 (X)' in row['description']

    test_set = Dataset.from_list([{'language_config': 'zh'}])
    result = MMLUProXEvaluator().score(['推理过程，答案是 (B)。'], ['B'], test_set)
    assert result['accuracy'] == 100.0
    assert result['details'][0]['parsed'] == 'B'


@pytest.mark.parametrize('lang,strings', LANG_LIBS.items())
def test_mmlu_prox_all_official_localized_answer_patterns(lang, strings):
    match = MMLUProXEvaluator._pattern(lang).search(strings[5].format('C'))
    assert match is not None
    assert match.group(1) == 'C'


def test_global_piqa_prompt_and_two_level_macro():
    prompt = _format_global_piqa(
        {
            'prompt': 'A situation',
            'solution0': 'first',
            'solution1': 'second',
            'label': 0,
        }, 'nonparallel', 'eng_latn')
    assert 'Situation:\nA situation ...' in prompt['question_prompt']
    assert prompt['answer_letter'] == 'A'

    test_set = Dataset.from_list([
        {
            'variant': 'nonparallel',
            'language_config': 'lang1'
        },
        {
            'variant': 'nonparallel',
            'language_config': 'lang2'
        },
        {
            'variant': 'parallel',
            'language_config': 'lang3'
        },
    ])
    predictions = [
        'The best answer is: A',
        'The best answer is: A',
        r'\boxed{D}',
    ]
    result = GlobalPIQAEvaluator().score(predictions, ['A', 'B', 'D'],
                                         test_set)
    assert result['nonparallel_accuracy'] == 50.0
    assert result['parallel_accuracy'] == 100.0
    assert result['accuracy'] == 75.0


def test_include_and_hmmt_prompts_match_official_order():
    include = _format_include({
        'question': '  Q? ',
        'option_a': 'a',
        'option_b': 'b',
        'option_c': 'c',
        'option_d': 'd',
        'answer': 2,
    })
    assert include == {
        'prompt': 'Q?\nA. a\nB. b\nC. c\nD. d\nAnswer:',
        'answer': 2,
    }
    hmmt = _format_hmmt({'problem': 'Problem text', 'answer': 42})
    assert hmmt['prompt'] == (
        'Put your final answer within \\boxed{}.\n\nProblem text')
    assert hmmt['answer'] == '42'


def test_include_uses_official_raw_continuation_loglikelihood():
    config = Config.fromfile(ROOT / 'opencompass/configs/datasets/include/'
                             'include_base_44_0shot_ppl.py')
    inferencer = config.include_datasets[0]['infer_cfg']['inferencer']

    assert config.include_datasets[0]['reader_cfg']['train_split'] == 'test'
    assert inferencer['type'].__name__ == 'LLInferencer'
    assert inferencer['continuations'] == {
        0: ' A',
        1: ' B',
        2: ' C',
        3: ' D',
    }


def test_hmmt_uses_matharena_non_strict_parser():
    result = MathArenaEvaluator().score([
        'Work shown here. The final answer is 103.',
        r'Therefore, \boxed{\frac{1}{2}}.'
    ], ['103', r'\frac{1}{2}'])
    assert result['accuracy'] == 100.0


def test_gen_inferencer_accepts_task_generation_kwargs(tmp_path):
    model = MagicMock()
    model.is_api = True
    inferencer = GenInferencer(model=model,
                               max_out_len=10,
                               output_json_filepath=str(tmp_path),
                               generation_kwargs={
                                   'temperature': 0.8,
                                   'top_p': 0.95
                               })
    assert inferencer.generation_kwargs == {
        'temperature': 0.8,
        'top_p': 0.95,
    }
