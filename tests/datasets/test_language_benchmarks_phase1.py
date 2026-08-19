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
from opencompass.datasets.mmlu_prox import (MMLU_PROX_LITE_PATH,
                                            MMLU_PROX_LITE_REVISION,
                                            MMLU_PROX_CATEGORIES,
                                            MMLUProXEvaluator, _format_example)
from opencompass.datasets.mmlu_prox_lang_libs import LANG_LIBS, LANG_SUBJECTS
from opencompass.datasets.mmlu_redux import (MMLUReduxEvaluator,
                                             MMLU_REDUX_SUBJECTS,
                                             _format_mmlu_redux,
                                             _mmlu_redux_description)
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.models.base import LMTemplateParser

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('relative_path,variable,count', [
    ('mmlu_redux/mmlu_redux_gen.py', 'mmlu_redux_datasets', 1),
    ('mmlu_redux/mmlu_redux_gen.py', 'mmlu_redux_lm_eval_datasets', 1),
    ('mmlu_prox/mmlu_prox_5shot_cot_gen.py', 'mmlu_prox_5shot_datasets', 406),
    ('mmlu_prox/mmlu_prox_0shot_cot_gen.py', 'mmlu_prox_0shot_datasets', 406),
    ('mmlu_prox/mmlu_prox_lite_5shot_cot_gen.py',
     'mmlu_prox_lite_5shot_datasets', 406),
    ('mmlu_prox/mmlu_prox_lite_0shot_cot_gen.py',
     'mmlu_prox_lite_0shot_datasets', 406),
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
    ('mmlu_redux/mmlu_redux_gen.py', 'mmlu_redux_lm_eval_datasets', 'test'),
    ('mmlu_prox/mmlu_prox_5shot_cot_gen.py', 'mmlu_prox_5shot_datasets',
     'validation'),
    ('mmlu_prox/mmlu_prox_0shot_cot_gen.py', 'mmlu_prox_0shot_datasets',
     'validation'),
    ('mmlu_prox/mmlu_prox_lite_5shot_cot_gen.py',
     'mmlu_prox_lite_5shot_datasets', 'validation'),
    ('mmlu_prox/mmlu_prox_lite_0shot_cot_gen.py',
     'mmlu_prox_lite_0shot_datasets', 'validation'),
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


@pytest.mark.parametrize('relative_path,variable,content', [
    ('global_piqa/global_piqa_generation.py', 'global_piqa_datasets',
     '{question_prompt}'),
    ('hmmt_2025/hmmt_2025_matharena_gen.py', 'hmmt_2025_datasets', '{prompt}'),
])
def test_phase_one_generation_configs_use_role_preserving_prompts(
        relative_path, variable, content):
    config = Config.fromfile(ROOT / 'opencompass/configs/datasets' /
                             relative_path)
    prompt_cfg = config[variable][0]['infer_cfg']['prompt_template']
    assert prompt_cfg['type'] is RawPromptTemplate
    assert prompt_cfg['messages'] == [dict(role='user', content=content)]


def test_mmlu_prox_summarizer_matches_official_weighted_language_groups():
    config = Config.fromfile(ROOT /
                             'opencompass/configs/summarizers/mmlu_prox.py')
    groups = config.summarizer.summary_groups
    assert len(groups) == 30
    assert groups[0]['name'] == 'mmlu_prox_5shot_en'
    assert groups[0]['subsets'][0] == 'mmlu_prox_5shot_en_biology'
    assert all(
        sum(group['weights'].values()) == 11759 for group in groups[:29])
    assert groups[-1]['name'] == 'mmlu_prox'
    assert len(groups[-1]['subsets']) == 29 * 14
    assert sum(groups[-1]['weights'].values()) == 29 * 11759

    variants = [
        ('mmlu_prox_0shot.py', 'mmlu_prox_0shot_en', 11759),
        ('mmlu_prox_lite.py', 'mmlu_prox_lite_5shot_en', 588),
        ('mmlu_prox_lite_0shot.py', 'mmlu_prox_lite_0shot_en', 588),
    ]
    for filename, expected_group_name, expected_size in variants:
        config = Config.fromfile(ROOT / 'opencompass/configs/summarizers' /
                                 filename)
        groups = config.summarizer.summary_groups
        assert len(groups) == 29
        assert groups[0]['name'] == expected_group_name
        assert all(
            sum(group['weights'].values()) == expected_size
            for group in groups)


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
    assert len(MMLU_REDUX_SUBJECTS) == 57
    assert _mmlu_redux_description('abstract_algebra') == (
        'The following are multiple choice questions (with answers) about '
        'abstract algebra.\n\n')

    prompt_template = RawPromptTemplate(messages=[
        dict(role='system', content='{description}'),
        dict(role='user', content='{prompt}'),
    ])
    messages = prompt_template.generate_item({
        **row,
        'description':
        _mmlu_redux_description('abstract_algebra'),
    })
    assert messages == [
        {
            'role':
            'system',
            'content': ('The following are multiple choice questions (with '
                        'answers) about abstract algebra.\n\n'),
        },
        {
            'role': 'user',
            'content': row['prompt'],
        },
    ]

    config = Config.fromfile(
        ROOT / 'opencompass/configs/datasets/mmlu_redux/mmlu_redux_gen.py')
    lm_eval_dataset = config.mmlu_redux_lm_eval_datasets[0]
    assert lm_eval_dataset['abbr'] == 'mmlu_redux_lm_eval'
    lm_eval_prompt_template = RawPromptTemplate(
        messages=lm_eval_dataset['infer_cfg']['prompt_template']['messages'])
    lm_eval_messages = lm_eval_prompt_template.generate_item({
        **row,
        'description':
        _mmlu_redux_description('abstract_algebra'),
    })
    assert lm_eval_messages == [{
        'role':
        'user',
        'content': ('The following are multiple choice questions (with '
                    'answers) about abstract algebra.\n\n' + row['prompt']),
    }]

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
    assert (row['fewshot_question_prompt'] +
            row['fewshot_answer_prompt'] == row['fewshot_prompt'])

    test_set = Dataset.from_list([{'language_config': 'zh'}])
    result = MMLUProXEvaluator().score(['推理过程，答案是 (B)。'], ['B'], test_set)
    assert result['accuracy'] == 100.0
    assert result['details'][0]['parsed'] == 'B'


def test_mmlu_prox_all_categories_have_localized_subjects():
    for lang in LANG_LIBS:
        for category in MMLU_PROX_CATEGORIES:
            assert category.replace(' ', '_') in LANG_SUBJECTS[lang]

    row = _format_example(
        {
            'question': 'A question?',
            'option_0': 'first',
            **{
                f'option_{index}': None
                for index in range(1, 10)
            },
            'cot_content': "A: Let's think step by step. Answer.",
            'answer': 'A',
        }, 'en', 'computer science')
    assert 'about computer_science.' in row['description']


def test_mmlu_prox_five_shot_is_one_official_task_context():
    source = {
        'question': 'A question?',
        'option_0': 'first',
        'option_1': 'second',
        **{
            f'option_{index}': None
            for index in range(2, 10)
        },
        'cot_content':
        "A: Let's think step by step. Because. the answer is (B)",
        'answer': 'B',
    }
    row = _format_example(source, 'en', 'biology')
    ice_template = PromptTemplate(template='{fewshot_prompt}')
    ice = ''.join(ice_template.generate_ice_item(row, row['answer_letter'])
                  for _ in range(5))
    five_shot_template = PromptTemplate(
        template='{description}</E>{question_prompt}', ice_token='</E>')
    prompt = five_shot_template.generate_item(
        row, ice_field_replace_token=ice)

    assert prompt == (row['description'] + row['fewshot_prompt'] * 5 +
                      row['question_prompt'])

    config = Config.fromfile(
        ROOT / 'opencompass/configs/datasets/mmlu_prox/'
        'mmlu_prox_5shot_cot_gen.py')
    infer_cfg = config.mmlu_prox_5shot_datasets[0]['infer_cfg']
    assert infer_cfg['ice_template']['type'] is PromptTemplate
    assert infer_cfg['prompt_template']['type'] is PromptTemplate
    assert infer_cfg['retriever']['ice_separator'] == ''
    assert infer_cfg['retriever']['ice_eos_token'] == ''


def test_mmlu_prox_lite_configs_pin_official_dataset_revision():
    assert MMLU_PROX_LITE_PATH == 'li-lab/MMLU-ProX-Lite'
    assert MMLU_PROX_LITE_REVISION == (
        'e82aafb9460529687d3c7e51b401d8dd1dd309dd')
    for filename, variable in [
        ('mmlu_prox_lite_5shot_cot_gen.py', 'mmlu_prox_lite_5shot_datasets'),
        ('mmlu_prox_lite_0shot_cot_gen.py', 'mmlu_prox_lite_0shot_datasets'),
    ]:
        config = Config.fromfile(
            ROOT / 'opencompass/configs/datasets/mmlu_prox' / filename)
        assert all(dataset['path'] == MMLU_PROX_LITE_PATH
                   for dataset in config[variable])
        assert all(dataset['revision'] == MMLU_PROX_LITE_REVISION
                   for dataset in config[variable])


def test_mmlu_prox_paths_survive_nested_smoke_config_imports():
    """MMEngine must not serialize imported constants as dotted strings."""
    package = 'opencompass.configs.datasets.mmlu_prox'
    imports = [
        ('mmlu_prox_5shot_cot_gen', 'mmlu_prox_5shot_datasets'),
        ('mmlu_prox_0shot_cot_gen', 'mmlu_prox_0shot_datasets'),
        ('mmlu_prox_lite_5shot_cot_gen', 'mmlu_prox_lite_5shot_datasets'),
        ('mmlu_prox_lite_0shot_cot_gen', 'mmlu_prox_lite_0shot_datasets'),
    ]
    source = ['from mmengine.config import read_base', 'with read_base():']
    source.extend(' ' * 4 + f'from {package}.{module} import {variable}'
                  for module, variable in imports)
    source.append('datasets = [' + ', '.join(f'{variable}[0]'
                                             for _, variable in imports) + ']')
    config = Config.fromstring('\n'.join(source), '.py')
    assert [dataset['path'] for dataset in config.datasets] == [
        'li-lab/MMLU-ProX',
        'li-lab/MMLU-ProX',
        'li-lab/MMLU-ProX-Lite',
        'li-lab/MMLU-ProX-Lite',
    ]
    assert all(not dataset['path'].startswith('opencompass.')
               for dataset in config.datasets)


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


def test_hmmt_2025_uses_matharena_official_four_runs():
    config = Config.fromfile(
        ROOT / 'opencompass/configs/datasets/hmmt_2025/'
        'hmmt_2025_matharena_gen.py')

    assert len(config.hmmt_2025_datasets) == 2
    assert all(dataset['num_repeats'] == 4
               for dataset in config.hmmt_2025_datasets)


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


def test_language_benchmark_runner_streams_long_thinking_responses():
    script = (ROOT / 'script/run_language_benchmarks_smoke.sh').read_text()

    assert 'STREAM_RESPONSES="1"' in script
    assert '--stream-responses) STREAM_RESPONSES="1"; shift ;;' in script
    assert '--no-stream-responses) STREAM_RESPONSES="0"; shift ;;' in script
    assert 'stream_chat=${STREAM_RESPONSES_PY}' in script

    objective_script = (ROOT /
                        'script/run_posttrain_objective_benchmark.sh').read_text()
    assert ('--dataset-kwargs-json) DATASET_KWARGS_JSON="$2"; shift 2 ;;'
            in objective_script)
    assert 'dataset.update(dataset_kwargs)' in objective_script
