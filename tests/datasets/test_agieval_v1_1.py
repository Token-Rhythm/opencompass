"""AGIEval v1.1 protocol and launcher regression tests (no model service)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mmengine.config import Config

from opencompass.datasets.agieval import dataset_loader as legacy
from opencompass.datasets.agieval.agieval import AGIEvalEvaluator
from opencompass.datasets.agieval.agieval_v1_1 import (
    AGIEVAL_V1_1_TASKS, AGIEvalV11Dataset, single_choice_label)
from opencompass.datasets.agieval.agieval_v1_1_postprocess import (
    agieval_mathqa_postprocess, normalize_mathqa_label)
from opencompass.models.base import LMTemplateParser
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.tasks.openicl_eval import extract_prediction_content
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets.agieval.agieval_v1_1_cloze import AGIEvalV11ClozeEvaluator
from opencompass.datasets.agieval.agieval_v1_1_postprocess import agieval_single_choice_postprocess

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / 'opencompass/configs/datasets/agieval'
CORE = ROOT / 'script/run_posttrain_objective_benchmark.sh'
MODES = [
    ('agieval_v1_1_gen', 'zero-shot', True),
    ('agieval_v1_1_zeroshot_cot_gen', 'zero-shot-CoT', True),
]


@pytest.fixture
def local_data(tmp_path):
    question = dict(passage='阅读材料。', question=r'待测题 {x} = 2?',
                    options=['(A)一', '(B)二', '(C)三', '(D)四'],
                    label=['B'], answer=None)
    (tmp_path / 'jec-qa-kd.jsonl').write_text(
        json.dumps(question, ensure_ascii=False) + '\n', encoding='utf-8')
    cloze = dict(passage=None, question='计算 1+1', options=None,
                 label=None, answer='2')
    (tmp_path / 'gaokao-mathcloze.jsonl').write_text(
        json.dumps(cloze, ensure_ascii=False) + '\n', encoding='utf-8')
    return tmp_path, question


@pytest.mark.parametrize('label,expected', [('B', 'B'), (['B'], 'B')])
def test_singleton_reference_normalization(label, expected):
    assert single_choice_label(label) == expected


@pytest.mark.parametrize('label', [[], ['A', 'B'], None, 'Z'])
def test_invalid_singleton_reference_is_not_silently_selected(label):
    with pytest.raises(ValueError):
        single_choice_label(label)


@pytest.mark.parametrize('setting,formatter', [
    ('zero-shot', legacy.convert_zero_shot),
    ('zero-shot-CoT', legacy.convert_zero_shot_CoT_stage1),
])
def test_zero_shot_prompts_reuse_existing_official_builders(
        local_data, setting, formatter):
    path, question = local_data
    data = AGIEvalV11Dataset.load(
        path=str(path), name='jec-qa-kd', setting_name=setting)
    assert data[0]['label'] == 'B'
    assert data[0]['messages'] == [
        dict(role='user', content=formatter(question, 'jec-qa-kd'))]
    assert data[0]['num_shots'] == 0
    assert '该问题为单选题' not in data[0]['messages'][0]['content']




@pytest.mark.parametrize('filename,setting,chat', MODES)
def test_configs_use_single_pass_and_common_choice_scorer(filename, setting, chat):
    cfg = Config.fromfile(CONFIG_DIR / f'{filename}.py')
    datasets = cfg.agieval_v1_1_datasets
    assert [d['name'] for d in datasets] == list(AGIEVAL_V1_1_TASKS)
    for dataset in datasets:
        assert dataset['setting_name'] == setting
        assert dataset['chat_mode'] is chat
        assert dataset['infer_cfg']['inferencer']['type'] is GenInferencer
        evaluation = dataset['eval_cfg']
        if dataset['name'] in ('math', 'gaokao-mathcloze'):
            assert evaluation['evaluator']['type'] is AGIEvalV11ClozeEvaluator
        elif dataset['name'] == 'gaokao-mathqa':
            assert evaluation['evaluator']['type'] is AccEvaluator
            assert evaluation['pred_postprocessor']['type'] is agieval_mathqa_postprocess
        else:
            assert evaluation['evaluator']['type'] is AccEvaluator
            assert evaluation['pred_postprocessor'] == dict(
                type=agieval_single_choice_postprocess, options='ABCDE')
    groups = cfg.agieval_v1_1_summary_groups
    assert [len(g['subsets']) for g in groups] == [21, 8, 11, 2]
    assert all(subset in {d['abbr'] for d in datasets}
               for g in groups for subset in g['subsets'])


@pytest.mark.parametrize('filename,setting,chat', MODES)
def test_end_to_end_prompt_inference_makes_one_generation(
        tmp_path, local_data, filename, setting, chat):
    path, _ = local_data
    cfg = Config.fromfile(CONFIG_DIR / f'{filename}.py')
    item = next(d for d in cfg.agieval_v1_1_datasets if d['name'] == 'jec-qa-kd')
    dataset = AGIEvalV11Dataset(
        path=str(path), name='jec-qa-kd', setting_name=setting,
        chat_mode=chat, reader_cfg=item['reader_cfg'])
    template_cfg = dict(item['infer_cfg']['prompt_template'])
    template_cfg.pop('type')
    template = RawPromptTemplate(**template_cfg)
    expected = template.generate_item(dataset.test[0])
    model = MagicMock()
    model.is_api = True
    model.generation_kwargs = {}
    model.parse_template.side_effect = lambda entries, mode: entries
    model.generate_from_template.return_value = [
        dict(reasoning_content='A 是干扰项', content='答案是 B')]
    inferencer = GenInferencer(
        model=model, max_out_len=4096, batch_size=1,
        output_json_filepath=str(tmp_path / filename),
        output_json_filename='predictions.json')
    inferencer.inference(ZeroRetriever(dataset), prompt_template=template)
    model.generate_from_template.assert_called_once()
    assert list(model.generate_from_template.call_args.args[0]) == [expected]
    assert LMTemplateParser().parse_template([expected], 'gen') == [expected]
    if chat:
        assert expected[0]['role'] == 'system'
    else:
        assert [m['role'] for m in expected] == ['user']
    saved = json.loads((tmp_path / filename / 'predictions.json').read_text())
    prediction = extract_prediction_content(saved['0'])
    assert prediction == '答案是 B'
    assert first_option_postprocess(prediction, 'ABCDE') == 'B'


def test_math_cloze_reuses_legacy_math_evaluator(local_data):
    path, _ = local_data
    row = AGIEvalV11Dataset.load(
        path=str(path), name='gaokao-mathcloze')[0]
    assert AGIEvalEvaluator().score([r'\boxed{2}'], [row['label']])['score'] == 100




def _core_config(tmp_path, *args):
    shim = tmp_path / 'python-shim'
    shim.write_text(
        f'#!{sys.executable}\n'
        'import os, sys\n'
        'if len(sys.argv) > 1 and sys.argv[1] == "run.py":\n'
        '    print("TEST_RUNPY", repr(sys.argv[1:]))\n'
        '    raise SystemExit(0)\n'
        f'os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n')
    shim.chmod(0o755)
    env = os.environ.copy()
    env.update(OPENCOMPASS_SERVED_MAX_SEQ_LEN_OVERRIDE='262144',
               OPENCOMPASS_RUNTIME_TMPDIR=str(tmp_path / 'runtime'))
    result = subprocess.run(
        ['bash', str(CORE), 'agieval', '--python', str(shim),
         '--model', 'test-model', '--base-url', 'http://127.0.0.1:1/v1',
         '--tokenizer-path', 'test/tokenizer', '--samples', '1',
         '--dry-run', *args], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    filename = next(line.split(': ', 1)[1] for line in result.stdout.splitlines()
                    if line.startswith('OpenCompass config: '))
    return Config.fromfile(filename), result.stdout


@pytest.mark.parametrize('setting,chat', [
    ('zero-shot', True), ('zero-shot-CoT', True),
])
def test_post_launcher_translates_protocol_and_preserves_common_defaults(
        tmp_path, setting, chat):
    args = [] if setting == 'zero-shot' else ['--agieval-setting', setting]
    cfg, stdout = _core_config(tmp_path, *args)
    model = cfg.models[0]
    assert model['max_seq_len'] == 65536
    assert model['max_out_len'] == 4096
    assert model['temperature'] == 1.0
    assert model['batch_size'] == 512
    assert model['max_workers'] == 64
    assert model['query_per_second'] == 64
    assert model['retry'] == 1
    assert model['timeout'] == 3600
    assert model['generation_endpoint'] == 'chat'
    for dataset in cfg.datasets:
        assert dataset['setting_name'] == setting
        assert dataset['chat_mode'] is chat
        assert dataset['reader_cfg']['test_range'] == '[:1]'
        assert dataset['infer_cfg']['inferencer']['max_out_len'] == 4096


@pytest.mark.parametrize('args', [
    ['--agieval-setting', 'few-shot'],
    ['--agieval-setting', 'few-shot-CoT'],
    ['--agieval-chat-mode', 'false'],
    ['--agieval-chat-mode', 'maybe'],
    ['--agieval-setting'],
])
def test_invalid_protocol_fails_before_endpoint_preflight(args):
    result = subprocess.run(
        ['bash', str(CORE), 'agieval', *args], cwd=ROOT,
        text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert 'OpenCompass config:' not in result.stdout
    assert 'preflight' not in result.stderr


@pytest.mark.parametrize('text,expected', [
    ('AD', 'AD'), ('A D', 'AD'), ('A、D', 'AD'), ('D，A', 'AD'),
    ("['A', 'D']", 'AD'), ('(A), (D)', 'AD'), ('ADDA', 'AD'),
    ('Ａ，Ｄ', 'AD'), ('a and d', 'AD'), ('A和D', 'AD'),
    ('B', 'B'), ('AB', 'AB'), ('答案是：A B D', 'ABD'),
    ('A 不正确，最终答案是 BD。', 'BD'),
    ('A、B、C、D 逐一分析。\n因此选 A、D。', 'AD'),
    ('The final answer is **D, A**.', 'AD'),
    (r'最终答案：$\boxed{\mathrm{AD}}$', 'AD'),
    (r'分析过程涉及 B 和 C，结论为 \boxed{A,D}', 'AD'),
    ('先分析所有选项。\nAD', 'AD'),
    ('最终答案是AD。下面解释为什么 B 不对。', 'AD'),
    ('答案是 A。重新检查，最终答案是 AD。', 'AD'),
    ('答案是 A 或 D', ''), ('答案是 A or D', ''),
    ('A不对，B可能正确', ''), ('A/B', ''),
    ('最终答案：无法确定', ''), ('答案是AD。最终答案：不确定', ''),
    ('ABCDE', ''), ('E', ''), ('', ''), (None, ''),
    ('答案是 AD，但也可能是 B', ''),
])
def test_mathqa_final_option_set_extraction(text, expected):
    assert agieval_mathqa_postprocess(text) == expected


@pytest.mark.parametrize('raw,canonical', [
    ('AD', 'AD'), ('ACD', 'ACD'), ('A B D', 'ABD'),
    ('A C', 'AC'), ('B C D', 'BCD'), ('CD', 'CD'), ('AC', 'AC'),
    ('D A', 'AD'), (['A', 'D'], 'AD'),
])
def test_mathqa_reference_normalization(raw, canonical):
    assert normalize_mathqa_label(raw) == canonical


def test_mathqa_loader_preserves_all_questions_and_original_file(tmp_path):
    path = tmp_path / 'gaokao-mathqa.jsonl'
    refs = ['B', 'AD', 'ACD', 'A B D', 'A C', 'B C D', 'CD', 'AC']
    original = '\n'.join(json.dumps(dict(
        passage=None, question=f'题目{i}', options=['(A)a', '(B)b', '(C)c', '(D)d'],
        label=ref, answer=None)) for i, ref in enumerate(refs))
    path.write_text(original)
    rows = AGIEvalV11Dataset.load(path=str(tmp_path), name='gaokao-mathqa')
    assert len(rows) == len(refs)
    assert rows['label'] == ['B', 'AD', 'ACD', 'ABD', 'AC', 'BCD', 'CD', 'AC']
    assert rows['id'] == list(range(len(refs)))
    assert path.read_text() == original


def test_mathqa_exact_matching_no_partial_credit_or_single_option_fallback():
    outputs = ['D、A', 'A', 'ABD', 'AB', 'B', '答案是 A 或 D']
    refs = ['AD', 'AD', 'AD', 'B', 'B', 'AD']
    predictions = [agieval_mathqa_postprocess(x) for x in outputs]
    result = AccEvaluator().score(predictions, refs)
    assert result['accuracy'] == pytest.approx(100 * 2 / 6)


@pytest.mark.parametrize('setting', ['few-shot', 'few-shot-CoT'])
def test_removed_modes_rejected_by_loader(local_data, setting):
    with pytest.raises(ValueError, match='Unsupported AGIEval setting'):
        AGIEvalV11Dataset.load(path=str(local_data[0]), name='jec-qa-kd',
                              setting_name=setting)


def test_false_chat_mode_rejected(local_data):
    with pytest.raises(ValueError, match='chat_mode=True'):
        AGIEvalV11Dataset.load(path=str(local_data[0]), name='jec-qa-kd',
                              chat_mode=False)


@pytest.mark.parametrize('text,expected', [
    ('因此应选：\n\n\\[\n\\boxed{AD}\n\\]', 'AD'),
    ('最终答案：\n\\[\n\\boxed{\\mathrm{D,A}}\n\\]', 'AD'),
    (r'答案是 \(AD\)。后续解释 B 为什么不正确。', 'AD'),
    ('答案：$$\nA D\n$$', 'AD'),
    (r'答案：$\boxed{AD}$', 'AD'),
    ('推导过程。\n\\[\n\\boxed{AD}\n\\]', 'AD'),
    ('推导过程。\n$$\nA、D\n$$', 'AD'),
    ('因此应选：\n\\[\n\\boxed{B}\n\\]', 'B'),
    ('答案：\\[\nA 或 D\n\\]', ''),
    ('答案：\\[\nA/D\n\\]', ''),
    (r'答案：\[AD\]，但也可能是 B', ''),
    (r'答案：\[AD\] or B', ''),
    ('答案：\\[\n\\boxed{AD}', ''),
    ('答案：\\[\nAD\n\\)\n', ''),
    ('最终答案：无法确定。\n\\[\n\\boxed{AD}\n\\]', ''),
    ('推导中的候选：\\[AD\\]\n仍然无法确定', ''),
])
def test_mathqa_formula_wrappers_preserve_option_set_rules(text, expected):
    assert agieval_mathqa_postprocess(text) == expected


def test_mathqa_real_display_formula_response_scores_exactly():
    fixture = json.loads(
        (ROOT / 'tests/data/agieval_v1_1_mathqa_display_answer.json').read_text())
    prediction = agieval_mathqa_postprocess(fixture['content'])
    assert prediction == 'AD'
    assert AccEvaluator().score([prediction], [fixture['reference']])[
        'accuracy'] == 100
    assert AccEvaluator().score([prediction], ['A'])['accuracy'] == 0
    assert AccEvaluator().score([prediction], ['ABD'])['accuracy'] == 0
