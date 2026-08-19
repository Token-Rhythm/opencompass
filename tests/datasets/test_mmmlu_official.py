from datasets import Dataset

from opencompass.datasets import mmmlu
from opencompass.configs.datasets.mmmlu.mmmlu_gen_c51a84 import mmmlu_datasets
from opencompass.openicl.icl_inferencer import ParallelGenInferencer


def test_mmmlu_loader_uses_pinned_revision_without_remote_code(monkeypatch):
    source = Dataset.from_list([{
        'Question': 'Question?',
        'A': 'a',
        'B': 'b',
        'C': 'c',
        'D': 'd',
        'Answer': 'C',
        'Subject': 'subject_name',
    }])
    called = {}

    def fake_load_dataset(**kwargs):
        called.update(kwargs)
        return source

    monkeypatch.setattr(mmmlu, 'load_dataset', fake_load_dataset)
    loaded = mmmlu.MMMLUDataset.load(
        path='openai/MMMLU',
        name='mmlu_ZH-CN',
        hf_revision='pinned',
    )

    assert called == {
        'path': 'openai/MMMLU',
        'name': 'ZH_CN',
        'split': 'test',
        'revision': 'pinned',
    }
    row = loaded['test'][0]
    assert row['target'] == 'C'
    assert row['subject'] == 'subject name'
    assert row['question_prompt'] == (
        'Answer the following multiple choice question. The last line of your '
        "response should be of the following format: 'Answer: $LETTER' "
        '(without quotes) where LETTER is one of ABCD. Think step by step '
        'before answering.\n\nQuestion?\n\nA) a\n\nB) b\n\nC) c\n\nD) d')


def test_mmmlu_answer_extractor_matches_simple_evals_protocol():
    extract = mmmlu.mmmlu_answer_postprocess
    assert extract('Reasoning A, then B.\nAnswer: $C') == 'C'
    assert extract('推理\n答案：Ｄ') == 'D'
    assert extract('الجواب: ج') == 'C'
    assert extract('উত্তরঃ ড') == 'C'
    assert extract('I compared A with B and prefer C') == ''


def test_mmmlu_config_uses_official_single_user_prompt_and_pinned_revision():
    dataset = mmmlu_datasets[0]
    assert dataset['hf_revision'] == \
        '325a01dc3e173cac1578df94120499aaca2e2504'
    prompt = dataset['infer_cfg']['prompt_template']
    assert prompt['messages'] == [
        dict(role='user', content='{question_prompt}')
    ]
    assert dataset['reader_cfg']['input_columns'] == ['question_prompt']
    assert dataset['infer_cfg']['inferencer'] == {
        'type': ParallelGenInferencer,
        'save_every': 1,
    }
