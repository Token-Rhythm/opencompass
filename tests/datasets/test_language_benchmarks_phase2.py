"""Protocol tests for the phase-two benchmark adapters."""

import csv
import zipfile
from pathlib import Path

import pytest
from mmengine import Config

from opencompass.datasets import aa_lcr as aa_lcr_module
from opencompass.datasets.aa_lcr import AALCRDataset, _build_aa_lcr_prompt
from opencompass.datasets.multichallenge import (_format_multichallenge,
                                                 _score_multichallenge)
from opencompass.datasets.polymath import (POLYMATH_ANSWER_INSTRUCTIONS,
                                           PolyMathEvaluator, _format_polymath,
                                           extract_first_boxed_content)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('relative_path,variable,count', [
    ('polymath/polymath_0shot_gen.py', 'polymath_datasets', 72),
    ('multichallenge/multichallenge_gen.py', 'multichallenge_datasets', 1),
    ('aa_lcr/aa_lcr_gen.py', 'aa_lcr_datasets', 1),
])
def test_configs_load(relative_path, variable, count):
    config = Config.fromfile(ROOT / 'opencompass/configs/datasets' /
                             relative_path)
    assert len(config[variable]) == count


def test_polymath_summarizer_has_official_difficulty_weights():
    config = Config.fromfile(ROOT /
                             'opencompass/configs/summarizers/polymath.py')
    groups = config.summarizer.summary_groups
    language_groups = groups[:-1]
    assert len(language_groups) == 18
    assert all(
        list(group['weights'].values()) == [1, 2, 4, 8]
        for group in language_groups)


def test_polymath_prompt_and_first_boxed_answer():
    row = _format_polymath(
        {
            'id': 'low-zh-0',
            'question': '问题',
            'answer': '2',
        }, 'zh', 'low')
    assert row['prompt'] == (f'问题\n\n{POLYMATH_ANSWER_INSTRUCTIONS["zh"]}')
    assert extract_first_boxed_content(
        r'first $\boxed{\frac{1}{2}}$, then $\boxed{3}$') == r'\frac{1}{2}'


def test_polymath_official_math_equality_cases():
    result = PolyMathEvaluator().score([
        r'$\boxed{50}$',
        r'$\boxed{B}$',
        r'$\boxed{\begin{pmatrix}1&2\\3&4\end{pmatrix}}$',
        r'$\boxed{-\pi\log_2}$',
        'There is no boxed answer.',
    ], [
        '0.5',
        'B',
        r'\begin{bmatrix}1&2\\3&4\end{bmatrix}',
        r'$-\pi \log_2$',
        '0',
    ])
    assert result['accuracy'] == 80.0
    assert [detail['correct'] for detail in result['details']
            ] == [True, True, True, True, False]


def test_multichallenge_conversation_and_official_axis_macro():
    formatted = _format_multichallenge({
        'QUESTION_ID':
        'q1',
        'AXIS':
        'A',
        'CONVERSATION': [{
            'role': 'user',
            'content': 'Remember this.'
        }],
        'TARGET_QUESTION':
        'Did it remember?',
        'PASS_CRITERIA':
        'YES',
    })
    assert formatted['dialogue'][-1] == {'role': 'assistant', 'content': 'YES'}

    rows = [{
        'question_id': 'q1',
        'axis': 'A',
        'pass_criteria': 'YES'
    }, {
        'question_id': 'q2',
        'axis': 'A',
        'pass_criteria': 'YES'
    }, {
        'question_id': 'q3',
        'axis': 'B',
        'pass_criteria': 'YES'
    }]
    score = _score_multichallenge(
        ['Reasoning. YES', 'Reasoning. NO', 'NO initially, but verdict: YES'],
        rows)
    assert score['axis/A'] == 50.0
    assert score['axis/B'] == 100.0
    assert score['overall_score'] == 75.0


def test_aa_lcr_prompt_uses_official_document_order():
    prompt = _build_aa_lcr_prompt(['first text', 'second text'], 'Question?')
    assert prompt.index('BEGIN DOCUMENT 1:\nfirst text') < prompt.index(
        'BEGIN DOCUMENT 2:\nsecond text')
    assert prompt.endswith('START QUESTION\n\nQuestion?\n\nEND QUESTION\n')


def test_aa_lcr_loader_preserves_order_and_splits_answer_criteria(
        tmp_path, monkeypatch):
    csv_path = tmp_path / 'AA-LCR_Dataset.csv'
    archive_path = tmp_path / 'AA-LCR_extracted-text.zip'
    fields = [
        'question_id', 'question', 'answer', 'data_source_filenames',
        'document_category', 'document_set_id', 'input_tokens'
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            'question_id': '1',
            'question': 'Question?',
            'answer': 'criterion one;criterion two',
            'data_source_filenames': 'b.txt;a’s.txt;Başev.txt',
            'document_category': 'Company Documents',
            'document_set_id': 'set-1',
            'input_tokens': '10',
        })
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('lcr/Company_Documents/set-1/b.txt', 'B')
        mojibake_name = 'a’s.txt'.encode('utf-8').decode('cp437')
        archive.writestr(f'lcr/Company_Documents/set-1/{mojibake_name}', 'A')
        normalized_name = 'Başev.txt'.encode('utf-8').decode('cp437')
        archive.writestr(f'lcr/Company_Documents/set-1/{normalized_name}', 'C')

    def fake_download(repo_id, filename, **kwargs):
        assert repo_id == 'ArtificialAnalysis/AA-LCR'
        return str(csv_path if filename.endswith('.csv') else archive_path)

    monkeypatch.setattr(aa_lcr_module, 'hf_hub_download', fake_download)
    row = AALCRDataset.load()['test'][0]
    assert row['answer'] == ['criterion one', 'criterion two']
    assert row['prompt'].index('BEGIN DOCUMENT 1:\nB') < row['prompt'].index(
        'BEGIN DOCUMENT 2:\nA')
    assert row['prompt'].index('BEGIN DOCUMENT 2:\nA') < row['prompt'].index(
        'BEGIN DOCUMENT 3:\nC')
