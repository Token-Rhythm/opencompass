import json

import pytest

from script.verify_multilingual_prediction_coverage import (
    verify_global_piqa, verify_include, verify_mmlu_prox)


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding='utf-8')


def test_global_piqa_requires_exact_row_count(tmp_path):
    dump(tmp_path / 'global_piqa_generation.json', {'0': {}, '1': {}})
    assert verify_global_piqa(tmp_path, expected_rows=2) == {
        'datasets': 1,
        'rows': 2,
    }
    with pytest.raises(SystemExit, match='expected=3, actual=2'):
        verify_global_piqa(tmp_path, expected_rows=3)


def test_mmlu_prox_requires_every_dataset_and_one_row_each(tmp_path):
    dump(tmp_path / 'mmlu_prox_5shot_en_math.json', {'0': {}})
    dump(tmp_path / 'mmlu_prox_5shot_zh_math.json', {'0': {}})
    expected = {
        'mmlu_prox_5shot_en_math',
        'mmlu_prox_5shot_zh_math',
    }
    assert verify_mmlu_prox(tmp_path, expected) == {
        'datasets': 2,
        'rows': 2,
    }

    (tmp_path / 'mmlu_prox_5shot_zh_math.json').unlink()
    with pytest.raises(SystemExit, match='Incomplete MMLU-ProX dataset set'):
        verify_mmlu_prox(tmp_path, expected)


def test_include_requires_every_language_and_exact_total(tmp_path):
    # Full/smoke launcher runs append a route-independent run suffix, while
    # the imported dataset config retains the canonical abbreviation.
    dump(tmp_path / 'include_base_44_arabic_full_chat.json',
         {'0': {}, '1': {}})
    dump(tmp_path / 'include_base_44_chinese_smoke_chat.json', {'0': {}})
    expected = {
        'include_base_44_arabic',
        'include_base_44_chinese',
    }
    assert verify_include(tmp_path, expected, expected_total_rows=3) == {
        'datasets': 2,
        'rows': 3,
    }

    with pytest.raises(SystemExit, match='expected=4, actual=3'):
        verify_include(tmp_path, expected, expected_total_rows=4)


def test_include_rejects_duplicate_canonical_dataset(tmp_path):
    dump(tmp_path / 'include_base_44_arabic.json', {'0': {}})
    dump(tmp_path / 'include_base_44_arabic_full_chat.json', {'0': {}})
    with pytest.raises(SystemExit, match='Duplicate INCLUDE prediction'):
        verify_include(tmp_path, {'include_base_44_arabic'},
                       expected_total_rows=1)
