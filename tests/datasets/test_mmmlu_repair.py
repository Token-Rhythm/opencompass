import json

import pytest

from script.find_mmmlu_unparsed import find_unparsed
from script.merge_mmmlu_rerun import merge, score_prediction_dir


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding='utf-8')


def test_find_and_merge_mmmlu_selective_rerun_uses_official_weighting(tmp_path):
    base = tmp_path / 'base'
    rerun = tmp_path / 'rerun'
    merged = tmp_path / 'merged'
    dump(base / 'openai_mmmlu_AR-XY.json', {
        '0': {'prediction': 'Answer: A', 'gold': 'A'},
        '1': {'prediction': 'unfinished reasoning', 'gold': 'B'},
        '2': {'prediction': 'Answer: C', 'gold': 'D'},
    })
    dump(base / 'openai_mmmlu_ZH-CN.json', {
        '0': {'prediction': '仍在推理', 'gold': 'D'},
    })
    dump(rerun / 'openai_mmmlu_AR-XY.json', {
        '0': {'prediction': 'Answer: B', 'gold': 'B'},
    })
    dump(rerun / 'openai_mmmlu_ZH-CN.json', {
        '0': {'prediction': '答案：Ｄ', 'gold': 'D'},
    })

    assert find_unparsed(base) == {
        'openai_mmmlu_AR-XY': [1],
        'openai_mmmlu_ZH-CN': [0],
    }
    report = merge(base, rerun, merged, require_complete=False)

    assert report['replacements'] == {
        'openai_mmmlu_AR-XY': 1,
        'openai_mmmlu_ZH-CN': 1,
    }
    assert report['before']['official_weighted_accuracy'] == 25
    assert report['after']['official_weighted_accuracy'] == 75
    assert report['after']['language_macro_accuracy'] == pytest.approx(
        (200 / 3 + 100) / 2)
    assert report['unparsed_recovered'] == 2
    assert report['correct_gain'] == 2
    assert (merged / 'mmmlu_merged_metrics.json').is_file()
    assert score_prediction_dir(merged)['total'] == 4


def test_mmmlu_merge_rejects_gold_mismatch(tmp_path):
    base = tmp_path / 'base'
    rerun = tmp_path / 'rerun'
    dump(base / 'openai_mmmlu_AR-XY.json', {
        '0': {'prediction': '', 'gold': 'A'},
    })
    dump(rerun / 'openai_mmmlu_AR-XY.json', {
        '0': {'prediction': 'Answer: B', 'gold': 'B'},
    })

    with pytest.raises(SystemExit, match='gold mismatch'):
        merge(base, rerun, tmp_path / 'merged', require_complete=False)


def test_mmmlu_noop_merge_still_writes_official_weighted_report(tmp_path):
    base = tmp_path / 'base'
    merged = tmp_path / 'merged'
    dump(base / 'openai_mmmlu_AR-XY.json', {
        '0': {'prediction': 'Answer: A', 'gold': 'A'},
        '1': {'prediction': 'Answer: C', 'gold': 'B'},
    })
    dump(base / 'openai_mmmlu_ZH-CN.json', {
        '0': {'prediction': '答案：Ｄ', 'gold': 'D'},
    })

    report = merge(base, base, merged, require_complete=False)

    assert report['replacements'] == {}
    assert report['before']['unparsed'] == 0
    assert report['after']['official_weighted_accuracy'] == pytest.approx(
        200 / 3)
    assert report['official_weighted_accuracy_gain'] == 0
    assert (merged / 'mmmlu_merged_metrics.json').is_file()


def test_mmmlu_strict_merge_rejects_incomplete_language_set(tmp_path):
    base = tmp_path / 'base'
    dump(base / 'openai_mmmlu_AR-XY.json', {
        '0': {
            'prediction': 'Answer: A',
            'gold': 'A'
        },
    })

    with pytest.raises(SystemExit, match='Incomplete MMMLU language set'):
        merge(base, base, tmp_path / 'merged')
