import json

import pytest

from script.merge_ceval_rerun import merge


def test_ceval_noop_merge_still_writes_strict_official_report(tmp_path):
    base = tmp_path / 'base'
    output = tmp_path / 'output'
    base.mkdir()
    rows = {
        '0': {'prediction': '推理。\n答案：A', 'gold': 'A'},
        '1': {'prediction': '推理。\n答案：B', 'gold': 'B'},
    }
    (base / 'ceval-operating_system.json').write_text(
        json.dumps(rows, ensure_ascii=False), encoding='utf-8')

    report = merge(base, base, output, require_complete=False)

    assert report['replacements'] == {}
    assert report['before']['micro_accuracy'] == 100
    assert report['after']['micro_accuracy'] == 100
    assert report['after']['unparsed'] == 0
    assert (output / 'ceval_merged_metrics.json').is_file()


def test_ceval_strict_merge_rejects_incomplete_subject_set(tmp_path):
    base = tmp_path / 'base'
    output = tmp_path / 'output'
    base.mkdir()
    (base / 'ceval-operating_system.json').write_text(
        json.dumps({'0': {
            'prediction': 'Answer: A',
            'gold': 'A'
        }}),
        encoding='utf-8')

    with pytest.raises(SystemExit, match='Incomplete C-Eval prediction set'):
        merge(base, base, output)
