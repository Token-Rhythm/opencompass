import json

import pytest

from script.find_supergpqa_unparsed import find_unparsed
from script.merge_supergpqa_rerun import merge


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding='utf-8')


def test_supergpqa_repair_preserves_option_content_answers(tmp_path):
    test_set = [
        {'options': ['Mercury', 'Venus']},
        {'options': ['one', 'two']},
        {'options': ['red', 'blue']},
    ]
    base = tmp_path / 'base'
    rerun = tmp_path / 'rerun'
    dump(base / 'supergpqa.json', {
        '0': {'prediction': 'The answer is Mercury.', 'gold': 'A'},
        '1': {'prediction': 'Answer: B', 'gold': 'B'},
        '2': {'prediction': 'unfinished', 'gold': 'B'},
    })
    dump(rerun / 'supergpqa.json', {
        '0': {'prediction': 'Answer: B', 'gold': 'B'},
    })

    assert find_unparsed(base, test_set) == {'supergpqa': [2]}
    report = merge(base, rerun, tmp_path / 'merged', test_set)

    assert report['replacements'] == 1
    assert report['before'] == {
        'total': 3,
        'correct': 2,
        'parsed': 2,
        'unparsed': 1,
        'accuracy': 200 / 3,
    }
    assert report['after']['accuracy'] == 100
    assert report['unparsed_recovered'] == 1


def test_supergpqa_noop_merge_still_writes_strict_report(tmp_path):
    test_set = [
        {'options': ['Mercury', 'Venus']},
        {'options': ['one', 'two']},
    ]
    base = tmp_path / 'base'
    merged = tmp_path / 'merged'
    dump(base / 'supergpqa.json', {
        '0': {'prediction': 'Answer: A', 'gold': 'A'},
        '1': {'prediction': 'Answer: A', 'gold': 'B'},
    })

    report = merge(base, base, merged, test_set)

    assert report['replacements'] == 0
    assert report['before']['unparsed'] == 0
    assert report['after']['accuracy'] == 50
    assert report['accuracy_gain'] == 0
    assert (merged / 'supergpqa_merged_metrics.json').is_file()


def test_supergpqa_merge_rejects_incomplete_row_set(tmp_path):
    test_set = [
        {'options': ['Mercury', 'Venus']},
        {'options': ['one', 'two']},
    ]
    base = tmp_path / 'base'
    dump(base / 'supergpqa.json', {
        '0': {
            'prediction': 'Answer: A',
            'gold': 'A'
        },
    })

    with pytest.raises(SystemExit,
                       match='Incomplete SuperGPQA prediction set'):
        merge(base, base, tmp_path / 'merged', test_set)
