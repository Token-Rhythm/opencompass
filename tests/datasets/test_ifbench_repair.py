import json
from types import SimpleNamespace

from opencompass.datasets.IFBench import ifbench
from script.find_ifbench_unparsed import find_unparsed
from script.merge_ifbench_rerun import merge


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding='utf-8')


def test_ifbench_blank_repair_recomputes_native_metrics(tmp_path, monkeypatch):
    def evaluate(_, prediction):
        return SimpleNamespace(
            follow_instruction_list=[bool(prediction)],
            instruction_id_list=['test:instruction'],
        )

    monkeypatch.setattr(ifbench, 'test_instruction_following_strict', evaluate)
    monkeypatch.setattr(ifbench, 'test_instruction_following_loose', evaluate)
    reference = {
        'key': 1,
        'instruction_id_list': ['test:instruction'],
        'prompt': 'prompt',
        'kwargs': [{}],
    }
    base = tmp_path / 'base'
    rerun = tmp_path / 'rerun'
    dump(base / 'IFBench.json', {
        '0': {'prediction': 'valid', 'gold': reference,
              'origin_prompt': 'prompt'},
        '1': {'prediction': '', 'gold': reference,
              'origin_prompt': 'prompt'},
    })
    dump(rerun / 'IFBench.json', {
        '0': {'prediction': 'recovered', 'gold': reference,
              'origin_prompt': 'prompt'},
    })

    assert find_unparsed(base) == {'IFBench': [1]}
    report = merge(base, rerun, tmp_path / 'merged')

    assert report['replacements'] == 1
    assert report['before']['score'] == 50
    assert report['after']['score'] == 100
    assert report['blank_recovered'] == 1
