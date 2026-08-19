import json

import pytest

from script.verify_qwen35_validation_state import (EVIDENCE_TOKENS,
                                                    verify)


def make_state(tmp_path, status='complete'):
    entries = {}
    for key, token in EVIDENCE_TOKENS.items():
        evidence = tmp_path / f'{token}.json'
        evidence.write_text('{}', encoding='utf-8')
        entries[key] = {
            'state': status,
            'score': 1.0 if status == 'complete' else None,
            'metric': 'accuracy' if status == 'complete' else None,
            'result_type': 'test' if status == 'complete' else None,
            'evidence': evidence.name,
        }
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'benchmarks': entries}), encoding='utf-8')
    return state, entries


def test_validation_state_accepts_exact_terminal_evidence(tmp_path):
    state, _ = make_state(tmp_path)
    result = verify(state, repo=tmp_path, require_terminal=True)
    assert result['target_count'] == 20
    assert result['completed_count'] == 20
    assert result['pending_count'] == 0


def test_validation_state_rejects_missing_target(tmp_path):
    state, entries = make_state(tmp_path)
    entries.pop('mmmlu')
    state.write_text(json.dumps({'benchmarks': entries}), encoding='utf-8')
    with pytest.raises(SystemExit, match='target mismatch'):
        verify(state, repo=tmp_path)


def test_validation_state_rejects_crossed_evidence(tmp_path):
    state, entries = make_state(tmp_path)
    entries['mmmlu']['evidence'] = entries['mmlu_prox']['evidence']
    state.write_text(json.dumps({'benchmarks': entries}), encoding='utf-8')
    with pytest.raises(SystemExit, match='another benchmark'):
        verify(state, repo=tmp_path)


def test_validation_state_terminal_gate_rejects_pending(tmp_path):
    state, entries = make_state(tmp_path)
    entries['global_piqa'].update(
        state='running', score=None, metric=None, result_type=None)
    state.write_text(json.dumps({'benchmarks': entries}), encoding='utf-8')
    with pytest.raises(SystemExit, match='not terminal'):
        verify(state, repo=tmp_path, require_terminal=True)
