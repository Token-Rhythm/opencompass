#!/usr/bin/env python3
"""Verify Qwen3.5-2B benchmark state and its evidence paths."""

import argparse
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EVIDENCE_TOKENS = {
    'mmlu_pro': 'mmlupro',
    'ceval': 'ceval',
    'supergpqa': 'supergpqa',
    'ifeval': 'ifeval',
    'mmmlu': 'mmmlu',
    'gpqa_diamond': 'gpqa',
    'ifbench': 'ifbench',
    'longbench_v2': 'longbench',
    'aa_lcr': 'aalcr',
    'aime_2024': 'aime2024',
    'aime_2025': 'aime2025',
    'aime_2026': 'aime2026',
    'hmmt_feb_2026': 'hmmtfeb2026',
    'livecodebench_v6': 'livecodebench',
    'mmlu_redux_2': 'mmluredux',
    'mmlu_prox': 'mmluprox',
    'global_piqa': 'globalpiqa',
    'hmmt_feb_2025': 'hmmt2025',
    'hmmt_nov_2025': 'hmmt2025',
    'include_base_44': 'include',
}


def normalized(value: str) -> str:
    return re.sub(r'[^a-z0-9]', '', value.lower())


def verify(state_path: Path,
           repo: Path = REPO,
           require_terminal: bool = False) -> dict:
    with state_path.open(encoding='utf-8') as file:
        state = json.load(file)
    benchmarks = state.get('benchmarks')
    if not isinstance(benchmarks, dict):
        raise SystemExit('Validation state has no benchmark object')

    expected = set(EVIDENCE_TOKENS)
    actual = set(benchmarks)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SystemExit(
            'Validation target mismatch: '
            f'expected={len(expected)}, actual={len(actual)}, '
            f'missing={missing}, extra={extra}')

    completed = []
    skipped = []
    pending = []
    for key in sorted(expected):
        entry = benchmarks[key]
        status = str(entry.get('state', ''))
        evidence = entry.get('evidence')
        if not isinstance(evidence, str) or not evidence:
            raise SystemExit(f'{key}: missing evidence path')
        evidence_path = Path(evidence)
        if not evidence_path.is_absolute():
            evidence_path = repo / evidence_path
        if not evidence_path.exists():
            raise SystemExit(f'{key}: evidence does not exist: {evidence}')
        token = EVIDENCE_TOKENS[key]
        if token not in normalized(evidence):
            raise SystemExit(
                f'{key}: evidence path appears to belong to another '
                f'benchmark: {evidence}')

        if status == 'complete':
            if not isinstance(entry.get('score'), (int, float)):
                raise SystemExit(f'{key}: complete entry has no numeric score')
            if not entry.get('metric') or not entry.get('result_type'):
                raise SystemExit(
                    f'{key}: complete entry lacks metric or result_type')
            if not evidence_path.is_file():
                raise SystemExit(
                    f'{key}: complete evidence must be a file: {evidence}')
            completed.append(key)
        elif status.startswith('skipped_'):
            if not evidence_path.is_file():
                raise SystemExit(
                    f'{key}: skipped evidence must be a file: {evidence}')
            skipped.append(key)
        else:
            pending.append(key)

    if require_terminal and pending:
        raise SystemExit(
            f'Validation is not terminal; pending={pending}')
    return {
        'target_count': len(expected),
        'completed_count': len(completed),
        'audited_skipped_count': len(skipped),
        'pending_count': len(pending),
        'completed': completed,
        'audited_skipped': skipped,
        'pending': pending,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--state', type=Path,
        default=REPO / 'outputs/qwen35_validation/benchmark_results.json')
    parser.add_argument('--require-terminal', action='store_true')
    args = parser.parse_args()
    print(json.dumps(
        verify(args.state, require_terminal=args.require_terminal),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == '__main__':
    main()
