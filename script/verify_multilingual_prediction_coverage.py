#!/usr/bin/env python3
"""Verify exact coverage for sampled multilingual benchmark runs."""

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> dict:
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    return rows


def verify_global_piqa(prediction_dir: Path, expected_rows: int = 267) -> dict:
    path = prediction_dir / 'global_piqa_generation.json'
    if not path.is_file():
        raise SystemExit(f'Missing Global PIQA prediction file: {path}')
    actual = len(load_rows(path))
    if actual != expected_rows:
        raise SystemExit(
            f'Incomplete Global PIQA predictions: expected={expected_rows}, '
            f'actual={actual}')
    return {'datasets': 1, 'rows': actual}


def verify_mmlu_prox(prediction_dir: Path,
                     expected_abbrs: set[str] = None,
                     expected_rows_per_dataset: int = 1) -> dict:
    if expected_abbrs is None:
        from opencompass.configs.datasets.mmlu_prox.mmlu_prox_5shot_cot_gen import (
            mmlu_prox_5shot_datasets,
        )
        expected_abbrs = {dataset['abbr']
                          for dataset in mmlu_prox_5shot_datasets}

    paths = sorted(prediction_dir.glob('mmlu_prox_5shot_*.json'))
    actual_abbrs = {path.stem for path in paths}
    missing = sorted(expected_abbrs - actual_abbrs)
    extra = sorted(actual_abbrs - expected_abbrs)
    if missing or extra:
        raise SystemExit(
            'Incomplete MMLU-ProX dataset set: '
            f'expected={len(expected_abbrs)}, actual={len(actual_abbrs)}, '
            f'missing={missing}, extra={extra}')

    wrong_sizes = {}
    total = 0
    for path in paths:
        size = len(load_rows(path))
        total += size
        if size != expected_rows_per_dataset:
            wrong_sizes[path.stem] = size
    if wrong_sizes:
        raise SystemExit(
            'Incomplete MMLU-ProX per-dataset samples: '
            f'expected_each={expected_rows_per_dataset}, '
            f'actual={wrong_sizes}')
    return {'datasets': len(paths), 'rows': total}


def verify_include(prediction_dir: Path,
                   expected_abbrs: set[str] = None,
                   expected_total_rows: int = 22639) -> dict:
    if expected_abbrs is None:
        from opencompass.configs.datasets.include.include_base_44_0shot_ppl import (
            include_datasets,
        )
        expected_abbrs = {dataset['abbr'] for dataset in include_datasets}

    paths = sorted(prediction_dir.glob('include_base_44_*.json'))
    # The language-benchmark launcher appends its run-mode suffix to every
    # dataset abbreviation. INCLUDE is routed through raw completions even
    # though the shared run suffix is ``full_chat``/``smoke_chat``, so compare
    # canonical dataset abbreviations rather than literal prediction stems.
    canonical_paths = {}
    for path in paths:
        canonical_abbr = path.stem
        for suffix in ('_full_chat', '_smoke_chat'):
            if canonical_abbr.endswith(suffix):
                canonical_abbr = canonical_abbr[:-len(suffix)]
                break
        if canonical_abbr in canonical_paths:
            raise SystemExit(
                'Duplicate INCLUDE prediction dataset after run-suffix '
                f'normalization: {canonical_abbr}')
        canonical_paths[canonical_abbr] = path

    actual_abbrs = set(canonical_paths)
    missing = sorted(expected_abbrs - actual_abbrs)
    extra = sorted(actual_abbrs - expected_abbrs)
    if missing or extra:
        raise SystemExit(
            'Incomplete INCLUDE language set: '
            f'expected={len(expected_abbrs)}, actual={len(actual_abbrs)}, '
            f'missing={missing}, extra={extra}')

    total = sum(len(load_rows(path)) for path in canonical_paths.values())
    if total != expected_total_rows:
        raise SystemExit(
            f'Incomplete INCLUDE predictions: expected={expected_total_rows}, '
            f'actual={total}')
    return {'datasets': len(paths), 'rows': total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'benchmark', choices=('global_piqa', 'mmlu_prox', 'include'))
    parser.add_argument('prediction_dir', type=Path)
    args = parser.parse_args()

    if args.benchmark == 'global_piqa':
        result = verify_global_piqa(args.prediction_dir)
    elif args.benchmark == 'mmlu_prox':
        result = verify_mmlu_prox(args.prediction_dir)
    else:
        result = verify_include(args.prediction_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
