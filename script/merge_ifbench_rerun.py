#!/usr/bin/env python3
"""Merge selective IFBench reruns and recompute native metrics."""

import argparse
import json
from pathlib import Path

from opencompass.datasets.IFBench.ifbench import IFBenchEvaluator

try:
    from script.find_ifbench_unparsed import find_unparsed
except ModuleNotFoundError:  # Direct execution adds script/ to sys.path.
    from find_ifbench_unparsed import find_unparsed


def load_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    return rows


def score_rows(rows: dict[str, dict]) -> dict:
    ordered = [
        row for _, row in sorted(rows.items(), key=lambda item: int(item[0]))
    ]
    evaluator = IFBenchEvaluator()
    result = evaluator.score(
        [row.get('prediction', '') for row in ordered],
        [row.get('gold') for row in ordered],
        [row.get('origin_prompt') for row in ordered],
    )
    result.pop('details', None)
    result['total'] = len(ordered)
    result['blank'] = sum(
        not str(row.get('prediction', '')).strip() for row in ordered)
    return result


def merge(base_dir: Path, rerun_dir: Path, output_dir: Path) -> dict:
    if output_dir.resolve() in {base_dir.resolve(), rerun_dir.resolve()}:
        raise SystemExit('Output directory must differ from both input dirs')
    base_path = base_dir / 'IFBench.json'
    rerun_path = rerun_dir / 'IFBench.json'
    if not base_path.is_file():
        raise SystemExit(f'Missing base file {base_path}')
    if not rerun_path.is_file():
        raise SystemExit(f'Missing selective rerun file {rerun_path}')

    indices = find_unparsed(base_dir).get('IFBench', [])
    if not indices:
        raise SystemExit('Base predictions contain no blank answers')
    base_rows = load_rows(base_path)
    rerun_rows = load_rows(rerun_path)
    ordered_reruns = [
        row for _, row in sorted(rerun_rows.items(),
                                 key=lambda item: int(item[0]))
    ]
    if len(ordered_reruns) != len(indices):
        raise SystemExit(
            f'Expected {len(indices)} reruns, got {len(ordered_reruns)}')
    for original_index, replacement in zip(indices, ordered_reruns):
        key = str(original_index)
        if key not in base_rows:
            raise SystemExit(f'Missing base index {key}')
        if base_rows[key].get('gold') != replacement.get('gold'):
            raise SystemExit(f'Gold mismatch at base index {key}')
        base_rows[key] = replacement

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'IFBench.json'
    with output_path.open('x', encoding='utf-8') as file:
        json.dump(base_rows, file, ensure_ascii=False, indent=2)

    before = score_rows(load_rows(base_path))
    after = score_rows(base_rows)
    excluded = {'total', 'blank'}
    metric_gains = {
        metric: after[metric] - before[metric]
        for metric in before if metric not in excluded
    }
    report = {
        'base_prediction_dir': str(base_dir),
        'rerun_prediction_dir': str(rerun_dir),
        'merged_prediction_dir': str(output_dir),
        'replacements': len(indices),
        'before': before,
        'after': after,
        'metric_gains': metric_gains,
        'blank_recovered': before['blank'] - after['blank'],
    }
    report_path = output_dir / 'ifbench_merged_metrics.json'
    with report_path.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('base_prediction_dir', type=Path)
    parser.add_argument('rerun_prediction_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    report = merge(args.base_prediction_dir, args.rerun_prediction_dir,
                   args.output_dir)
    print(json.dumps({
        'replacements': report['replacements'],
        'before': report['before'],
        'after': report['after'],
        'metric_gains': report['metric_gains'],
        'blank_recovered': report['blank_recovered'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
