#!/usr/bin/env python3
"""Merge selective GPQA reruns without modifying base predictions."""

import argparse
import json
from pathlib import Path

try:
    from script.find_gpqa_secondary_unparsed import find_secondary_unparsed
    from script.find_gpqa_unparsed import find_unparsed
except ModuleNotFoundError:  # Direct ``python script/...`` execution.
    from find_gpqa_secondary_unparsed import find_secondary_unparsed
    from find_gpqa_unparsed import find_unparsed
from opencompass.datasets.gpqa import GPQA_Simple_Eval_postprocess


def load_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    return rows


def score_rows(rows: dict[str, dict]) -> dict:
    correct = parsed = 0
    for row in rows.values():
        answer = GPQA_Simple_Eval_postprocess(row.get('prediction', ''))
        reference = str(row.get('gold', '')).upper()
        parsed += bool(answer)
        correct += answer == reference
    total = len(rows)
    return {
        'total': total,
        'correct': correct,
        'parsed': parsed,
        'unparsed': total - parsed,
        'accuracy': 100 * correct / total if total else 0,
    }


def merge(base_dir: Path,
          rerun_dir: Path,
          output_dir: Path,
          secondary_rerun_dir: Path | None = None) -> dict:
    input_dirs = {base_dir.resolve(), rerun_dir.resolve()}
    if secondary_rerun_dir is not None:
        input_dirs.add(secondary_rerun_dir.resolve())
    if output_dir.resolve() in input_dirs:
        raise SystemExit('Output directory must differ from both input dirs')
    base_path = base_dir / 'GPQA_diamond.json'
    rerun_path = rerun_dir / 'GPQA_diamond.json'
    if not base_path.is_file():
        raise SystemExit(f'Missing base file {base_path}')
    if not rerun_path.is_file():
        raise SystemExit(f'Missing selective rerun file {rerun_path}')

    ranges = find_unparsed(base_dir)
    indices = ranges.get('GPQA_diamond', [])
    if not indices:
        raise SystemExit('Base predictions contain no unparsed answers')
    base_rows = load_rows(base_path)
    rerun_rows = load_rows(rerun_path)
    ordered_reruns = [
        row for _, row in sorted(rerun_rows.items(),
                                 key=lambda item: int(item[0]))
    ]
    if len(ordered_reruns) != len(indices):
        raise SystemExit(
            f'Expected {len(indices)} reruns, got {len(ordered_reruns)}')

    secondary_indices = []
    if secondary_rerun_dir is not None:
        secondary_indices = find_secondary_unparsed(
            base_dir, rerun_dir).get('GPQA_diamond', [])
        secondary_path = secondary_rerun_dir / 'GPQA_diamond.json'
        if secondary_indices and not secondary_path.is_file():
            raise SystemExit(f'Missing secondary rerun file {secondary_path}')
        secondary_rows = (load_rows(secondary_path)
                          if secondary_indices else {})
        ordered_secondary = [
            row for _, row in sorted(secondary_rows.items(),
                                     key=lambda item: int(item[0]))
        ]
        if len(ordered_secondary) != len(secondary_indices):
            raise SystemExit(
                f'Expected {len(secondary_indices)} secondary reruns, got '
                f'{len(ordered_secondary)}')
        local_positions = {
            original_index: position
            for position, original_index in enumerate(indices)
        }
        for original_index, replacement in zip(secondary_indices,
                                               ordered_secondary):
            position = local_positions[original_index]
            if str(ordered_reruns[position].get('gold')) != str(
                    replacement.get('gold')):
                raise SystemExit(
                    f'Secondary gold mismatch at base index {original_index}')
            ordered_reruns[position] = replacement
    for original_index, replacement in zip(indices, ordered_reruns):
        key = str(original_index)
        if key not in base_rows:
            raise SystemExit(f'Missing base index {key}')
        if str(base_rows[key].get('gold')) != str(replacement.get('gold')):
            raise SystemExit(f'Gold mismatch at base index {key}')
        base_rows[key] = replacement

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'GPQA_diamond.json'
    with output_path.open('x', encoding='utf-8') as file:
        json.dump(base_rows, file, ensure_ascii=False, indent=2)

    before = score_rows(load_rows(base_path))
    after = score_rows(base_rows)
    report = {
        'base_prediction_dir': str(base_dir),
        'rerun_prediction_dir': str(rerun_dir),
        'secondary_rerun_prediction_dir': (
            str(secondary_rerun_dir)
            if secondary_rerun_dir is not None else None),
        'merged_prediction_dir': str(output_dir),
        'replacements': len(indices),
        'secondary_replacements': len(secondary_indices),
        'before': before,
        'after': after,
        'accuracy_gain': after['accuracy'] - before['accuracy'],
        'correct_gain': after['correct'] - before['correct'],
        'unparsed_recovered': before['unparsed'] - after['unparsed'],
    }
    report_path = output_dir / 'gpqa_merged_metrics.json'
    with report_path.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('base_prediction_dir', type=Path)
    parser.add_argument('rerun_prediction_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--secondary-rerun-dir', type=Path)
    args = parser.parse_args()
    report = merge(args.base_prediction_dir, args.rerun_prediction_dir,
                   args.output_dir, args.secondary_rerun_dir)
    print(json.dumps({
        'replacements': report['replacements'],
        'secondary_replacements': report['secondary_replacements'],
        'before_accuracy': report['before']['accuracy'],
        'after_accuracy': report['after']['accuracy'],
        'accuracy_gain': report['accuracy_gain'],
        'correct_gain': report['correct_gain'],
        'unparsed_recovered': report['unparsed_recovered'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
