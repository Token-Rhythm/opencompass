#!/usr/bin/env python3
"""Merge selective SuperGPQA reruns without modifying base predictions."""

import argparse
import json
from pathlib import Path

try:
    from script.find_supergpqa_unparsed import (extract_answer, find_unparsed,
                                                load_test_set)
except ModuleNotFoundError:  # Direct execution adds script/ to sys.path.
    from find_supergpqa_unparsed import (extract_answer, find_unparsed,
                                         load_test_set)


def load_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    return rows


def validate_row_coverage(rows: dict[str, dict], test_set: list) -> None:
    try:
        actual = {int(index) for index in rows}
    except ValueError as error:
        raise SystemExit(f'Non-integer SuperGPQA prediction index: {error}')
    expected = set(range(len(test_set)))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SystemExit(
            'Incomplete SuperGPQA prediction set: '
            f'expected={len(expected)}, actual={len(actual)}, '
            f'missing={missing}, extra={extra}')


def score_rows(rows: dict[str, dict], test_set=None) -> dict:
    if test_set is None:
        test_set = load_test_set()
    validate_row_coverage(rows, test_set)
    total = correct = parsed = 0
    for raw_index, row in sorted(rows.items(), key=lambda item: int(item[0])):
        index = int(raw_index)
        if index >= len(test_set):
            raise SystemExit(f'Prediction index {index} exceeds test set')
        answer = extract_answer(str(row.get('prediction', '')),
                                test_set[index]['options'])
        reference = str(row.get('gold', '')).upper()
        total += 1
        parsed += answer is not None
        correct += answer == reference
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
          test_set=None) -> dict:
    if output_dir.resolve() in {base_dir.resolve(), rerun_dir.resolve()}:
        raise SystemExit('Output directory must differ from both input dirs')
    base_path = base_dir / 'supergpqa.json'
    rerun_path = rerun_dir / 'supergpqa.json'
    if not base_path.is_file():
        raise SystemExit(f'Missing base file {base_path}')

    if test_set is None:
        test_set = load_test_set()
    base_rows = load_rows(base_path)
    validate_row_coverage(base_rows, test_set)
    indices = find_unparsed(base_dir, test_set).get('supergpqa', [])
    if indices:
        if not rerun_path.is_file():
            raise SystemExit(f'Missing selective rerun file {rerun_path}')
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
            if str(base_rows[key].get('gold')) != str(
                    replacement.get('gold')):
                raise SystemExit(f'Gold mismatch at base index {key}')
            base_rows[key] = replacement

    validate_row_coverage(base_rows, test_set)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'supergpqa.json'
    with output_path.open('x', encoding='utf-8') as file:
        json.dump(base_rows, file, ensure_ascii=False, indent=2)

    before = score_rows(load_rows(base_path), test_set)
    after = score_rows(base_rows, test_set)
    report = {
        'base_prediction_dir': str(base_dir),
        'rerun_prediction_dir': str(rerun_dir),
        'merged_prediction_dir': str(output_dir),
        'replacements': len(indices),
        'before': before,
        'after': after,
        'accuracy_gain': after['accuracy'] - before['accuracy'],
        'correct_gain': after['correct'] - before['correct'],
        'unparsed_recovered': before['unparsed'] - after['unparsed'],
    }
    report_path = output_dir / 'supergpqa_merged_metrics.json'
    with report_path.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('base_prediction_dir', type=Path)
    parser.add_argument('rerun_prediction_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--samples-per-discipline-difficulty', type=int)
    args = parser.parse_args()
    test_set = load_test_set(args.samples_per_discipline_difficulty)
    report = merge(args.base_prediction_dir, args.rerun_prediction_dir,
                   args.output_dir, test_set)
    print(json.dumps({
        'replacements': report['replacements'],
        'before': report['before'],
        'after': report['after'],
        'accuracy_gain': report['accuracy_gain'],
        'correct_gain': report['correct_gain'],
        'unparsed_recovered': report['unparsed_recovered'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
