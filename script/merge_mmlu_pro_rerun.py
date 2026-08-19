#!/usr/bin/env python3
"""Merge selective MMLU-Pro reruns without modifying base predictions."""

import argparse
import json
import re
from pathlib import Path

from find_mmlu_pro_unparsed import find_unparsed


ANSWER_PATTERN = re.compile(r'answer is \(?([A-J])\)?', re.IGNORECASE)


def load_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    return rows


def score_prediction_dir(prediction_dir: Path) -> dict:
    categories = {}
    total = correct = parsed = 0
    paths = [
        path for path in sorted(prediction_dir.glob('mmlu_pro_*.json'))
        if path.stem != 'mmlu_pro_merged_metrics'
    ]
    for path in paths:
        rows = load_rows(path)
        category_correct = category_parsed = 0
        for row in rows.values():
            matches = ANSWER_PATTERN.findall(str(row.get('prediction', '')))
            answer = matches[-1].upper() if matches else ''
            reference = str(row.get('gold', '')).upper()
            category_parsed += bool(answer)
            category_correct += answer == reference
        count = len(rows)
        categories[path.stem] = {
            'total': count,
            'correct': category_correct,
            'parsed': category_parsed,
            'unparsed': count - category_parsed,
            'accuracy': 100 * category_correct / count if count else 0,
        }
        total += count
        correct += category_correct
        parsed += category_parsed
    return {
        'total': total,
        'correct': correct,
        'parsed': parsed,
        'unparsed': total - parsed,
        'accuracy': 100 * correct / total if total else 0,
        'categories': categories,
    }


def merge(base_dir: Path, rerun_dir: Path, output_dir: Path) -> dict:
    if output_dir.resolve() in {base_dir.resolve(), rerun_dir.resolve()}:
        raise SystemExit('Output directory must differ from both input dirs')
    base_paths = [
        path for path in sorted(base_dir.glob('mmlu_pro_*.json'))
        if path.stem != 'mmlu_pro_merged_metrics'
    ]
    if not base_paths:
        raise SystemExit(f'No final mmlu_pro_*.json files in {base_dir}')
    ranges = find_unparsed(base_dir)
    if not ranges:
        raise SystemExit('Base predictions contain no unparsed answers')

    output_dir.mkdir(parents=True, exist_ok=True)
    replacements = {}
    for base_path in base_paths:
        target_path = output_dir / base_path.name
        if target_path.exists():
            raise SystemExit(f'Refusing to overwrite {target_path}')
        base_rows = load_rows(base_path)
        indices = ranges.get(base_path.stem, [])
        if indices:
            rerun_path = rerun_dir / base_path.name
            if not rerun_path.is_file():
                raise SystemExit(f'Missing selective rerun file {rerun_path}')
            rerun_rows = load_rows(rerun_path)
            ordered_reruns = [
                row for _, row in sorted(rerun_rows.items(),
                                         key=lambda item: int(item[0]))
            ]
            if len(ordered_reruns) != len(indices):
                raise SystemExit(
                    f'{base_path.stem}: expected {len(indices)} reruns, got '
                    f'{len(ordered_reruns)}')
            for original_index, replacement in zip(indices, ordered_reruns):
                key = str(original_index)
                if key not in base_rows:
                    raise SystemExit(
                        f'{base_path.stem}: missing base index {key}')
                if str(base_rows[key].get('gold')) != str(
                        replacement.get('gold')):
                    raise SystemExit(
                        f'{base_path.stem}: gold mismatch at base index {key}')
                base_rows[key] = replacement
            replacements[base_path.stem] = len(indices)
        with target_path.open('x', encoding='utf-8') as file:
            json.dump(base_rows, file, ensure_ascii=False, indent=2)

    before = score_prediction_dir(base_dir)
    after = score_prediction_dir(output_dir)
    report = {
        'base_prediction_dir': str(base_dir),
        'rerun_prediction_dir': str(rerun_dir),
        'merged_prediction_dir': str(output_dir),
        'replacements': replacements,
        'before': before,
        'after': after,
        'accuracy_gain': after['accuracy'] - before['accuracy'],
        'correct_gain': after['correct'] - before['correct'],
        'unparsed_recovered': before['unparsed'] - after['unparsed'],
    }
    report_path = output_dir / 'mmlu_pro_merged_metrics.json'
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
        'before_accuracy': report['before']['accuracy'],
        'after_accuracy': report['after']['accuracy'],
        'accuracy_gain': report['accuracy_gain'],
        'correct_gain': report['correct_gain'],
        'unparsed_recovered': report['unparsed_recovered'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
