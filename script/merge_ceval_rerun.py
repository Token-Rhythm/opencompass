#!/usr/bin/env python3
"""Merge selective C-Eval reruns without modifying base predictions."""

import argparse
import json
from pathlib import Path

try:
    from script.find_ceval_unparsed import find_unparsed
except ModuleNotFoundError:  # Direct ``python script/...`` execution.
    from find_ceval_unparsed import find_unparsed
from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import (
    ceval_subject_mapping,
)
from opencompass.datasets.ceval import ceval_evalscope_answer_postprocess


def load_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    return rows


def validate_complete_subject_set(paths: list[Path]) -> None:
    expected = {f'ceval-{subject}' for subject in ceval_subject_mapping}
    actual = {path.stem for path in paths}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SystemExit(
            'Incomplete C-Eval prediction set: '
            f'expected={len(expected)}, actual={len(actual)}, '
            f'missing={missing}, extra={extra}')


def score_prediction_dir(prediction_dir: Path) -> dict:
    subjects = {}
    total = correct = parsed = 0
    for path in sorted(prediction_dir.glob('ceval-*.json')):
        rows = load_rows(path)
        subject_correct = subject_parsed = 0
        for row in rows.values():
            answer = ceval_evalscope_answer_postprocess(
                row.get('prediction', ''))
            reference = str(row.get('gold', '')).upper()
            subject_parsed += bool(answer)
            subject_correct += answer == reference
        count = len(rows)
        subject = path.stem.removeprefix('ceval-')
        subjects[path.stem] = {
            'group': ceval_subject_mapping[subject][2],
            'total': count,
            'correct': subject_correct,
            'parsed': subject_parsed,
            'unparsed': count - subject_parsed,
            'accuracy': 100 * subject_correct / count if count else 0,
        }
        total += count
        correct += subject_correct
        parsed += subject_parsed

    accuracies = [entry['accuracy'] for entry in subjects.values()]
    groups = {}
    for group in sorted({entry['group'] for entry in subjects.values()}):
        values = [entry['accuracy'] for entry in subjects.values()
                  if entry['group'] == group]
        groups[group] = sum(values) / len(values)
    return {
        'total': total,
        'correct': correct,
        'parsed': parsed,
        'unparsed': total - parsed,
        'micro_accuracy': 100 * correct / total if total else 0,
        'macro_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
        'group_macro_accuracy': groups,
        'subjects': subjects,
    }


def merge(base_dir: Path,
          rerun_dir: Path,
          output_dir: Path,
          require_complete: bool = True) -> dict:
    if output_dir.resolve() in {base_dir.resolve(), rerun_dir.resolve()}:
        raise SystemExit('Output directory must differ from both input dirs')
    base_paths = sorted(base_dir.glob('ceval-*.json'))
    if not base_paths:
        raise SystemExit(f'No final ceval-*.json files in {base_dir}')
    if require_complete:
        validate_complete_subject_set(base_paths)
    ranges = find_unparsed(base_dir)

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

    output_paths = sorted(output_dir.glob('ceval-*.json'))
    if require_complete:
        validate_complete_subject_set(output_paths)
    before = score_prediction_dir(base_dir)
    after = score_prediction_dir(output_dir)
    report = {
        'base_prediction_dir': str(base_dir),
        'rerun_prediction_dir': str(rerun_dir),
        'merged_prediction_dir': str(output_dir),
        'replacements': replacements,
        'before': before,
        'after': after,
        'micro_accuracy_gain': (
            after['micro_accuracy'] - before['micro_accuracy']),
        'macro_accuracy_gain': (
            after['macro_accuracy'] - before['macro_accuracy']),
        'correct_gain': after['correct'] - before['correct'],
        'unparsed_recovered': before['unparsed'] - after['unparsed'],
    }
    report_path = output_dir / 'ceval_merged_metrics.json'
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
        'before_micro_accuracy': report['before']['micro_accuracy'],
        'after_micro_accuracy': report['after']['micro_accuracy'],
        'before_macro_accuracy': report['before']['macro_accuracy'],
        'after_macro_accuracy': report['after']['macro_accuracy'],
        'micro_accuracy_gain': report['micro_accuracy_gain'],
        'macro_accuracy_gain': report['macro_accuracy_gain'],
        'correct_gain': report['correct_gain'],
        'unparsed_recovered': report['unparsed_recovered'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
