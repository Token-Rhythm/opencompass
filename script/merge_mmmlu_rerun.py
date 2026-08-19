#!/usr/bin/env python3
"""Merge selective MMMLU reruns without modifying base predictions."""

import argparse
import json
from pathlib import Path

from opencompass.configs.datasets.mmmlu.mmmlu_gen_c51a84 import (
    mmmlu_datasets,
)
from opencompass.datasets.mmmlu import mmmlu_answer_postprocess

try:
    from script.find_mmmlu_unparsed import find_unparsed
except ModuleNotFoundError:  # Direct execution adds script/ to sys.path.
    from find_mmmlu_unparsed import find_unparsed


def load_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    return rows


def validate_complete_prediction_set(
        paths: list[Path], expected_samples_per_language: int = None) -> None:
    expected = {dataset['abbr'] for dataset in mmmlu_datasets}
    actual = {path.stem for path in paths}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SystemExit(
            'Incomplete MMMLU language set: '
            f'expected={len(expected)}, actual={len(actual)}, '
            f'missing={missing}, extra={extra}')
    if expected_samples_per_language is not None:
        wrong_sizes = {
            path.stem: len(load_rows(path))
            for path in paths
            if len(load_rows(path)) != expected_samples_per_language
        }
        if wrong_sizes:
            raise SystemExit(
                'Incomplete MMMLU per-language samples: '
                f'expected_each={expected_samples_per_language}, '
                f'actual={wrong_sizes}')


def score_prediction_dir(prediction_dir: Path) -> dict:
    languages = {}
    total = correct = parsed = 0
    paths = sorted(prediction_dir.glob('openai_mmmlu_*.json'))
    if not paths:
        raise SystemExit(
            f'No final openai_mmmlu_*.json files in {prediction_dir}')
    for path in paths:
        rows = load_rows(path)
        language_correct = language_parsed = 0
        for row in rows.values():
            answer = mmmlu_answer_postprocess(row.get('prediction', ''))
            reference = str(row.get('gold', '')).upper()
            language_parsed += bool(answer)
            language_correct += answer == reference
        count = len(rows)
        languages[path.stem] = {
            'total': count,
            'correct': language_correct,
            'parsed': language_parsed,
            'unparsed': count - language_parsed,
            'accuracy': 100 * language_correct / count if count else 0,
        }
        total += count
        correct += language_correct
        parsed += language_parsed

    language_accuracies = [entry['accuracy']
                           for entry in languages.values()]
    weighted_accuracy = 100 * correct / total if total else 0
    return {
        'total': total,
        'correct': correct,
        'parsed': parsed,
        'unparsed': total - parsed,
        # lm-evaluation-harness' top-level mmmlu group uses
        # weight_by_size=True, which is exactly the micro accuracy here.
        'official_weighted_accuracy': weighted_accuracy,
        'micro_accuracy': weighted_accuracy,
        'language_macro_accuracy': (
            sum(language_accuracies) / len(language_accuracies)
            if language_accuracies else 0),
        'languages': languages,
    }


def merge(base_dir: Path,
          rerun_dir: Path,
          output_dir: Path,
          require_complete: bool = True,
          expected_samples_per_language: int = None) -> dict:
    if output_dir.resolve() in {base_dir.resolve(), rerun_dir.resolve()}:
        raise SystemExit('Output directory must differ from both input dirs')
    base_paths = sorted(base_dir.glob('openai_mmmlu_*.json'))
    if not base_paths:
        raise SystemExit(
            f'No final openai_mmmlu_*.json files in {base_dir}')
    if require_complete:
        validate_complete_prediction_set(base_paths,
                                         expected_samples_per_language)
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
                row for _, row in sorted(
                    rerun_rows.items(), key=lambda item: int(item[0]))
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

    output_paths = sorted(output_dir.glob('openai_mmmlu_*.json'))
    if require_complete:
        validate_complete_prediction_set(output_paths,
                                         expected_samples_per_language)
    before = score_prediction_dir(base_dir)
    after = score_prediction_dir(output_dir)
    report = {
        'base_prediction_dir': str(base_dir),
        'rerun_prediction_dir': str(rerun_dir),
        'merged_prediction_dir': str(output_dir),
        'replacements': replacements,
        'before': before,
        'after': after,
        'official_weighted_accuracy_gain': (
            after['official_weighted_accuracy']
            - before['official_weighted_accuracy']),
        'language_macro_accuracy_gain': (
            after['language_macro_accuracy']
            - before['language_macro_accuracy']),
        'correct_gain': after['correct'] - before['correct'],
        'unparsed_recovered': before['unparsed'] - after['unparsed'],
    }
    report_path = output_dir / 'mmmlu_merged_metrics.json'
    with report_path.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('base_prediction_dir', type=Path)
    parser.add_argument('rerun_prediction_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--expected-samples-per-language', type=int)
    args = parser.parse_args()

    report = merge(
        args.base_prediction_dir,
        args.rerun_prediction_dir,
        args.output_dir,
        expected_samples_per_language=args.expected_samples_per_language)
    print(json.dumps({
        'replacements': report['replacements'],
        'before_official_weighted_accuracy': (
            report['before']['official_weighted_accuracy']),
        'after_official_weighted_accuracy': (
            report['after']['official_weighted_accuracy']),
        'before_language_macro_accuracy': (
            report['before']['language_macro_accuracy']),
        'after_language_macro_accuracy': (
            report['after']['language_macro_accuracy']),
        'official_weighted_accuracy_gain': (
            report['official_weighted_accuracy_gain']),
        'language_macro_accuracy_gain': (
            report['language_macro_accuracy_gain']),
        'correct_gain': report['correct_gain'],
        'unparsed_recovered': report['unparsed_recovered'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
