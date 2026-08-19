#!/usr/bin/env python3
"""Print SuperGPQA indices whose final answer is empty or unparseable."""

import argparse
import json
from pathlib import Path

from opencompass.datasets.supergpqa.supergpqa import SuperGPQADataset
from opencompass.datasets.supergpqa.supergpqa_eval import (
    extract_option_content, extract_option_labels)


def load_test_set(samples_per_discipline_difficulty=None):
    # Keep selective recovery on exactly the same pinned source and prompt mode
    # as the benchmark config.
    from opencompass.configs.datasets.supergpqa.supergpqa_gen import (
        supergpqa_dataset,
    )
    return SuperGPQADataset.load(
        path=supergpqa_dataset['path'],
        prompt_mode=supergpqa_dataset['prompt_mode'],
        hf_revision=supergpqa_dataset['hf_revision'],
        samples_per_discipline_difficulty=(
            samples_per_discipline_difficulty),
    )


def extract_answer(prediction: str, options: list[str]):
    answer = extract_option_labels(prediction, 'ABCDEFGHIJ')
    if answer is None:
        content = extract_option_content(prediction, options)
        if content:
            answer = chr(options.index(content) + 65)
    return answer


def find_unparsed(prediction_dir: Path,
                  test_set=None) -> dict[str, list[int]]:
    path = prediction_dir / 'supergpqa.json'
    if not path.is_file():
        raise SystemExit(f'Missing final prediction file {path}')
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')

    if test_set is None:
        test_set = load_test_set()
    indices = []
    for raw_index, row in sorted(rows.items(), key=lambda item: int(item[0])):
        index = int(raw_index)
        if index >= len(test_set):
            raise SystemExit(f'Prediction index {index} exceeds test set')
        prediction = str(row.get('prediction', ''))
        if extract_answer(prediction, test_set[index]['options']) is None:
            indices.append(index)
    return {'supergpqa': indices} if indices else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_dir', type=Path)
    parser.add_argument('--samples-per-discipline-difficulty', type=int)
    parser.add_argument('--pretty', action='store_true')
    args = parser.parse_args()
    test_set = load_test_set(args.samples_per_discipline_difficulty)
    ranges = find_unparsed(args.prediction_dir, test_set)
    print(json.dumps(ranges,
                     ensure_ascii=False,
                     indent=2 if args.pretty else None,
                     separators=None if args.pretty else (',', ':')))


if __name__ == '__main__':
    main()
