#!/usr/bin/env python3
"""Print explicit C-Eval indices whose final answer is unparseable."""

import argparse
import json
from pathlib import Path

from opencompass.datasets.ceval import ceval_evalscope_answer_postprocess


def find_unparsed(prediction_dir: Path) -> dict[str, list[int]]:
    ranges = {}
    paths = sorted(prediction_dir.glob('ceval-*.json'))
    if not paths:
        raise SystemExit(f'No final ceval-*.json files in {prediction_dir}')
    for path in paths:
        with path.open(encoding='utf-8') as file:
            rows = json.load(file)
        if not isinstance(rows, dict):
            raise SystemExit(f'Expected a JSON object in {path}')
        indices = []
        for raw_index, row in sorted(rows.items(), key=lambda item: int(item[0])):
            if not ceval_evalscope_answer_postprocess(
                    row.get('prediction', '')):
                indices.append(int(raw_index))
        if indices:
            ranges[path.stem] = indices
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_dir', type=Path)
    parser.add_argument('--pretty', action='store_true')
    args = parser.parse_args()

    ranges = find_unparsed(args.prediction_dir)
    print(json.dumps(ranges,
                     ensure_ascii=False,
                     indent=2 if args.pretty else None,
                     separators=None if args.pretty else (',', ':')))


if __name__ == '__main__':
    main()
