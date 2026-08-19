#!/usr/bin/env python3
"""Print explicit MMLU-Pro indices whose final answer is unparseable."""

import argparse
import json
import re
from pathlib import Path


ANSWER_PATTERN = re.compile(r'answer is \(?([A-J])\)?', re.IGNORECASE)


def find_unparsed(prediction_dir: Path) -> dict[str, list[int]]:
    ranges = {}
    paths = [
        path for path in sorted(prediction_dir.glob('mmlu_pro_*.json'))
        if path.stem != 'mmlu_pro_merged_metrics'
    ]
    if not paths:
        raise SystemExit(f'No final mmlu_pro_*.json files in {prediction_dir}')
    for path in paths:
        with path.open(encoding='utf-8') as file:
            rows = json.load(file)
        if not isinstance(rows, dict):
            raise SystemExit(f'Expected a JSON object in {path}')
        indices = []
        for raw_index, row in sorted(rows.items(), key=lambda item: int(item[0])):
            prediction = row.get('prediction', '')
            if not ANSWER_PATTERN.findall(str(prediction)):
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
