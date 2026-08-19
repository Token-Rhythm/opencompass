#!/usr/bin/env python3
"""Print explicit GPQA indices whose final answer is unparseable."""

import argparse
import json
from pathlib import Path

from opencompass.datasets.gpqa import GPQA_Simple_Eval_postprocess


def find_unparsed(prediction_dir: Path) -> dict[str, list[int]]:
    path = prediction_dir / 'GPQA_diamond.json'
    if not path.is_file():
        raise SystemExit(f'Missing final prediction file {path}')
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    indices = [
        int(raw_index)
        for raw_index, row in sorted(rows.items(), key=lambda item: int(item[0]))
        if not GPQA_Simple_Eval_postprocess(row.get('prediction', ''))
    ]
    return {'GPQA_diamond': indices} if indices else {}


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
