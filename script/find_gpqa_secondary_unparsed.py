#!/usr/bin/env python3
"""Map still-unparsed GPQA repair rows back to original dataset indices."""

import argparse
import json
from pathlib import Path

try:
    from script.find_gpqa_unparsed import find_unparsed
except ModuleNotFoundError:  # Direct ``python script/...`` execution.
    from find_gpqa_unparsed import find_unparsed
from opencompass.datasets.gpqa import GPQA_Simple_Eval_postprocess


def _load_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding='utf-8') as file:
        rows = json.load(file)
    if not isinstance(rows, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    return rows


def find_secondary_unparsed(base_dir: Path,
                            primary_rerun_dir: Path) -> dict[str, list[int]]:
    """Return original GPQA indices still unparsed after the primary repair."""
    original_indices = find_unparsed(base_dir).get('GPQA_diamond', [])
    rerun_path = primary_rerun_dir / 'GPQA_diamond.json'
    if not rerun_path.is_file():
        raise SystemExit(f'Missing primary rerun file {rerun_path}')
    rerun_rows = _load_rows(rerun_path)
    ordered_rows = [
        row for _, row in sorted(rerun_rows.items(),
                                 key=lambda item: int(item[0]))
    ]
    if len(ordered_rows) != len(original_indices):
        raise SystemExit(
            f'Expected {len(original_indices)} primary reruns, got '
            f'{len(ordered_rows)}')
    remaining = [
        original_index
        for original_index, row in zip(original_indices, ordered_rows)
        if not GPQA_Simple_Eval_postprocess(row.get('prediction', ''))
    ]
    return {'GPQA_diamond': remaining} if remaining else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('base_prediction_dir', type=Path)
    parser.add_argument('primary_rerun_prediction_dir', type=Path)
    parser.add_argument('--pretty', action='store_true')
    args = parser.parse_args()
    result = find_secondary_unparsed(args.base_prediction_dir,
                                     args.primary_rerun_prediction_dir)
    print(json.dumps(result,
                     ensure_ascii=False,
                     indent=2 if args.pretty else None,
                     separators=None if args.pretty else (',', ':')))


if __name__ == '__main__':
    main()
