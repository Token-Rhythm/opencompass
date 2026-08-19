from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Hashable

from mmengine import Config
from mmengine.config import ConfigDict

from opencompass.utils import build_dataset_from_cfg


DEFAULT_SEED = 20260801
DEFAULT_SAMPLE_BUDGET = 1400


BENCHMARKS = {
    "mmmlu": {
        "config": "opencompass/configs/datasets/mmmlu/mmmlu_gen.py",
        "variable": "mmmlu_datasets",
        "stratification": "language_config x subject",
        "group": lambda row, abbr: (abbr, row["subject"]),
    },
    "mmlu_prox": {
        "config": (
            "opencompass/configs/datasets/mmlu_prox/"
            "mmlu_prox_5shot_cot_gen.py"
        ),
        "variable": "mmlu_prox_5shot_datasets",
        "stratification": "language_config x category",
        "group": lambda row, abbr: abbr,
    },
    "mmlu_redux": {
        "config": (
            "opencompass/configs/datasets/mmlu_redux/mmlu_redux_gen.py"
        ),
        "variable": "mmlu_redux_datasets",
        "stratification": "subject",
        "group": lambda row, abbr: row["subject"],
    },
    "global_piqa": {
        "config": (
            "opencompass/configs/datasets/global_piqa/"
            "global_piqa_generation.py"
        ),
        "variable": "global_piqa_datasets",
        "stratification": "variant x language_config",
        "group": lambda row, abbr: (
            row["variant"],
            row["language_config"],
        ),
    },
    "include": {
        "config": (
            "opencompass/configs/datasets/include/"
            "include_base_44_0shot_ppl.py"
        ),
        "variable": "include_datasets",
        "stratification": "language_config",
        "group": lambda row, abbr: abbr,
    },
}


def _stable_seed(seed: int, benchmark: str, dataset: str, group: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{benchmark}:{dataset}:{group}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def allocate_proportionally(
    counts: dict[str, int], target: int
) -> dict[str, int]:
    """Hamilton allocation with at least one sample in every nonempty stratum."""
    if target <= 0:
        raise ValueError("target must be positive")
    counts = {name: count for name, count in counts.items() if count > 0}
    source_total = sum(counts.values())
    if target >= source_total:
        return dict(counts)
    if target < len(counts):
        raise ValueError(
            f"target {target} cannot cover all {len(counts)} strata"
        )

    exact = {
        name: target * count / source_total for name, count in counts.items()
    }
    allocation = {
        name: min(counts[name], max(1, math.floor(value)))
        for name, value in exact.items()
    }

    while sum(allocation.values()) < target:
        candidates = [
            name for name in counts if allocation[name] < counts[name]
        ]
        name = max(
            candidates,
            key=lambda item: (
                exact[item] - allocation[item],
                counts[item],
                item,
            ),
        )
        allocation[name] += 1

    while sum(allocation.values()) > target:
        candidates = [name for name in counts if allocation[name] > 1]
        name = min(
            candidates,
            key=lambda item: (
                exact[item] - allocation[item],
                -allocation[item],
                item,
            ),
        )
        allocation[name] -= 1

    return allocation


def _group_indices(
    rows,
    dataset_abbr: str,
    group_fn: Callable[[dict, str], Hashable],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = group_fn(row, dataset_abbr)
        if isinstance(key, tuple):
            key = " / ".join(str(part) for part in key)
        groups[str(key)].append(index)
    return dict(groups)


def generate_benchmark(
    root: Path,
    benchmark: str,
    target: int,
    seed: int,
) -> dict:
    spec = BENCHMARKS[benchmark]
    config = Config.fromfile(root / spec["config"])
    dataset_cfgs = config[spec["variable"]]

    datasets: dict[str, dict] = {}
    all_groups: dict[str, tuple[str, list[int]]] = {}
    source_total = 0
    for dataset_cfg in dataset_cfgs:
        dataset_abbr = str(dataset_cfg["abbr"])
        instance = build_dataset_from_cfg(ConfigDict(dataset_cfg))
        rows = instance.test
        groups = _group_indices(rows, dataset_abbr, spec["group"])
        datasets[dataset_abbr] = {
            "source_count": len(rows),
            "strata": {},
        }
        source_total += len(rows)
        for group, indices in groups.items():
            global_group = f"{dataset_abbr}::{group}"
            all_groups[global_group] = (dataset_abbr, indices)

    allocation = allocate_proportionally(
        {name: len(item[1]) for name, item in all_groups.items()},
        min(target, source_total),
    )
    selected_by_dataset: dict[str, list[int]] = defaultdict(list)
    for global_group, (dataset_abbr, indices) in sorted(all_groups.items()):
        sample_count = allocation[global_group]
        rng = random.Random(
            _stable_seed(seed, benchmark, dataset_abbr, global_group)
        )
        selected = sorted(rng.sample(indices, sample_count))
        selected_by_dataset[dataset_abbr].extend(selected)
        group = global_group.split("::", 1)[1]
        datasets[dataset_abbr]["strata"][group] = {
            "source_count": len(indices),
            "sample_count": sample_count,
        }

    for dataset_abbr, details in datasets.items():
        indices = sorted(selected_by_dataset[dataset_abbr])
        details["sample_count"] = len(indices)
        details["indices"] = indices

    sample_total = sum(item["sample_count"] for item in datasets.values())
    return {
        "canonical_name": f"{benchmark}_downsampling",
        "source_total": source_total,
        "sample_total": sample_total,
        "sample_fraction": sample_total / source_total,
        "stratification": spec["stratification"],
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sample-budget", type=int, default=DEFAULT_SAMPLE_BUDGET
    )
    args = parser.parse_args()

    results = {}
    for benchmark in BENCHMARKS:
        results[benchmark] = generate_benchmark(
            args.repo_root,
            benchmark,
            args.sample_budget,
            args.seed,
        )

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "target_runtime_minutes": 30,
        "reference_throughput": {
            "api_concurrency": 512,
            "vllm_replicas": 2,
            "measured_rows_per_minute": 49.5,
            "sample_budget_per_benchmark": args.sample_budget,
            "estimated_minutes_per_generation_benchmark": round(
                args.sample_budget / 49.5, 2
            ),
        },
        "benchmarks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
