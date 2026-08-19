import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT / "opencompass/configs/datasets/downsampling/manifest.json"
)


EXPECTED_SOURCE_TOTALS = {
    "mmmlu": 196588,
    "mmlu_prox": 341011,
    "mmlu_redux": 5330,
    "global_piqa": 27091,
    "include": 22639,
}


def test_downsampling_manifest_is_complete_and_reproducible():
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["seed"] == 20260801
    assert document["target_runtime_minutes"] == 30
    assert document["reference_throughput"] == {
        "api_concurrency": 512,
        "vllm_replicas": 2,
        "measured_rows_per_minute": 49.5,
        "sample_budget_per_benchmark": 1400,
        "estimated_minutes_per_generation_benchmark": 28.28,
    }

    assert set(document["benchmarks"]) == set(EXPECTED_SOURCE_TOTALS)
    for name, source_total in EXPECTED_SOURCE_TOTALS.items():
        benchmark = document["benchmarks"][name]
        assert benchmark["canonical_name"] == f"{name}_downsampling"
        assert benchmark["source_total"] == source_total
        assert benchmark["sample_total"] == 1400
        assert math.isclose(
            benchmark["sample_fraction"], 1400 / source_total
        )

        selected = 0
        for dataset in benchmark["datasets"].values():
            indices = dataset["indices"]
            assert indices == sorted(indices)
            assert len(indices) == len(set(indices))
            assert all(0 <= index < dataset["source_count"] for index in indices)
            assert len(indices) == dataset["sample_count"]
            assert sum(
                stratum["sample_count"]
                for stratum in dataset["strata"].values()
            ) == dataset["sample_count"]
            assert all(
                stratum["sample_count"] >= 1
                for stratum in dataset["strata"].values()
            )
            selected += len(indices)
        assert selected == 1400


def test_each_stratum_tracks_the_original_distribution_with_rounding():
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for benchmark in document["benchmarks"].values():
        source_total = benchmark["source_total"]
        sample_total = benchmark["sample_total"]
        for dataset in benchmark["datasets"].values():
            for stratum in dataset["strata"].values():
                ideal = sample_total * stratum["source_count"] / source_total
                assert abs(stratum["sample_count"] - ideal) <= 1
