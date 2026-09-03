import json
from pathlib import Path

from mmengine.config import Config

from opencompass.datasets.scicode import SciCodeEvaluator
from opencompass.datasets.scicode_official import (
    OFFICIAL_SKIPPED_STEPS,
    OfficialExtractionSciCodeEvaluator,
    OfficialPromptBuilder,
    extract_python_script_official,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data/scicode_official"


def _records():
    with (DATA_ROOT / "problems_test.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def test_official_scicode_snapshot_has_288_scored_steps():
    records = _records()
    steps = [step for problem in records for step in problem["sub_steps"]]
    assert len(records) == 65
    assert len(steps) == 291
    assert sum(
        step["step_number"] not in OFFICIAL_SKIPPED_STEPS for step in steps
    ) == 288


def test_official_prompt_explicitly_includes_previous_model_code():
    problem = _records()[0]
    builder = OfficialPromptBuilder(
        str(DATA_ROOT / "multistep_template.txt"),
        str(DATA_ROOT / "special_steps"),
    )
    previous_codes = [None] * len(problem["sub_steps"])
    previous_codes[0] = "def wrap(r, L):\n    return r % L"
    prompt = builder.build(problem, 1, previous_codes)
    assert previous_codes[0] in prompt
    assert problem["sub_steps"][0]["step_background"] in prompt
    assert problem["sub_steps"][1]["step_background"] in prompt


def test_official_parser_accepts_generic_fence_and_unfenced_code():
    fenced = "```\ndef f():\n    return 1\n```"
    unfenced = "def f():\n    return 1"
    assert extract_python_script_official(fenced).strip().startswith("def f")
    assert extract_python_script_official(unfenced).startswith("def f")

    legacy = SciCodeEvaluator.__new__(SciCodeEvaluator)
    aligned = OfficialExtractionSciCodeEvaluator.__new__(
        OfficialExtractionSciCodeEvaluator
    )
    assert legacy.extract_python_script(fenced) == ""
    assert legacy.extract_python_script(unfenced) == ""
    assert aligned.extract_python_script(fenced).strip().startswith("def f")
    assert aligned.extract_python_script(unfenced).startswith("def f")


def test_official_config_is_isolated_and_uses_official_extraction():
    config = Config.fromfile(
        str(
            REPO_ROOT
            / "opencompass/configs/datasets/scicode/scicode_official_wbg_gen.py"
        )
    )
    dataset = config.SciCode_datasets[0]
    assert dataset["abbr"] == "SciCode_official_with_background_sandboxed"
    assert dataset["with_background"] is True
    evaluator = dataset["eval_cfg"]["evaluator"]
    assert evaluator["type"] is OfficialExtractionSciCodeEvaluator
    assert issubclass(evaluator["type"], SciCodeEvaluator)
    assert "timeout_seconds" not in evaluator
