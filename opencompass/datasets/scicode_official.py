"""Official-prompt SciCode integration for OpenCompass.

This module is intentionally additive.  It does not replace or monkey-patch
the legacy OpenCompass SciCode dataset, inferencer, or evaluator.
"""

from __future__ import annotations

import ast
import concurrent.futures
import json
import os
import os.path as osp
import re
from pathlib import Path

from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.datasets.scicode import SciCodeEvaluator
from opencompass.openicl.icl_inferencer.icl_base_inferencer import (
    prediction_content,
)
from opencompass.openicl.icl_inferencer.icl_chat_inferencer_parallel import (
    ParallelChatInferencer,
)
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import BaseRetriever
from opencompass.openicl.utils.logging import get_logger
from opencompass.registry import ICL_EVALUATORS, ICL_INFERENCERS, LOAD_DATASET
from opencompass.utils import get_data_path

logger = get_logger(__name__)


# These are the three steps skipped by the current official Inspect scorer.
OFFICIAL_SKIPPED_STEPS = frozenset({"13.6", "62.1", "76.3"})


def extract_python_script_official(response: str) -> str:
    """Match SciCode's current official code-block extraction behavior."""
    response = response or ""
    if "```" in response:
        if "```python" in response:
            python_script = response.split("```python", 1)[1].split("```", 1)[0]
        else:
            python_script = response.split("```", 1)[1].split("```", 1)[0]
    else:
        python_script = response
    return re.sub(
        r"^\s*(import .*|from .*\s+import\s+.*)",
        "",
        python_script,
        flags=re.MULTILINE,
    )


@ICL_EVALUATORS.register_module()
class OfficialExtractionSciCodeEvaluator(SciCodeEvaluator):
    """Reuse the local sandboxed scorer with official answer extraction."""

    def extract_python_script(self, response: str):
        return extract_python_script_official(response)


def extract_function_name(function_header: str) -> str:
    match = re.search(r"\bdef\s+(\w+)\s*\(", function_header)
    if match:
        return match.group(1)
    match = re.search(r"\bclass\s+(\w+)\s*\(", function_header)
    if match:
        return match.group(1)
    raise ValueError("Function name or class name not found")


def get_function_from_code(code_string: str, function_name: str) -> str:
    """Vendored equivalent of the official resume-path AST extraction."""
    if code_string is None:
        return code_string
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except (SyntaxError, TypeError, ValueError) as error:
        logger.warning("%s not found while restoring SciCode state: %s", function_name, error)
    return code_string


class OfficialPromptBuilder:
    """Build the per-step prompt used by SciCode's current official runner."""

    def __init__(self, template_path: str, special_code_dir: str):
        template_path = get_data_path(template_path, local_mode=True)
        special_code_dir = get_data_path(special_code_dir, local_mode=True)
        self.template = Path(template_path).read_text(encoding="utf-8")
        self.special_code_dir = Path(special_code_dir)

    def special_code(self, step_number: str) -> str:
        path = self.special_code_dir / f"{step_number}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing official skipped-step code: {path}")
        return path.read_text(encoding="utf-8")

    def build(self, problem: dict, step_index: int, previous_codes: list[str]) -> str:
        sub_steps = problem["sub_steps"]
        output_lines = []
        for previous_index in range(step_index):
            previous_step = sub_steps[previous_index]
            previous_code = previous_codes[previous_index]
            if previous_code is None:
                raise RuntimeError(
                    f"Problem {problem['problem_id']} step {step_index + 1} "
                    f"was requested before step {previous_index + 1}"
                )
            output_lines.append(
                previous_step["step_description_prompt"]
                + "\n"
                + previous_step["step_background"]
            )
            output_lines.append(previous_code)
            output_lines.append("------")

        # The official implementation removes the final separator this way.
        problem_steps_str = "\n\n".join(output_lines[:-1])

        current_step = sub_steps[step_index]
        next_step_str = "\n\n".join(
            [
                current_step["step_description_prompt"]
                + "\n"
                + current_step["step_background"],
                current_step["function_header"]
                + "\n\n"
                + current_step["return_line"],
            ]
        )
        return self.template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=problem["required_dependencies"],
        )


@LOAD_DATASET.register_module()
class OfficialSciCodeDataset(BaseDataset):
    """Load the official structured SciCode JSONL test split."""

    @staticmethod
    def load(path: str, with_background: bool = True, **kwargs):
        if not with_background:
            raise ValueError(
                "OfficialSciCodeDataset is the isolated with-background "
                "configuration; use the legacy dataset for no-background runs"
            )
        path = get_data_path(path, local_mode=True)
        records = []
        with open(osp.join(path, "problems_test.jsonl"), "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
        return Dataset.from_list(records)


@ICL_INFERENCERS.register_module()
class OfficialSciCodeWithBackgroundInferencer(ParallelChatInferencer):
    """Run problems in parallel and official sub-steps sequentially."""

    def __init__(
        self,
        model,
        output_json_filepath: str | None = "./icl_inference_output",
        output_json_filename: str | None = "predictions",
        save_every: int | None = 1,
        max_out_len: int | None = 512,
        max_infer_workers: int | None = None,
        official_prompt_template: str = "./data/scicode_official/multistep_template.txt",
        official_special_code_dir: str = "./data/scicode_official/special_steps",
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            output_json_filepath=output_json_filepath,
            output_json_filename=output_json_filename,
            save_every=save_every,
            infer_mode="every",
            max_out_len=max_out_len,
            max_infer_workers=max_infer_workers,
            **kwargs,
        )
        self.prompt_builder = OfficialPromptBuilder(
            official_prompt_template, official_special_code_dir
        )

    def _infer_problem(self, problem: dict, index: int) -> dict:
        local_handler = self.HandlerType()
        sub_steps = problem["sub_steps"]
        previous_codes = [None] * len(sub_steps)

        for step_index, step in enumerate(sub_steps):
            step_number = step["step_number"]
            if step_number in OFFICIAL_SKIPPED_STEPS:
                previous_codes[step_index] = self.prompt_builder.special_code(step_number)
                continue

            prompt = self.prompt_builder.build(problem, step_index, previous_codes)
            try:
                output = self.model.generate_from_template(
                    [[{"role": "user", "content": prompt}]],
                    max_out_len=self.max_out_len,
                )[0]
            except Exception:
                # Match the official runner's generation-failure fallback while
                # preserving a useful server log.
                logger.exception(
                    "Generation failed for SciCode problem %s step %s",
                    problem["problem_id"],
                    step_number,
                )
                output = "Blah blah\n```python\nprint('Hello, World!')\n```\n"

            content = prediction_content(output)
            previous_codes[step_index] = extract_python_script_official(content)
            local_handler.save_multiround_results(
                origin_prompt=prompt,
                prediction=output,
                idx=index,
                gold=None,
            )

        return local_handler.results_dict

    def _inference(
        self,
        retriever: BaseRetriever,
        ice_template: PromptTemplate | None = None,
        prompt_template: PromptTemplate | None = None,
        output_json_filepath: str | None = None,
        output_json_filename: str | None = None,
    ) -> dict:
        del ice_template, prompt_template
        output_handler = self.HandlerType()
        output_json_filepath = output_json_filepath or self.output_json_filepath
        output_json_filename = output_json_filename or self.output_json_filename

        total_samples = len(retriever.test_ds)
        tmp_jsonl_filename = Path("tmp_" + output_json_filename).with_suffix(".jsonl").name
        tmp_jsonl_filepath = Path(output_json_filepath) / tmp_jsonl_filename
        restored = output_handler.restore_from_jsonl(
            output_json_filepath, tmp_jsonl_filename
        )
        todo = [index for index in range(total_samples) if str(index) not in restored]

        if self.progress_tracker is not None:
            self.progress_tracker.set_total(total_samples)
            self.progress_tracker.set_completed(total_samples - len(todo))

        logger.info(
            "Starting official-prompt SciCode inference for %d remaining problems",
            len(todo),
        )
        completed = total_samples - len(todo)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._resolve_max_workers()
        ) as executor:
            futures = {
                executor.submit(
                    self._infer_problem,
                    retriever.test_ds[index],
                    index,
                ): index
                for index in todo
            }
            for future in concurrent.futures.as_completed(futures):
                result_dict = future.result()
                output_handler.results_dict.update(result_dict)
                completed += 1
                self._progress_update(1)
                if (
                    self.save_every is not None
                    and completed % self.save_every == 0
                    and self.is_main_process
                ):
                    output_handler.write_to_jsonl(
                        output_json_filepath, tmp_jsonl_filename
                    )

        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath, output_json_filename)
            if tmp_jsonl_filepath.exists():
                tmp_jsonl_filepath.unlink()
        return output_handler.results_dict
