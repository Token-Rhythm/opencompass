#!/usr/bin/env python3
"""Send one scored OpenCompass benchmark to Agent Eval Hub.

The adapter supports review-gated ``/api/submissions`` and administrator-only
canonical ``/api/ingest``. It never accepts token values on the command line:
credentials are read from named environment variables so they cannot leak
through process listings or launcher output.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ADAPTER_VERSION = 'opencompass-agent-eval-adapter@2'
TOKEN_ENV_DEFAULT = 'AGENT_EVAL_SUBMISSION_PAT'
INGEST_TOKEN_ENV_DEFAULT = 'INGEST_ADMIN_TOKEN'

_MISSING = object()
_SECRET_CONFIG_KEY = re.compile(
    r'(?:^|_)(?:api_?key|authorization|cookie|password|secret|token)(?:_|$)',
    re.IGNORECASE,
)
_STRUCTURED_SAMPLING_KEYS = {
    'temperature',
    'top_p',
    'top_k',
    'min_p',
    'seed',
    'max_tokens',
    'max_completion_tokens',
    'max_output_tokens',
    'thinking_enabled',
    'enable_thinking',
    'reasoning_effort',
    'reasoning_budget',
    'thinking_budget',
    'thinking_budget_tokens',
}
_OPENAI_SAMPLING_KEYS = {
    'best_of',
    'frequency_penalty',
    'logprobs',
    'n',
    'presence_penalty',
    'repetition_penalty',
    'response_format',
    'stop',
    'top_logprobs',
}

# Summary names are OpenCompass identities.  Benchmark aliases and metric
# aliases are exact Agent Eval catalog candidates; resolution still has to be
# unique, so this table can never silently pick between catalog versions.
BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    'mmlu_pro': {
        'summary_datasets': ['mmlu_pro'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['MMLU-Pro', 'mmlu-pro'],
        'benchmark_version': '2024-5shot-cot',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
    'ceval': {
        'summary_datasets': ['ceval'],
        'summary_metrics': ['naive_average', 'accuracy'],
        'benchmark_aliases': ['C-Eval', 'ceval'],
        'benchmark_version': '2023-evalscope-5shot',
        'split': 'val',
        'metric_aliases': ['accuracy'],
    },
    'supergpqa': {
        'summary_datasets': ['supergpqa'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['SuperGPQA', 'supergpqa'],
        'benchmark_version': '2025-opencompass-zero-shot',
        'split': 'train',
        'metric_aliases': ['accuracy'],
    },
    'ifeval': {
        'summary_datasets': ['IFEval'],
        'summary_metrics': ['Prompt-level-strict-accuracy'],
        'benchmark_aliases': ['IFEval', 'if-eval'],
        'benchmark_version': '2023-966cd895-zero-shot',
        'split': 'train',
        'metric_aliases': ['prompt_level_strict_accuracy',
                           'Prompt-level-strict-accuracy'],
    },
    'mmmlu': {
        'summary_datasets': ['mmmlu'],
        'summary_metrics': ['accuracy', 'naive_average'],
        'benchmark_aliases': ['MMMLU', 'OpenAI MMMLU', 'mmmlu'],
        'benchmark_version': '2024-simple-evals-zero-shot',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
    'gpqa_diamond': {
        'summary_datasets': ['GPQA_diamond', 'gpqa_diamond'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['GPQA-Diamond', 'GPQA Diamond', 'gpqa-diamond'],
        'benchmark_version': 'original-2023-simple-evals-4x',
        'split': 'diamond',
        'metric_aliases': ['accuracy'],
    },
    'ifbench': {
        'summary_datasets': ['IFBench'],
        'summary_metrics': ['score'],
        'benchmark_aliases': ['IFBench', 'if-bench'],
        'benchmark_version': '2025-2e8a48de-zero-shot',
        'split': 'train',
        'metric_aliases': ['prompt_level_loose_accuracy'],
    },
    'longbench_v2': {
        'summary_datasets': ['LongBenchv2'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['LongBench v2', 'LongBenchv2', 'longbench-v2'],
        'benchmark_version': '2.0-opencompass-zero-shot',
        'split': 'train',
        'metric_aliases': ['accuracy'],
    },
    'aime_2024': {
        'summary_datasets': ['aime2024'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['AIME', 'aime'],
        'benchmark_version': '2024-opencompass',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
    'aime_2025': {
        'summary_datasets': ['aime2025'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['AIME', 'aime'],
        'benchmark_version': '2025-opencompass',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
    'aime_2026': {
        'summary_datasets': ['aime2026'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['AIME', 'aime'],
        'benchmark_version': '2026-matharena-4x',
        'split': 'train',
        'metric_aliases': ['accuracy'],
    },
    'hmmt_feb_2026': {
        'summary_datasets': ['hmmt2026'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['HMMT', 'hmmt'],
        'benchmark_version': '2026-02-matharena-4x',
        'split': 'train',
        'metric_aliases': ['accuracy'],
    },
    'livecodebench': {
        'summary_datasets': ['livecodebench_v6_codegen'],
        'summary_metrics': ['pass@1'],
        'benchmark_aliases': ['LiveCodeBench v6', 'LiveCodeBench'],
        'benchmark_version': 'v6',
        'split': 'test',
        'metric_aliases': ['pass@1', 'pass_at_1'],
    },
    'humaneval': {
        'summary_datasets': ['openai_humaneval'],
        'summary_metrics': ['humaneval_pass@1'],
        'benchmark_aliases': ['HumanEval', 'OpenAI HumanEval', 'humaneval'],
        'benchmark_version': 'openai-sample-evals-zero-shot',
        'split': 'test',
        'metric_aliases': ['pass@1', 'pass_at_1', 'humaneval_pass@1'],
    },
    'mmlu_redux': {
        'summary_datasets': ['mmlu_redux_full_chat', 'mmlu_redux'],
        'summary_metrics': ['accuracy'],
        # Keep 2.0 aliases exact: production also has the distinct original
        # MMLU-Redux family, so the generic ``MMLU-Redux`` name is ambiguous.
        'benchmark_aliases': ['MMLU-Redux-2.0', 'MMLU-Redux 2.0',
                              'mmlu-redux-2-0'],
        'benchmark_version': '2.0-ok-opencompass-chat',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
    'mmlu_prox': {
        'summary_datasets': ['mmlu_prox'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['MMLU-ProX', 'mmlu-prox'],
        'benchmark_version': '2025.05-full-5shot-cot',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
    'global_piqa': {
        'summary_datasets': ['global_piqa_generation', 'global_piqa'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['Global PIQA', 'global-piqa'],
        'benchmark_version': '2026-opencompass-zero-shot',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
    'hmmt_feb_2025': {
        'summary_datasets': ['hmmt_feb_2025_full_chat'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['HMMT', 'hmmt'],
        'benchmark_version': '2025-02-matharena-4x',
        'split': 'train',
        'metric_aliases': ['accuracy'],
    },
    'hmmt_nov_2025': {
        'summary_datasets': ['hmmt_nov_2025_full_chat'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['HMMT', 'hmmt'],
        'benchmark_version': '2025-11-matharena-4x',
        'split': 'train',
        'metric_aliases': ['accuracy'],
    },
    'include': {
        'summary_datasets': ['include_base_44'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['INCLUDE base-44', 'INCLUDE', 'include-base-44'],
        'benchmark_version': '2024-d2e1f601-zero-shot',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
    'multichallenge': {
        'summary_datasets': ['multichallenge'],
        'summary_metrics': ['overall_score'],
        'benchmark_aliases': ['MultiChallenge', 'multi-challenge',
                              'multichallenge'],
        'benchmark_version': '2025-5ccefcca-gpt4o-judge',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
    'aa_lcr': {
        'summary_datasets': ['aa_lcr'],
        'summary_metrics': ['accuracy'],
        'benchmark_aliases': ['AA-LCR', 'aa-lcr'],
        'benchmark_version': '2025-bdae010',
        'split': 'test',
        'metric_aliases': ['accuracy'],
    },
}


def _downsampling_spec(source, summary_dataset, aliases, version):
    spec = dict(BENCHMARK_SPECS[source])
    spec.update(
        summary_datasets=[summary_dataset],
        benchmark_aliases=aliases,
        benchmark_version=version,
    )
    return spec


BENCHMARK_SPECS.update({
    'mmmlu_downsampling': _downsampling_spec(
        'mmmlu', 'mmmlu_downsampling',
        ['MMMLU Downsampling', 'mmmlu-downsampling'],
        '2024-simple-evals-zero-shot-downsampling-1400'),
    'mmlu_prox_downsampling': _downsampling_spec(
        'mmlu_prox', 'mmlu_prox_downsampling',
        ['MMLU-ProX Downsampling', 'mmlu-prox-downsampling'],
        '2025.05-full-5shot-cot-downsampling-1400'),
    'mmlu_redux_downsampling': _downsampling_spec(
        'mmlu_redux', 'mmlu_redux_downsampling',
        ['MMLU-Redux-2.0 Downsampling', 'mmlu-redux-2-0-downsampling'],
        '2.0-ok-opencompass-chat-downsampling-1400'),
    'global_piqa_downsampling': _downsampling_spec(
        'global_piqa', 'global_piqa_generation_downsampling',
        ['Global PIQA Downsampling', 'global-piqa-downsampling'],
        '2026-opencompass-zero-shot-downsampling-1400'),
    'include_downsampling': _downsampling_spec(
        'include', 'include_downsampling',
        ['INCLUDE base-44 Downsampling', 'include-base-44-downsampling'],
        '2024-d2e1f601-zero-shot-downsampling-1400'),
})


class AdapterError(RuntimeError):
    """A safe, user-actionable adapter failure."""


def normalized(value: Any) -> str:
    return str(value or '').strip().casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.',
                                     dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      sort_keys=True)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_http_url(value: str, label: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as error:
        raise AdapterError(f'{label} is not a valid URL') from error
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise AdapterError(f'{label} must be an absolute HTTP/HTTPS URL')
    if parsed.username or parsed.password:
        raise AdapterError(f'{label} must not contain credentials')
    if label == 'platform URL' and (parsed.query or parsed.fragment
                                    or parsed.path not in {'', '/'}):
        raise AdapterError(
            'platform URL must not contain a path, query, or fragment')
    return value.rstrip('/') if label == 'platform URL' else value


def load_spec(benchmark: str, mapping_file: Path | None) -> dict[str, Any]:
    if benchmark not in BENCHMARK_SPECS:
        names = ', '.join(BENCHMARK_SPECS)
        raise AdapterError(f'unsupported benchmark {benchmark!r}; expected: {names}')
    spec = dict(BENCHMARK_SPECS[benchmark])
    spec.setdefault('split', 'test')
    if mapping_file:
        try:
            document = json.loads(mapping_file.read_text(encoding='utf-8'))
            override = document.get(benchmark, {})
        except (OSError, json.JSONDecodeError) as error:
            raise AdapterError(f'cannot read mapping file {mapping_file}') from error
        if not isinstance(override, dict):
            raise AdapterError(f'mapping for {benchmark} must be an object')
        spec.update(override)
    return spec


def latest_summary(run_dir: Path) -> Path:
    candidates = list((run_dir / 'summary').glob('summary_*.csv'))
    if not candidates:
        raise AdapterError(f'no summary_*.csv found under {run_dir / "summary"}')
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns,
                                             path.name))


def read_primary_score(summary: Path, spec: dict[str, Any], model: str) -> dict[str, Any]:
    with summary.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    fixed = {'dataset', 'version', 'metric', 'mode'}
    model_columns = [name for name in fieldnames if name not in fixed]
    exact_model = [name for name in model_columns if normalized(name) == normalized(model)]
    if len(exact_model) == 1:
        model_column = exact_model[0]
    elif len(model_columns) == 1:
        model_column = model_columns[0]
    else:
        raise AdapterError(
            f'cannot select model column for {model!r}; columns={model_columns}')

    datasets = {normalized(value) for value in spec['summary_datasets']}
    metrics = [normalized(value) for value in spec['summary_metrics']]
    matches = [row for row in rows
               if normalized(row.get('dataset')) in datasets
               and normalized(row.get('metric')) in set(metrics)]
    matches.sort(key=lambda row: metrics.index(normalized(row.get('metric'))))
    for row in matches:
        try:
            score = float(str(row.get(model_column, '')).strip())
        except ValueError:
            continue
        if math.isfinite(score):
            return {
                'dataset': str(row.get('dataset', '')).strip(),
                'version': str(row.get('version', '')).strip(),
                'metric': str(row.get('metric', '')).strip(),
                'mode': str(row.get('mode', '')).strip(),
                'model_column': model_column,
                'score_percent': score,
            }
    raise AdapterError(
        f'no numeric primary score in {summary}; expected datasets='
        f'{spec["summary_datasets"]}, metrics={spec["summary_metrics"]}')


def count_predictions(run_dir: Path) -> int | None:
    root = run_dir / 'predictions'
    if not root.is_dir():
        return None
    count = 0
    found = False
    for path in root.glob('*/*.json'):
        if path.name.endswith(('_metrics.json', '_audit.json')):
            continue
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, (list, dict)):
            count += len(value)
            found = True
    return count if found else None


def _static_config_value(node: ast.AST) -> Any:
    """Evaluate only literal containers used by MMEngine's dumped configs."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        values = [_static_config_value(item) for item in node.elts]
        return _MISSING if any(item is _MISSING for item in values) else values
    if isinstance(node, ast.Tuple):
        values = [_static_config_value(item) for item in node.elts]
        return (_MISSING if any(item is _MISSING for item in values)
                else tuple(values))
    if isinstance(node, ast.Dict):
        result = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                return _MISSING
            key = _static_config_value(key_node)
            value = _static_config_value(value_node)
            if key is _MISSING or value is _MISSING:
                return _MISSING
            result[key] = value
        return result
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'dict' and not node.args):
        result = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                return _MISSING
            value = _static_config_value(keyword.value)
            if value is _MISSING:
                return _MISSING
            result[keyword.arg] = value
        return result
    if (isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))):
        value = _static_config_value(node.operand)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _MISSING
        return value if isinstance(node.op, ast.UAdd) else -value
    return _MISSING


def _config_assignments(config: Path, names: set[str]) -> dict[str, Any]:
    try:
        tree = ast.parse(config.read_text(encoding='utf-8'), filename=str(config))
    except (OSError, SyntaxError, UnicodeError):
        return {}
    values = {}
    for statement in tree.body:
        target_name = None
        if (isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            target_name = statement.targets[0].id
        elif (isinstance(statement, ast.AnnAssign)
              and isinstance(statement.target, ast.Name)
              and statement.value is not None):
            target_name = statement.target.id
        if target_name not in names:
            continue
        value = _static_config_value(statement.value)
        if value is not _MISSING:
            values[target_name] = value
    return values


def _model_identity_rank(config: dict[str, Any], model: str) -> int:
    reference = normalized(model)
    if normalized(config.get('path')) == reference:
        return 0
    if normalized(config.get('summarizer_abbr')) == reference:
        return 1
    abbr = normalized(config.get('abbr'))
    if abbr == reference or abbr.removesuffix('-chat').removesuffix(
            '-completions') == reference:
        return 2
    return 3


def _select_model_config(models: Any, model: str,
                         benchmark: str) -> dict[str, Any] | None:
    candidates = [item for item in models or [] if isinstance(item, dict)]
    if not candidates:
        return None
    best_identity = min(_model_identity_rank(item, model)
                        for item in candidates)
    if best_identity < 3:
        candidates = [item for item in candidates
                      if _model_identity_rank(item, model) == best_identity]
    elif len(candidates) != 1:
        return None
    preferred_endpoint = (
        'completions' if benchmark in {'include', 'include_downsampling'}
        else 'chat'
    )
    routed = [item for item in candidates
              if normalized(item.get('generation_endpoint'))
              == preferred_endpoint]
    if routed:
        candidates = routed
    return candidates[0] if len(candidates) == 1 else None


def _dataset_max_out_len(datasets: Any) -> int | None:
    values = []
    for dataset in datasets or []:
        if not isinstance(dataset, dict):
            continue
        inferencer = dataset.get('infer_cfg', {}).get('inferencer', {})
        value = inferencer.get('max_out_len') if isinstance(
            inferencer, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            values.append(value)
    return max(values, default=None)


def _safe_sampling_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return _MISSING
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            safe = _safe_sampling_value(item)
            if safe is not _MISSING:
                result.append(safe)
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or _SECRET_CONFIG_KEY.search(key):
                continue
            safe = _safe_sampling_value(item)
            if safe is not _MISSING:
                result[key] = safe
        return result
    return _MISSING


def _option(model_config: dict[str, Any], extra_body: dict[str, Any],
            openai_kwargs: dict[str, Any], *names: str) -> Any:
    for container in (openai_kwargs, extra_body, model_config):
        for name in names:
            if name in container:
                return container[name]
    return _MISSING


def generation_config(config: Path | None, model: str,
                      benchmark: str) -> dict[str, Any]:
    """Extract the scored model's effective, non-secret generation settings."""
    # INCLUDE is scored from prompt log-probabilities. The API model object
    # still carries generation defaults, but none of them participate in its
    # continuation-likelihood score and recording them would be misleading.
    if config is None or benchmark in {'include', 'include_downsampling'}:
        return {}
    assignments = _config_assignments(config, {'models', 'datasets'})
    model_config = _select_model_config(assignments.get('models'), model,
                                        benchmark)
    if model_config is None:
        return {}
    openai_kwargs = model_config.get('openai_extra_kwargs')
    openai_kwargs = openai_kwargs if isinstance(openai_kwargs, dict) else {}
    extra_body = model_config.get('extra_body')
    extra_body = extra_body if isinstance(extra_body, dict) else {}
    if isinstance(openai_kwargs.get('extra_body'), dict):
        extra_body = openai_kwargs['extra_body']

    result: dict[str, Any] = {}
    validators = {
        'temperature': lambda value: (isinstance(value, (int, float))
                                      and not isinstance(value, bool)
                                      and value >= 0),
        'top_p': lambda value: (isinstance(value, (int, float))
                                and not isinstance(value, bool)
                                and 0 <= value <= 1),
        'top_k': lambda value: (isinstance(value, int)
                                and not isinstance(value, bool)
                                and value >= 0),
        'min_p': lambda value: (isinstance(value, (int, float))
                                and not isinstance(value, bool)
                                and 0 <= value <= 1),
        'seed': lambda value: (isinstance(value, int)
                               and not isinstance(value, bool)),
    }
    for field, validator in validators.items():
        value = _option(model_config, extra_body, openai_kwargs, field)
        if value is not _MISSING and validator(value):
            result[field] = value

    max_output_tokens = _option(
        model_config, extra_body, openai_kwargs,
        'max_completion_tokens', 'max_tokens', 'max_output_tokens')
    if max_output_tokens is _MISSING:
        max_output_tokens = _dataset_max_out_len(assignments.get('datasets'))
    if max_output_tokens is None:
        max_output_tokens = model_config.get('max_out_len')
    if (isinstance(max_output_tokens, int)
            and not isinstance(max_output_tokens, bool)
            and max_output_tokens > 0):
        result['max_output_tokens'] = max_output_tokens

    chat_template_kwargs = extra_body.get('chat_template_kwargs')
    chat_template_kwargs = (chat_template_kwargs
                            if isinstance(chat_template_kwargs, dict) else {})
    thinking_enabled = _option(
        model_config, extra_body, openai_kwargs,
        'thinking_enabled', 'enable_thinking')
    if thinking_enabled is _MISSING:
        thinking_enabled = chat_template_kwargs.get('enable_thinking',
                                                     _MISSING)
    if isinstance(thinking_enabled, bool):
        result['thinking_enabled'] = thinking_enabled

    reasoning_effort = _option(model_config, extra_body, openai_kwargs,
                               'reasoning_effort')
    if (isinstance(reasoning_effort, str)
            and 0 < len(reasoning_effort.strip()) <= 64):
        result['reasoning_effort'] = reasoning_effort.strip()
    thinking_budget = _option(
        model_config, extra_body, openai_kwargs,
        'thinking_budget_tokens', 'thinking_budget', 'reasoning_budget')
    if (isinstance(thinking_budget, int)
            and not isinstance(thinking_budget, bool)
            and thinking_budget >= 0):
        result['thinking_budget_tokens'] = thinking_budget

    sampling_config = _safe_sampling_value(extra_body)
    sampling_config = sampling_config if isinstance(sampling_config, dict) else {}
    for field in _STRUCTURED_SAMPLING_KEYS:
        sampling_config.pop(field, None)
    for key in _OPENAI_SAMPLING_KEYS:
        if key in openai_kwargs:
            safe = _safe_sampling_value(openai_kwargs[key])
            if safe is not _MISSING:
                sampling_config[key] = safe
    if sampling_config:
        result['sampling_config'] = sampling_config
    return result


def git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(['git', *arguments], check=True,
                                text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def run_metadata(run_dir: Path, summary: Path) -> dict[str, Any]:
    config_candidates = [path for path in (run_dir / 'configs').glob('*.py')
                         if path.stat().st_mtime_ns <= summary.stat().st_mtime_ns]
    config = max(config_candidates,
                 key=lambda path: (path.stat().st_mtime_ns, path.name),
                 default=None)
    started_source = min(config_candidates or [summary],
                         key=lambda path: path.stat().st_mtime_ns)
    started_at = datetime.fromtimestamp(started_source.stat().st_mtime,
                                        timezone.utc).isoformat()
    completed_at = datetime.fromtimestamp(summary.stat().st_mtime,
                                          timezone.utc).isoformat()
    config_hash = f'sha256:{sha256_file(config)}' if config else None
    commit = git_value('rev-parse', 'HEAD')
    dirty = bool(git_value('status', '--porcelain'))
    return {
        'started_at': started_at,
        'completed_at': completed_at,
        'total_time_ms': max(0, int((summary.stat().st_mtime
                                     - started_source.stat().st_mtime) * 1000)),
        'config_hash': config_hash,
        'config_file': str(config) if config else None,
        'config_path': config,
        'code_commit_sha': commit,
        'worktree_dirty': dirty,
    }


def runner_version(commit: str | None) -> str:
    package_init = Path(__file__).resolve().parents[1] / 'opencompass/__init__.py'
    version = 'unknown'
    try:
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)",
                          package_init.read_text(encoding='utf-8'))
        if match:
            version = match.group(1)
    except OSError:
        pass
    revision = f'+{commit[:12]}' if commit else ''
    return f'opencompass@{version}{revision}'


def catalog_data(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict) or envelope.get('source') != 'd1':
        raise AdapterError('submission catalog is not backed by canonical D1 data')
    data = envelope.get('data')
    if not isinstance(data, dict):
        raise AdapterError('submission catalog response has no data object')
    for key in ('systems', 'benchmarks', 'benchmarkVersions', 'metrics'):
        if not isinstance(data.get(key), list):
            raise AdapterError(f'submission catalog has no {key} array')
    return data


def unique_alias(items: list[dict[str, Any]], aliases: list[str],
                 fields: tuple[str, ...], label: str) -> dict[str, Any]:
    references = {normalized(alias) for alias in aliases}
    matches = {str(item.get('id')): item for item in items
               if any(normalized(item.get(field)) in references
                      for field in fields)}
    if len(matches) != 1:
        candidates = [
            {field: item.get(field) for field in ('id', *fields)}
            for item in items[:20]
        ]
        qualifier = 'no exact match' if not matches else 'ambiguous match'
        raise AdapterError(f'{label}: {qualifier} for {aliases!r}; '
                           f'candidates={candidates!r}')
    return next(iter(matches.values()))


def resolve_catalog(envelope: Any, spec: dict[str, Any], system_ref: str,
                    scope: str, summary_version: str) -> dict[str, Any]:
    data = catalog_data(envelope)
    systems = [item for item in data['systems'] if item.get('scope') == scope]
    system = unique_alias(systems, [system_ref],
                          ('id', 'name', 'displayName'),
                          f'system in scope {scope}')
    benchmark = unique_alias(data['benchmarks'], spec['benchmark_aliases'],
                             ('id', 'slug', 'name', 'displayName'),
                             'benchmark')
    versions = [item for item in data['benchmarkVersions']
                if item.get('benchmarkId') == benchmark.get('id')
                and normalized(item.get('split')) == normalized(spec['split'])]
    version_ref = spec.get('benchmark_version')
    if version_ref:
        version = unique_alias(versions, [str(version_ref)], ('id', 'version'),
                               'benchmark version')
    else:
        summary_matches = [item for item in versions
                           if normalized(item.get('version')) == normalized(summary_version)
                           and summary_version not in {'', '-'}]
        if len(summary_matches) == 1:
            version = summary_matches[0]
        elif len(versions) == 1:
            version = versions[0]
        else:
            choices = [{'id': item.get('id'), 'version': item.get('version'),
                        'split': item.get('split')} for item in versions]
            raise AdapterError(
                'benchmark version is not unique; set benchmark_version in '
                f'--agent-eval-mapping-file. candidates={choices!r}')

    metric_aliases = spec['metric_aliases']
    metric_matches = []
    references = {normalized(value) for value in metric_aliases}
    for item in data['metrics']:
        if any(normalized(item.get(field)) in references
               for field in ('id', 'key', 'displayName')):
            metric_matches.append(item)
    percent = [item for item in metric_matches
               if normalized(item.get('unit')) == 'percent']
    ratio = [item for item in metric_matches
             if normalized(item.get('unit')) == 'ratio']
    candidates = percent if len(percent) == 1 else ratio if len(ratio) == 1 else []
    if len(candidates) != 1:
        choices = [{'id': item.get('id'), 'key': item.get('key'),
                    'unit': item.get('unit')} for item in metric_matches]
        raise AdapterError(f'metric is not unique for aliases={metric_aliases!r}; '
                           f'candidates={choices!r}')
    return {'system': system, 'benchmark': benchmark, 'version': version,
            'metric': candidates[0]}


def safe_slug(value: Any, maximum: int = 30) -> str:
    slug = re.sub(r'[^A-Za-z0-9._:-]+', '-', str(value)).strip('-._:')
    return (slug or 'unknown')[:maximum]


def idempotency_key(benchmark: str, resolved: dict[str, Any], run_id: str) -> str:
    parts = ['opencompass', benchmark, resolved['system']['id'],
             resolved['version']['id'], run_id, resolved['metric']['id']]
    key = '.'.join(safe_slug(part) for part in parts)
    if len(key) <= 128:
        return key
    suffix = hashlib.sha256(key.encode()).hexdigest()[:20]
    return f'{key[:107]}.{suffix}'


def render_evidence_url(template: str, benchmark: str, run_id: str,
                        summary: Path, summary_hash: str) -> str:
    try:
        value = template.format(benchmark=benchmark, run_id=run_id,
                                summary_file=summary.name,
                                summary_sha256=summary_hash)
    except KeyError as error:
        raise AdapterError(f'unknown evidence URL placeholder {error}') from error
    return validate_http_url(value, 'evidence URL')


def publish_evidence(summary: Path, evidence_root: Path,
                     benchmark: str, run_id: str) -> Path:
    safe_benchmark = safe_slug(benchmark, maximum=80)
    safe_run_id = safe_slug(run_id, maximum=120)
    if safe_benchmark != benchmark or safe_run_id != run_id:
        raise AdapterError('benchmark or run id is unsafe for evidence publishing')
    destination = evidence_root / benchmark / run_id / summary.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f'.{destination.name}.{os.getpid()}.tmp')
    try:
        with summary.open('rb') as source, temporary.open('xb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if sha256_file(temporary) != sha256_file(summary):
            raise AdapterError('published evidence SHA256 mismatch')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_submission(*, benchmark: str, model: str, run_dir: Path,
                     summary: Path, score: dict[str, Any],
                     spec: dict[str, Any], resolved: dict[str, Any],
                     evidence_url: str, provenance: str, publisher: str,
                     source: str) -> dict[str, Any]:
    summary_hash = sha256_file(summary)
    metadata = run_metadata(run_dir, summary)
    sample_count = count_predictions(run_dir)
    metric = resolved['metric']
    unit = str(metric.get('unit'))
    value = score['score_percent'] / 100 if normalized(unit) == 'ratio' \
        else score['score_percent']
    run_id = run_dir.name
    scorer_key = metric.get('scorerKey') or score['metric']
    system = resolved['system']
    benchmark_row = resolved['benchmark']
    version = resolved['version']
    canonical_result = {
        'runner_version': runner_version(metadata['code_commit_sha']),
        'started_at': metadata['started_at'],
        'completed_at': metadata['completed_at'],
        'total_time_ms': metadata['total_time_ms'],
        'dataset_snapshot': (f'opencompass:{score["version"]}'
                             if score['version'] not in {'', '-'} else
                             f'catalog:{version.get("version")}'),
        'scorer_key': scorer_key,
        'result_level': 'aggregate',
        'is_primary': True,
        'metadata': {
            'adapter_version': ADAPTER_VERSION,
            'producer': source,
            'opencompass_benchmark': benchmark,
            'opencompass_dataset': score['dataset'],
            'opencompass_metric': score['metric'],
            'opencompass_mode': score['mode'],
            'opencompass_model_column': score['model_column'],
            'run_id': run_id,
            'summary_file': str(summary),
            'config_file': metadata['config_file'],
            'worktree_dirty': metadata['worktree_dirty'],
        },
    }
    canonical_result.update(generation_config(
        metadata['config_path'], model, benchmark))
    if metadata['config_hash']:
        canonical_result['config_hash'] = metadata['config_hash']
    if metadata['code_commit_sha']:
        canonical_result['code_commit_sha'] = metadata['code_commit_sha']
    title_name = benchmark_row.get('displayName') or benchmark_row.get('name')
    submission = {
        'submission_kind': 'result',
        'provenance': provenance,
        'scope': system.get('scope'),
        'model_name': system.get('modelName') or system.get('displayName')
        or system.get('name') or model,
        'benchmark_name': title_name,
        'benchmark_version': f'{version.get("version")} / {version.get("split")}',
        'metric_key': metric.get('key'),
        'metric_value': value,
        'metric_unit': unit,
        'completed_at': metadata['completed_at'],
        'source_url': evidence_url,
        'evidence_title': f'OpenCompass {title_name} run {run_id}',
        'publisher': publisher,
        'notes': (f'OpenCompass primary aggregate; summary sha256:{summary_hash}; '
                  f'local run {run_dir}.'),
        'canonical_mapping': {
            'system_id': system.get('id'),
            'benchmark_version_id': version.get('id'),
            'metric_id': metric.get('id'),
            'scorer_key': scorer_key,
        },
        'canonical_evidence': {
            'is_open': provenance == 'internal_public',
            'content_hash': f'sha256:{summary_hash}',
            'metadata': {
                'adapter_version': ADAPTER_VERSION,
                'opencompass_summary_file': summary.name,
            },
        },
        'canonical_result': canonical_result,
    }
    if sample_count is not None:
        submission['sample_count'] = sample_count
    model_version = system.get('modelVersion')
    if model_version and model_version != 'unknown':
        submission['model_version'] = model_version
    harness = system.get('harnessName')
    if harness:
        submission['harness_name'] = harness
    return {
        'idempotency_key': idempotency_key(benchmark, resolved, run_id),
        'dry_run': True,
        'submission': submission,
    }


def build_ingest_payload(submission_document: dict[str, Any],
                         resolved: dict[str, Any], source: str) -> dict[str, Any]:
    """Translate the reviewed-submission shape into canonical /api/ingest."""
    submission = submission_document['submission']
    result_input = submission['canonical_result']
    evidence_input = submission['canonical_evidence']
    mapping = submission['canonical_mapping']
    sample_count = submission.get('sample_count')
    planned_count = resolved['version'].get('sampleCount')
    if not isinstance(planned_count, int) or isinstance(planned_count, bool):
        planned_count = sample_count
    if (isinstance(sample_count, int) and not isinstance(sample_count, bool)
            and isinstance(planned_count, int)
            and sample_count > planned_count):
        planned_count = sample_count

    evidence = {
        'local_id': 'evidence',
        'provenance': submission['provenance'],
        'title': submission['evidence_title'],
        'publisher': submission['publisher'],
        'url': submission['source_url'],
        'retrieved_at': submission['completed_at'],
        'is_open': evidence_input.get('is_open', False),
        'content_hash': evidence_input.get('content_hash'),
        'notes': submission.get('notes'),
        'metadata': evidence_input.get('metadata', {}),
    }
    run_fields = {
        'runner_version', 'code_commit_sha', 'dataset_snapshot', 'config_hash',
        'temperature', 'top_p', 'top_k', 'min_p', 'max_output_tokens',
        'thinking_enabled', 'reasoning_effort', 'thinking_budget_tokens',
        'sampling_config', 'seed', 'started_at', 'completed_at',
        'total_time_ms', 'working_time_ms', 'input_tokens',
        'cached_input_tokens', 'output_tokens', 'reasoning_tokens',
        'calculated_cost_usd', 'trace_uri',
    }
    run = {key: result_input[key] for key in run_fields
           if key in result_input}
    run.update({
        'local_id': 'run',
        'system_id': mapping['system_id'],
        'benchmark_version_id': mapping['benchmark_version_id'],
        'evidence_local_id': 'evidence',
        'scope': submission['scope'],
        'status': 'completed',
        'sample_count_planned': planned_count,
        'sample_count_completed': sample_count or 0,
        'metadata': result_input.get('metadata', {}),
    })
    if mapping.get('model_checkpoint_id'):
        run['model_checkpoint_id'] = mapping['model_checkpoint_id']

    result_fields = {
        'result_level', 'attempt', 'lower_bound', 'upper_bound',
        'input_ref_uri', 'output_ref_uri', 'expectation_ref_uri', 'trace_uri',
        'input_tokens', 'cached_input_tokens', 'output_tokens',
        'reasoning_tokens', 'total_time_ms', 'working_time_ms', 'error_code',
        'error_message', 'is_invalid', 'invalidation_reason', 'is_primary',
    }
    result = {key: result_input[key] for key in result_fields
              if key in result_input}
    result.update({
        'run_local_id': 'run',
        'metric_id': mapping['metric_id'],
        'evidence_local_id': 'evidence',
        'scorer_key': mapping['scorer_key'],
        'value': submission['metric_value'],
        'sample_count': sample_count,
        'metadata': result_input.get('metadata', {}),
    })
    return {
        'idempotency_key': submission_document['idempotency_key'],
        'dry_run': True,
        'source': source,
        'evidence': [evidence],
        'runs': [run],
        'results': [result],
    }


def request_json(platform_url: str, path: str, token: str, *,
                 method: str = 'GET', body: Any = None,
                 timeout: int = 30, retries: int = 3) -> tuple[int, Any]:
    encoded = None if body is None else json.dumps(body).encode('utf-8')
    headers = {'Accept': 'application/json',
               'Authorization': f'Bearer {token}'}
    if encoded is not None:
        headers['Content-Type'] = 'application/json'
    target = f'{platform_url}{path}'
    for attempt in range(retries + 1):
        request = urllib.request.Request(target, data=encoded, headers=headers,
                                         method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode('utf-8')
                return response.status, json.loads(raw or '{}')
        except urllib.error.HTTPError as error:
            raw = error.read().decode('utf-8', errors='replace')
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {'error': 'invalid_response'}
            if (error.code == 429 or error.code >= 500) and attempt < retries:
                time.sleep(min(0.25 * (2 ** attempt), 10))
                continue
            code = payload.get('error', 'http_error') if isinstance(payload, dict) \
                else 'http_error'
            issues = payload.get('issues') if isinstance(payload, dict) else None
            raise AdapterError(f'Agent Eval API HTTP {error.code}: {code}; '
                               f'issues={issues!r}') from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt < retries:
                time.sleep(min(0.25 * (2 ** attempt), 10))
                continue
            raise AdapterError(f'Agent Eval API transport failure: {error.reason}') from error
    raise AdapterError('Agent Eval API retry loop ended unexpectedly')


def validate_receipt(payload: Any, *, dry_run: bool,
                     idempotency_key_value: str) -> None:
    if not isinstance(payload, dict) or payload.get('ok') is not True \
            or payload.get('dry_run') is not dry_run:
        raise AdapterError('Agent Eval API returned an invalid receipt')
    if dry_run and not isinstance(payload.get('normalized'), dict):
        raise AdapterError('Agent Eval dry-run receipt has no normalized object')
    if dry_run and payload['normalized'].get('idempotencyKey') \
            != idempotency_key_value:
        raise AdapterError('Agent Eval dry-run receipt has a different idempotency key')
    if not dry_run and (payload.get('status') != 'pending_review'
                        or not isinstance(payload.get('submission_id'), str)
                        or not isinstance(payload.get('review_id'), str)):
        raise AdapterError('Agent Eval apply receipt is not pending_review')


def validate_ingest_receipt(payload: Any, *, dry_run: bool,
                            idempotency_key_value: str) -> None:
    if (not isinstance(payload, dict) or payload.get('ok') is not True
            or payload.get('dry_run') is not dry_run):
        raise AdapterError('Agent Eval ingest API returned an invalid receipt')
    if dry_run:
        normalized_payload = payload.get('normalized')
        if (payload.get('validation_mode') != 'schema_only'
                or not isinstance(normalized_payload, dict)):
            raise AdapterError(
                'Agent Eval ingest dry-run has no schema-only normalized object')
        if normalized_payload.get('idempotencyKey') != idempotency_key_value:
            raise AdapterError(
                'Agent Eval ingest dry-run has a different idempotency key')
        return

    if payload.get('idempotency_key') != idempotency_key_value:
        raise AdapterError(
            'Agent Eval ingest apply receipt has a different idempotency key')
    if not isinstance(payload.get('replayed'), bool):
        raise AdapterError('Agent Eval ingest apply receipt has no replay flag')

    created = payload.get('created')
    ids = payload.get('ids')
    entity_types = ('evidence', 'runs', 'samples', 'results')
    if (not isinstance(created, dict) or any(
            not isinstance(created.get(entity), int)
            or created[entity] < 0 for entity in entity_types)):
        raise AdapterError(
            'Agent Eval ingest apply receipt has invalid created counts')
    if (not isinstance(ids, dict) or any(
            not isinstance(ids.get(entity), list)
            or any(not isinstance(value, str) for value in ids[entity])
            for entity in entity_types)):
        raise AdapterError('Agent Eval ingest apply receipt has invalid ids')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True,
                        choices=list(BENCHMARK_SPECS))
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--system', help='Exact Agent Eval system ID/name; defaults to --model')
    parser.add_argument('--scope', choices=['pretrained', 'posttrained',
                                            'agent_system'],
                        default='posttrained')
    parser.add_argument('--platform-url', required=True)
    parser.add_argument('--evidence-url', required=True,
                        help='HTTP(S) URL or template with {benchmark}, {run_id}, '
                             '{summary_file}, {summary_sha256}')
    parser.add_argument('--evidence-root', type=Path,
                        help='Publish the scored summary under this local root')
    parser.add_argument('--model-checkpoint-id',
                        help='Canonical Agent Eval model checkpoint ID')
    parser.add_argument('--provenance', choices=['internal_public',
                                                 'internal_private'],
                        default='internal_private')
    parser.add_argument('--publisher', default='OpenCompass Evaluation Team')
    parser.add_argument('--source', default='opencompass')
    parser.add_argument('--mapping-file', type=Path)
    parser.add_argument('--mode', choices=['review', 'ingest'], default='ingest',
                        help='ingest (default) writes canonical data directly '
                             'with INGEST_ADMIN_TOKEN; review creates a '
                             'pending human review')
    parser.add_argument('--token-env', default=TOKEN_ENV_DEFAULT)
    parser.add_argument('--ingest-token-env', default=INGEST_TOKEN_ENV_DEFAULT)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate with the platform without writing data '
                             'or creating a review')
    parser.add_argument('--catalog-file', type=Path,
                        help='Test/offline catalog envelope; skips catalog GET')
    parser.add_argument('--payload-only', action='store_true',
                        help='Build and save payload without POSTing')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.timeout <= 300:
        raise AdapterError('--timeout must be between 1 and 300 seconds')
    if not 0 <= args.retries <= 8:
        raise AdapterError('--retries must be between 0 and 8')
    platform_url = validate_http_url(args.platform_url, 'platform URL')
    if not args.run_dir.is_dir():
        raise AdapterError(f'run directory does not exist: {args.run_dir}')
    spec = load_spec(args.benchmark, args.mapping_file)
    summary = latest_summary(args.run_dir)
    score = read_primary_score(summary, spec, args.model)
    summary_hash = sha256_file(summary)
    evidence_url = render_evidence_url(args.evidence_url, args.benchmark,
                                       args.run_dir.name, summary, summary_hash)
    if args.evidence_root:
        publish_evidence(
            summary, args.evidence_root, args.benchmark, args.run_dir.name)
    catalog_token = os.environ.get(args.token_env, '').strip()
    offline_payload = bool(args.catalog_file and args.payload_only)
    if not catalog_token and not offline_payload:
        raise AdapterError(
            f'{args.token_env} is required and must come from the environment')
    ingest_token = os.environ.get(args.ingest_token_env, '').strip()
    if args.mode == 'ingest' and not ingest_token and not offline_payload:
        raise AdapterError(
            f'{args.ingest_token_env} is required for direct ingest and must '
            'come from the environment')
    if args.catalog_file:
        try:
            catalog = json.loads(args.catalog_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise AdapterError(f'cannot read catalog file {args.catalog_file}') from error
    else:
        _, catalog = request_json(platform_url, '/api/submissions/catalog',
                                  catalog_token,
                                  timeout=args.timeout, retries=args.retries)
    resolved = resolve_catalog(catalog, spec, args.system or args.model,
                               args.scope, score['version'])
    submission_payload = build_submission(
        benchmark=args.benchmark, model=args.model, run_dir=args.run_dir,
        summary=summary, score=score, spec=spec, resolved=resolved,
        evidence_url=evidence_url, provenance=args.provenance,
        publisher=args.publisher, source=args.source)
    if args.model_checkpoint_id:
        submission_payload['submission']['canonical_mapping'][
            'model_checkpoint_id'] = args.model_checkpoint_id
    payload = (build_ingest_payload(submission_payload, resolved, args.source)
               if args.mode == 'ingest' else submission_payload)
    receipt_name = (f'{args.benchmark}.ingest.json'
                    if args.mode == 'ingest' else f'{args.benchmark}.json')
    receipt_path = args.run_dir / 'agent_eval' / receipt_name
    prior = None
    if receipt_path.is_file():
        try:
            prior = json.loads(receipt_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            prior = None
    completed_status = 'ingested' if args.mode == 'ingest' else 'pending_review'
    if (prior and prior.get('idempotency_key') == payload['idempotency_key']
            and prior.get('summary_sha256') == summary_hash
            and prior.get('status') == completed_status and not args.dry_run):
        identity = (prior.get('ids') if args.mode == 'ingest'
                    else prior.get('submission_id'))
        print(f'Agent Eval already {completed_status}: {identity} '
              f'({receipt_path})')
        return 0

    record: dict[str, Any] = {
        'adapter_version': ADAPTER_VERSION,
        'mode': args.mode,
        'benchmark': args.benchmark,
        'run_dir': str(args.run_dir),
        'summary_file': str(summary),
        'summary_sha256': summary_hash,
        'idempotency_key': payload['idempotency_key'],
        'payload': payload,
    }
    if args.payload_only:
        record['operation'] = 'payload_only'
        atomic_json(receipt_path, record)
        print(f'Agent Eval payload written: {receipt_path}')
        return 0

    target_path = '/api/ingest' if args.mode == 'ingest' else '/api/submissions'
    write_token = ingest_token if args.mode == 'ingest' else catalog_token
    dry_payload = dict(payload, dry_run=True)
    dry_status, dry_receipt = request_json(
        platform_url, target_path, write_token, method='POST', body=dry_payload,
        timeout=args.timeout, retries=args.retries)
    if args.mode == 'ingest':
        validate_ingest_receipt(
            dry_receipt, dry_run=True,
            idempotency_key_value=payload['idempotency_key'])
    else:
        validate_receipt(dry_receipt, dry_run=True,
                         idempotency_key_value=payload['idempotency_key'])
    record['dry_run'] = {'http_status': dry_status, 'validated': True}
    if args.dry_run:
        record['operation'] = 'dry_run'
        atomic_json(receipt_path, record)
        print(f'Agent Eval dry-run validated: {receipt_path}')
        return 0

    apply_payload = dict(payload, dry_run=False)
    apply_status, apply_receipt = request_json(
        platform_url, target_path, write_token, method='POST', body=apply_payload,
        timeout=args.timeout, retries=args.retries)
    if args.mode == 'ingest':
        validate_ingest_receipt(
            apply_receipt, dry_run=False,
            idempotency_key_value=payload['idempotency_key'])
        record.update({
            'operation': 'apply',
            'http_status': apply_status,
            'status': 'ingested',
            'replayed': bool(apply_receipt.get('replayed')),
            'created': apply_receipt['created'],
            'ids': apply_receipt['ids'],
        })
        atomic_json(receipt_path, record)
        print('Agent Eval canonical ingest completed: '
              f'{record["ids"]} ({receipt_path})')
        return 0

    validate_receipt(apply_receipt, dry_run=False,
                     idempotency_key_value=payload['idempotency_key'])
    record.update({
        'operation': 'apply',
        'http_status': apply_status,
        'status': apply_receipt['status'],
        'replayed': bool(apply_receipt.get('replayed')),
        'submission_id': apply_receipt['submission_id'],
        'review_id': apply_receipt['review_id'],
    })
    atomic_json(receipt_path, record)
    print('Agent Eval submitted for review: '
          f'{record["submission_id"]} / {record["review_id"]} '
          f'({receipt_path})')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except AdapterError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2)
