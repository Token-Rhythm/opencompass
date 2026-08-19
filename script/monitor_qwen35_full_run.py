#!/usr/bin/env python3
"""Build an evidence-backed progress report for a Qwen3.5 full-suite run.

The report reads OpenCompass artifacts only. Output-token counts are obtained
by re-tokenizing the persisted reasoning/content fields with the official
Qwen3.5 tokenizer; they exclude chat framing and are therefore slightly lower
than vLLM's completion-token counter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


OFFICIAL_SCORES = {
    'mmlu_pro': 79.1,
    'mmlu_redux': 88.8,
    'ceval': 85.1,
    'supergpqa': 52.9,
    'gpqa_diamond': 76.2,
    'ifeval': 89.8,
    'ifbench': 59.2,
    'aa_lcr': 57.0,
    'longbench_v2': 50.0,
    'hmmt_feb_2025': 74.0,
    'hmmt_nov_2025': 76.8,
    'livecodebench': 55.8,
    'mmmlu': 76.1,
    'mmlu_prox': 71.5,
    'include': 71.0,
    'global_piqa': 78.9,
}

PRIMARY_DATASETS = {
    'mmlu_pro': ('mmlu_pro',),
    'mmlu_redux': ('mmlu_redux', 'mmlu_redux_full_chat'),
    'ceval': ('ceval',),
    'supergpqa': ('supergpqa',),
    'gpqa_diamond': ('GPQA_diamond', 'gpqa_diamond'),
    'ifeval': ('IFEval', 'ifeval'),
    'ifbench': ('IFBench', 'ifbench'),
    'aa_lcr': ('aa_lcr',),
    'longbench_v2': ('LongBenchv2', 'longbench_v2'),
    'hmmt_feb_2025': ('hmmt_feb_2025_full_chat', 'hmmt_feb_2025'),
    'hmmt_nov_2025': ('hmmt_nov_2025_full_chat', 'hmmt_nov_2025'),
    'livecodebench': ('livecodebench_v6_codegen', 'livecodebench'),
    'mmmlu': ('mmmlu',),
    'mmlu_prox': ('mmlu_prox',),
    'include': ('include_base_44',),
    'global_piqa': ('global_piqa_generation', 'global_piqa'),
    'aime_2024': ('aime2024', 'aime_2024'),
    'aime_2025': ('aime2025', 'aime_2025'),
    'aime_2026': ('aime2026', 'aime_2026'),
    'hmmt_feb_2026': ('hmmt2026', 'hmmt_feb_2026'),
}

# Metrics used for the model-card comparison. In particular, IFEval's
# published number is prompt-level strict accuracy and IFBench's ``score`` is
# its prompt-level loose accuracy. Avoid silently selecting a different row
# merely because it appears first in a generated CSV.
PRIMARY_METRICS = {
    'mmlu_pro': ('accuracy',),
    'mmlu_redux': ('accuracy',),
    'ceval': ('naive_average',),
    'supergpqa': ('accuracy',),
    'gpqa_diamond': ('accuracy',),
    'ifeval': ('Prompt-level-strict-accuracy',),
    'ifbench': ('score', 'Prompt-level-loose-accuracy'),
    'aa_lcr': ('accuracy',),
    'longbench_v2': ('accuracy',),
    'hmmt_feb_2025': ('accuracy',),
    'hmmt_nov_2025': ('accuracy',),
    'livecodebench': ('pass@1',),
    'mmmlu': ('accuracy',),
    'mmlu_prox': ('accuracy',),
    'include': ('accuracy',),
    'global_piqa': ('accuracy',),
    'aime_2024': ('accuracy',),
    'aime_2025': ('accuracy',),
    'aime_2026': ('accuracy',),
    'hmmt_feb_2026': ('accuracy',),
}

EXPECTED_SAMPLES = {
    'mmlu_pro': 12032,
    'mmlu_redux': 5330,
    'ceval': 1346,
    'supergpqa': 26529,
    'ifeval': 541,
    'ifbench': 300,
    'aa_lcr': 100,
    'longbench_v2': 503,
    'gpqa_diamond': 792,
    'aime_2024': 30,
    'aime_2025': 30,
    # These counts are the expanded MathArena rows after applying the
    # configured four runs per source problem (30*4 or 33*4).
    'aime_2026': 120,
    'hmmt_feb_2026': 132,
    'livecodebench': 175,
    'hmmt_feb_2025': 120,
    'hmmt_nov_2025': 120,
    'include': 22639,
}

OFFICIAL_SOURCE = 'https://huggingface.co/Qwen/Qwen3.5-4B'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--tokenizer', default='Qwen/Qwen3.5-4B')
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--output-md', type=Path)
    parser.add_argument('--skip-tokens', action='store_true')
    return parser.parse_args()


def load_prediction_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    prediction_root = run_dir / 'predictions'
    final_files = sorted(prediction_root.glob('*/*.json'))
    rows: list[dict[str, Any]] = []
    used_files: list[Path] = []
    if final_files:
        for path in final_files:
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            values = payload.values() if isinstance(payload, dict) else payload
            if not isinstance(values, (list, tuple)) and not hasattr(values, '__iter__'):
                continue
            file_rows = [item for item in values if isinstance(item, dict)]
            rows.extend(file_rows)
            used_files.append(path)
        return rows, used_files

    for path in sorted(prediction_root.glob('*/tmp_*.jsonl')):
        file_rows = []
        try:
            with path.open(encoding='utf-8') as file:
                for line in file:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        file_rows.append(item)
        except OSError:
            continue
        rows.extend(file_rows)
        used_files.append(path)
    return rows, used_files


def output_texts(row: dict[str, Any]) -> tuple[str, str]:
    if 'content' in row or 'reasoning_content' in row:
        return (str(row.get('reasoning_content') or ''),
                str(row.get('content') or ''))
    return '', str(row.get('prediction') or '')


def percentile(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def token_stats(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    reasoning, content = zip(*(output_texts(row) for row in rows)) if rows else ((), ())

    def lengths(texts: tuple[str, ...]) -> list[int]:
        result: list[int] = []
        for offset in range(0, len(texts), 256):
            encoded = tokenizer(list(texts[offset:offset + 256]),
                                add_special_tokens=False,
                                return_length=True,
                                truncation=False)
            result.extend(encoded['length'])
        return result

    reasoning_lengths = lengths(reasoning)
    content_lengths = lengths(content)
    totals = [left + right for left, right in zip(reasoning_lengths,
                                                   content_lengths)]
    if not totals:
        return {}
    return {
        'output_tokens_total': sum(totals),
        'output_tokens_mean': round(statistics.mean(totals), 2),
        'output_tokens_median': statistics.median(totals),
        'output_tokens_p95': percentile(totals, 0.95),
        'output_tokens_max': max(totals),
        'reasoning_tokens_total': sum(reasoning_lengths),
        'content_tokens_total': sum(content_lengths),
        'empty_content': sum(not value.strip() for value in content),
        'near_32k_cap': sum(value >= 32760 for value in totals),
    }


def summary_score(benchmark: str, run_dir: Path) -> tuple[float | None, str | None]:
    summaries = sorted((run_dir / 'summary').glob('summary_*.csv'))
    aliases = PRIMARY_DATASETS.get(benchmark, (benchmark,))
    metrics = PRIMARY_METRICS.get(benchmark)
    for path in reversed(summaries):
        with path.open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        for alias in aliases:
            candidates = [item for item in rows
                          if item.get('dataset') == alias]
            if metrics is not None:
                row = next((item for metric in metrics
                            for item in candidates
                            if item.get('metric') == metric), None)
            else:
                row = candidates[0] if candidates else None
            if row is None:
                continue
            raw = str(list(row.values())[-1]).strip()
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                return value, str(path)
    return None, None


def first_log_timestamp(run_dir: Path, year: int) -> datetime | None:
    pattern = re.compile(r'(?m)^(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})')
    found: list[datetime] = []
    for path in (run_dir / 'logs').glob('**/*.out'):
        try:
            match = pattern.search(path.read_text(encoding='utf-8', errors='replace'))
        except OSError:
            continue
        if match:
            found.append(datetime(year, *(int(value) for value in match.groups())))
    return min(found) if found else None


def inference_seconds(run_dir: Path) -> float | None:
    pattern = re.compile(r'time elapsed: ([0-9.]+)s')
    values = []
    for path in (run_dir / 'logs/infer').glob('**/*.out'):
        try:
            values.extend(float(value) for value in pattern.findall(
                path.read_text(encoding='utf-8', errors='replace')))
        except OSError:
            continue
    return max(values) if values else None


def fatal_stream_error_after_last_prediction(
        run_dir: Path, prediction_files: list[Path],
        year: int) -> dict[str, str] | None:
    """Detect a failed future whose executor is still draining queued work.

    ``ParallelGenInferencer`` currently calls ``future.result()`` inside a
    ``ThreadPoolExecutor`` context.  A final-attempt streaming error escapes
    that loop, so result persistence stops immediately, while executor
    shutdown continues waiting for every already-submitted future.  The
    process and API traffic therefore remain alive even though no later
    prediction can be saved.  An ERROR on the final retry after the newest
    persisted prediction is authoritative evidence of that state.
    """
    if not prediction_files:
        return None
    last_prediction = datetime.fromtimestamp(
        max(path.stat().st_mtime for path in prediction_files))
    timestamp_pattern = re.compile(
        r'^(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})')
    attempt_pattern = re.compile(
        r'vLLM streaming chat request (?:failed|timed out).*'
        r'\(attempt (\d+)/(\d+)\)')
    fatal: dict[str, str] | None = None
    for path in sorted((run_dir / 'logs/infer').glob('**/*.out')):
        try:
            lines = path.read_text(encoding='utf-8',
                                   errors='replace').splitlines()
        except OSError:
            continue
        for line in lines:
            timestamp_match = timestamp_pattern.match(line)
            attempt_match = attempt_pattern.search(line)
            if timestamp_match is None or attempt_match is None:
                continue
            if attempt_match.group(1) != attempt_match.group(2):
                continue
            timestamp = datetime(
                year, *(int(value) for value in timestamp_match.groups()))
            if timestamp <= last_prediction:
                continue
            fatal = {
                'timestamp': timestamp.isoformat(),
                'log': str(path),
                'message': line,
            }
    return fatal


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return '-'
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def collect(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = None
    if not args.skip_tokens:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                  trust_remote_code=True)
    now = datetime.now()
    year = int(args.run_id[:4]) if args.run_id[:4].isdigit() else now.year
    benchmarks = []
    for benchmark_dir in sorted(path for path in args.root.iterdir()
                                if path.is_dir() and not path.name.startswith('_')):
        benchmark = benchmark_dir.name
        run_dir = benchmark_dir / args.run_id
        if not run_dir.is_dir():
            continue
        rows, prediction_files = load_prediction_rows(run_dir)
        score, summary = summary_score(benchmark, run_dir)
        start = first_log_timestamp(run_dir, year)
        if summary:
            end = datetime.fromtimestamp(Path(summary).stat().st_mtime)
        else:
            activity = [path.stat().st_mtime for path in prediction_files]
            activity.extend(path.stat().st_mtime
                            for path in (run_dir / 'logs').glob('**/*.out'))
            end = datetime.fromtimestamp(max(activity)) if activity else None
        wall_seconds = ((end - start).total_seconds()
                        if start is not None and end is not None else None)
        expected = EXPECTED_SAMPLES.get(benchmark)
        if score is not None:
            state = 'complete'
        elif rows:
            state = 'partial'
        else:
            state = 'started'
        fatal_stream_error = (
            fatal_stream_error_after_last_prediction(
                run_dir, prediction_files, year)
            if score is None else None)
        if fatal_stream_error is not None:
            state = 'failed-draining'
        entry: dict[str, Any] = {
            'benchmark': benchmark,
            'state': state,
            'samples_persisted': len(rows),
            'samples_expected': expected,
            'progress_percent': (round(100 * len(rows) / expected, 2)
                                 if expected else None),
            'score': score,
            'official_score': OFFICIAL_SCORES.get(benchmark),
            'delta_pp': (round(score - OFFICIAL_SCORES[benchmark], 2)
                         if score is not None and benchmark in OFFICIAL_SCORES
                         else None),
            'inference_seconds': inference_seconds(run_dir),
            'wall_seconds_to_last_artifact': wall_seconds,
            'wall_seconds_elapsed': ((end - start).total_seconds()
                                     if state == 'complete' and start and end
                                     else (now - start).total_seconds()
                                     if start else None),
            'last_activity': end.isoformat() if end else None,
            'summary': summary,
            'prediction_files': [str(path) for path in prediction_files],
            'fatal_stream_error': fatal_stream_error,
        }
        if tokenizer is not None:
            entry.update(token_stats(rows, tokenizer))
        benchmarks.append(entry)
    return {
        'generated_at': now.isoformat(),
        'root': str(args.root),
        'run_id': args.run_id,
        'token_count_method': ('official tokenizer over persisted reasoning + '
                               'content; excludes chat framing'),
        'official_source': OFFICIAL_SOURCE,
        'benchmarks': benchmarks,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        '# Qwen3.5-4B 全量评测监控', '',
        f"更新时间：{report['generated_at']}", '',
        f"Run ID：`{report['run_id']}`", '',
        '| Benchmark | 状态 | 进度 | 推理耗时 | 总/当前墙钟 | 分数 | 官网 | 差值(pp) | 输出 tokens | 平均/题 | P95 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for item in report['benchmarks']:
        expected = item.get('samples_expected')
        progress = (f"{item['samples_persisted']}/{expected}"
                    if expected else str(item['samples_persisted']))
        values = [
            item['benchmark'], item['state'], progress,
            format_duration(item.get('inference_seconds')),
            format_duration(item.get('wall_seconds_elapsed')),
            '-' if item.get('score') is None else f"{item['score']:.2f}",
            '-' if item.get('official_score') is None else f"{item['official_score']:.2f}",
            '-' if item.get('delta_pp') is None else f"{item['delta_pp']:+.2f}",
            f"{item.get('output_tokens_total', 0):,}",
            '-' if item.get('output_tokens_mean') is None else f"{item['output_tokens_mean']:,.1f}",
            '-' if item.get('output_tokens_p95') is None else f"{item['output_tokens_p95']:,}",
        ]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += [
        '',
        '说明：token 数由官方 tokenizer 对持久化的 `reasoning_content + content` '
        '重新编码，未包含 chat framing，通常会略低于 vLLM completion token 计数。',
        '',
        f"官网来源：{report['official_source']}",
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.output_json = (args.output_json
                        or args.root / f'monitor_{args.run_id}.json')
    args.output_md = (args.output_md
                      or args.root / f'monitor_{args.run_id}.md')
    report = collect(args)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False,
                                           indent=2) + '\n',
                                encoding='utf-8')
    args.output_md.write_text(render_markdown(report), encoding='utf-8')
    print(args.output_md)


if __name__ == '__main__':
    main()
