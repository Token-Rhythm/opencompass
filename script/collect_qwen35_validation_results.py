#!/usr/bin/env python3
"""Collect completed Qwen3.5-2B validation artifacts into one JSON report."""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

try:
    from script.verify_multilingual_prediction_coverage import (
        verify_global_piqa, verify_include, verify_mmlu_prox)
except ModuleNotFoundError:  # Direct execution adds script/ to sys.path.
    from verify_multilingual_prediction_coverage import (
        verify_global_piqa, verify_include, verify_mmlu_prox)


REPO = Path(__file__).resolve().parents[1]
MMLU_PROX_LANGUAGES = (
    'en', 'ja', 'zh', 'ko', 'fr', 'de', 'es', 'pt', 'zu', 'sw', 'wo',
    'yo', 'th', 'ar', 'hi', 'bn', 'mr', 'ne', 'af', 'te', 'ur', 'ru',
    'id', 'vi', 'cs', 'hu', 'it', 'sr', 'uk')


def load_json(path: Path):
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def latest_path(pattern: str) -> Path | None:
    paths = sorted(REPO.glob(pattern))
    return paths[-1] if paths else None


def prediction_rows_for_summary(summary: Path, filename: str) -> dict | None:
    run_dir = summary.parent.parent
    paths = list(run_dir.glob(f'predictions/*/{filename}'))
    if len(paths) != 1:
        return None
    rows = load_json(paths[0])
    return rows if isinstance(rows, dict) else None


def prediction_count_for_summary(summary: Path, filename: str) -> int | None:
    rows = prediction_rows_for_summary(summary, filename)
    return len(rows) if rows is not None else None


def final_answer_counts(summary: Path, filename: str) -> dict:
    rows = prediction_rows_for_summary(summary, filename) or {}
    blank = sum(
        not str(row.get('prediction', '')).strip() for row in rows.values())
    return {
        'blank_final_answers': blank,
        'nonempty_final_answers': len(rows) - blank,
    }


def longbenchv2_parse_counts(summary: Path, filename: str) -> dict:
    """Count answers accepted by the official LongBench v2 patterns.

    A non-empty response is not necessarily scoreable: the public evaluator
    only accepts ``The correct answer is (X)`` or the parenthesis-free
    equivalent.  Keep this audit next to the aggregate score so malformed
    final answers cannot be mistaken for ordinary wrong multiple-choice
    answers.
    """
    rows = prediction_rows_for_summary(summary, filename) or {}
    patterns = (
        re.compile(r'The correct answer is \(([A-D])\)'),
        re.compile(r'The correct answer is ([A-D])'),
    )
    parseable = 0
    for row in rows.values():
        response = str(row.get('prediction', '')).replace('*', '')
        if any(pattern.search(response) for pattern in patterns):
            parseable += 1
    return {
        'parseable_final_answers': parseable,
        'unparseable_final_answers': len(rows) - parseable,
    }


def summary_score(root: str,
                  datasets: tuple[str, ...],
                  prediction_file: str = None,
                  expected_rows: int = None,
                  run_id: str = None):
    summary_pattern = (f'{run_id}/summary/summary_*.csv'
                       if run_id else '2*/summary/summary_*.csv')
    for summary in sorted(
            (REPO / root).glob(summary_pattern), reverse=True):
        with summary.open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        for dataset in datasets:
            row = next((item for item in rows
                        if item.get('dataset') == dataset), None)
            if not row:
                continue
            value = list(row.values())[-1].strip()
            if not value or value == '-':
                continue
            if prediction_file and prediction_count_for_summary(
                    summary, prediction_file) != expected_rows:
                continue
            return float(value), summary
    return None


def mmlu_prox_summary_score(root: str):
    """Read the official overall, with a strict legacy-summary fallback.

    Older MMLU-ProX summarizer configs computed the overall group internally
    but omitted it from the CSV display list.  Their 29 weighted language
    groups are still present, and their arithmetic mean is exactly equivalent
    to the official overall because every language uses the same category
    weights.  Require every official language so a partial run cannot pass.
    """
    expected = {
        f'mmlu_prox_5shot_{language}' for language in MMLU_PROX_LANGUAGES
    }
    for summary in sorted(
            (REPO / root).glob('2*/summary/summary_*.csv'), reverse=True):
        with summary.open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        overall = next(
            (row for row in rows if row.get('dataset') == 'mmlu_prox'), None)
        if overall:
            value = list(overall.values())[-1].strip()
            if value and value != '-':
                return float(value), summary
        language_scores = {}
        for row in rows:
            dataset = row.get('dataset')
            if dataset not in expected:
                continue
            value = list(row.values())[-1].strip()
            if value and value != '-':
                language_scores[dataset] = float(value)
        if set(language_scores) == expected:
            return sum(language_scores.values()) / len(expected), summary
    return None


def mark_complete(benchmarks, key, score, metric, evidence, result_type,
                  **details):
    entry = benchmarks[key]
    entry.update(
        state='complete',
        score=score,
        metric=metric,
        evidence=str(evidence.relative_to(REPO)),
        result_type=result_type,
        **details,
    )


def collect(state_path: Path) -> dict:
    state = load_json(state_path)
    benchmarks = state['benchmarks']

    result = summary_score(
        'outputs/longbenchv2_full_qwen3.5_2b_thinking_262144_32768_'
        'token_mid_stream16_batch16_idle600', ('LongBenchv2',),
        'LongBenchv2.json', 503, run_id='20260726_173404')
    if result:
        score, evidence = result
        mark_complete(
            benchmarks, 'longbench_v2', score, 'accuracy', evidence,
            'qwen_native_thinking_official_scorer_32k_token_mid', total=503,
            max_seq_len=262144, max_out_len=32768,
            truncated_inputs=113,
            **final_answer_counts(evidence, 'LongBenchv2.json'),
            **longbenchv2_parse_counts(evidence, 'LongBenchv2.json'))

    aa_lcr_root = (
        REPO / 'outputs/aa_lcr_full_qwen3.5_2b_thinking_262144_32768_'
        'stream8_batch16')
    aa_lcr_run_id = '20260726_163358'
    result = summary_score(
        str(aa_lcr_root.relative_to(REPO)), ('aa_lcr',), 'aa_lcr.json', 100,
        run_id=aa_lcr_run_id)
    if result:
        score, evidence = result
        mark_complete(
            benchmarks, 'aa_lcr', score, 'accuracy', evidence,
            'official_equality_judge_qwen3_235b_non_thinking', total=100,
            max_seq_len=262144, max_out_len=32768,
            **final_answer_counts(evidence, 'aa_lcr.json'))
    else:
        aa_lcr_prediction = (
            aa_lcr_root / aa_lcr_run_id / 'predictions/Qwen3.5-2B-chat/'
            'aa_lcr.json')
        if aa_lcr_prediction.is_file():
            rows = load_json(aa_lcr_prediction)
            if isinstance(rows, dict) and len(rows) == 100:
                blank = sum(
                    not str(row.get('prediction', '')).strip()
                    for row in rows.values())
                benchmarks['aa_lcr'].update(
                    state='generation_complete_pending_official_judge',
                    score=None,
                    metric=None,
                    result_type='official_generation_pending_judge',
                    evidence=str(aa_lcr_prediction.relative_to(REPO)),
                    total=100,
                    blank_final_answers=blank,
                    nonempty_final_answers=100 - blank,
                    max_seq_len=262144,
                    max_out_len=32768,
                )

    ceval = latest_path(
        'outputs/ceval_evalscope_merged_32768_stream512/**/'
        'ceval_merged_metrics.json')
    if ceval:
        report = load_json(ceval)
        after = report['after']
        if len(after['subjects']) == 52:
            mark_complete(
                benchmarks, 'ceval', after['macro_accuracy'],
                'subject_macro_accuracy', ceval, 'strict_merged_diagnostic',
                micro_accuracy=after['micro_accuracy'],
                total=after['total'], parsed=after['parsed'])

    supergpqa = latest_path(
        'outputs/supergpqa_validation_merged_32768_stream64/**/'
        'supergpqa_merged_metrics.json')
    if supergpqa:
        report = load_json(supergpqa)
        after = report['after']
        if after['total'] == 674:
            mark_complete(
                benchmarks, 'supergpqa', after['accuracy'], 'accuracy',
                supergpqa, 'stratified_diagnostic', total=after['total'],
                parsed=after['parsed'], correct=after['correct'])

    result = summary_score(
        'outputs/hmmt_feb_2026_full_chat_qwen3.5_2b_thinking_stream33',
        ('hmmt_feb_2026', 'hmmt2026'), 'hmmt2026.json', 132,
        run_id='20260725_213038')
    if result:
        score, evidence = result
        mark_complete(benchmarks, 'hmmt_feb_2026', score, 'accuracy',
                      evidence, 'official_dataset_scorer_32k_cap', total=132,
                      **final_answer_counts(evidence, 'hmmt2026.json'))

    result = summary_score(
        'outputs/livecodebench_v6_codegen_qwen3.5_2b_thinking_stream175',
        ('livecodebench_v6_codegen',), 'livecodebench_v6_codegen.json', 175,
        run_id='20260725_213506')
    if result:
        score, evidence = result
        mark_complete(benchmarks, 'livecodebench_v6', score, 'pass@1',
                      evidence, 'official_dataset_scorer_32k_cap', total=175,
                      **final_answer_counts(
                          evidence, 'livecodebench_v6_codegen.json'))

    hmmt_root = 'outputs/hmmt_2025_full_chat_qwen3.5_2b_thinking_stream60'
    feb = summary_score(hmmt_root, ('hmmt_feb_2025_full_chat',),
                        'hmmt_feb_2025_full_chat.json', 120,
                        run_id='20260725_225833')
    nov = summary_score(hmmt_root, ('hmmt_nov_2025_full_chat',),
                        'hmmt_nov_2025_full_chat.json', 120,
                        run_id='20260725_225833')
    if feb and nov and feb[1] == nov[1]:
        mark_complete(benchmarks, 'hmmt_feb_2025', feb[0], 'accuracy',
                      feb[1], 'official_dataset_scorer_32k_cap', total=120,
                      **final_answer_counts(
                          feb[1], 'hmmt_feb_2025_full_chat.json'))
        mark_complete(benchmarks, 'hmmt_nov_2025', nov[0], 'accuracy',
                      nov[1], 'official_dataset_scorer_32k_cap', total=120,
                      **final_answer_counts(
                          nov[1], 'hmmt_nov_2025_full_chat.json'))

    mmmlu = latest_path(
        'outputs/mmmlu_validation_merged_32768_stream56/**/'
        'mmmlu_merged_metrics.json')
    if mmmlu:
        report = load_json(mmmlu)
        after = report['after']
        if len(after['languages']) == 14 and after['total'] == 280:
            mark_complete(
                benchmarks, 'mmmlu', after['official_weighted_accuracy'],
                'official_weighted_accuracy', mmmlu,
                'stratified_diagnostic', total=280, parsed=after['parsed'])

    multilingual = {
        'global_piqa': (
            'outputs/global_piqa_validation_1_per_config_qwen3.5_2b_'
            'thinking_8192_stream64', ('global_piqa_generation',),
            verify_global_piqa, 'Qwen3.5-2B-chat', 'accuracy'),
        'include_base_44': (
            'outputs/include_base_44_qwen3.5_2b_full',
            ('include_base_44',), verify_include,
            'Qwen3.5-2B-completions', 'accuracy'),
    }
    for key, (root, datasets, verifier, model_abbr, metric) in multilingual.items():
        result = summary_score(root, datasets)
        if not result:
            continue
        score, summary = result
        prediction_dir = summary.parent.parent / 'predictions' / model_abbr
        try:
            coverage = verifier(prediction_dir)
        except SystemExit:
            continue
        mark_complete(benchmarks, key, score, metric, summary,
                      'official_full' if key == 'include_base_44'
                      else 'stratified_diagnostic', **coverage)

    mmlu_prox_root = (
        'outputs/mmlu_prox_validation_1_per_language_subject_5shot_'
        'qwen3.5_2b_thinking_16384_stream32')
    result = mmlu_prox_summary_score(mmlu_prox_root)
    if result:
        score, summary = result
        prediction_dir = (summary.parent.parent / 'predictions' /
                          'Qwen3.5-2B-chat')
        try:
            coverage = verify_mmlu_prox(prediction_dir)
        except SystemExit:
            pass
        else:
            mark_complete(benchmarks, 'mmlu_prox', score, 'accuracy', summary,
                          'stratified_diagnostic', **coverage)

    state['collected_at'] = datetime.now().astimezone().isoformat(
        timespec='seconds')
    completed = sorted(
        key for key, entry in benchmarks.items()
        if entry.get('state') == 'complete')
    audited_skipped = sorted(
        key for key, entry in benchmarks.items()
        if str(entry.get('state', '')).startswith('skipped_'))
    terminal = set(completed) | set(audited_skipped)
    state['validation_summary'] = {
        'target_count': len(benchmarks),
        'completed_count': len(completed),
        'audited_skipped_count': len(audited_skipped),
        'pending_count': len(benchmarks) - len(terminal),
        'completed': completed,
        'audited_skipped': audited_skipped,
        'pending': sorted(set(benchmarks) - terminal),
    }
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--state', type=Path,
        default=REPO / 'outputs/qwen35_validation/benchmark_results.json')
    args = parser.parse_args()
    print(json.dumps(collect(args.state), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
