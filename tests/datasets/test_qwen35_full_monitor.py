import csv
import os
from datetime import datetime

from script.monitor_qwen35_full_run import (EXPECTED_SAMPLES,
                                            OFFICIAL_SCORES,
                                            fatal_stream_error_after_last_prediction,
                                            summary_score)


def write_summary(run_dir, rows):
    path = run_dir / 'summary/summary_test.csv'
    path.parent.mkdir(parents=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'version', 'metric', 'mode', 'model'])
        writer.writerows(rows)
    return path


def test_summary_score_selects_the_published_ifeval_metric(tmp_path):
    summary = write_summary(tmp_path, [
        ['IFEval', 'v1', 'Prompt-level-loose-accuracy', 'gen', '91.0'],
        ['IFEval', 'v1', 'Prompt-level-strict-accuracy', 'gen', '89.8'],
    ])

    assert summary_score('ifeval', tmp_path) == (89.8, str(summary))


def test_summary_score_covers_local_math_and_multilingual_aliases(tmp_path):
    fixtures = {
        'aime_2026': ('aime2026', 'accuracy'),
        'hmmt_feb_2026': ('hmmt2026', 'accuracy'),
        'hmmt_feb_2025': ('hmmt_feb_2025_full_chat', 'accuracy'),
        'global_piqa': ('global_piqa_generation', 'accuracy'),
    }
    for index, (benchmark, (dataset, metric)) in enumerate(fixtures.items()):
        run_dir = tmp_path / benchmark
        summary = write_summary(
            run_dir, [[dataset, 'v1', metric, 'gen', str(40 + index)]])
        assert summary_score(benchmark, run_dir) == (40 + index,
                                                     str(summary))


def test_unpublished_math_scores_are_not_invented():
    for benchmark in ('aime_2024', 'aime_2025', 'aime_2026',
                      'hmmt_feb_2026'):
        assert benchmark not in OFFICIAL_SCORES

    assert EXPECTED_SAMPLES['aime_2026'] == 120
    assert EXPECTED_SAMPLES['hmmt_feb_2026'] == 132
    assert EXPECTED_SAMPLES['hmmt_feb_2025'] == 120
    assert EXPECTED_SAMPLES['hmmt_nov_2025'] == 120


def test_final_stream_error_after_prediction_marks_executor_drain(tmp_path):
    prediction = tmp_path / 'predictions/model/tmp_data.jsonl'
    prediction.parent.mkdir(parents=True)
    prediction.write_text('{}\n', encoding='utf-8')
    prediction_time = datetime(2026, 7, 29, 21, 56, 7).timestamp()
    os.utime(prediction, (prediction_time, prediction_time))

    log = tmp_path / 'logs/infer/model/data.out'
    log.parent.mkdir(parents=True)
    log.write_text(
        '07/29 21:56:20 - OpenCompass - ERROR - '
        'vLLM streaming chat request failed (attempt 1/1): timed out\n',
        encoding='utf-8')

    error = fatal_stream_error_after_last_prediction(
        tmp_path, [prediction], 2026)

    assert error is not None
    assert error['timestamp'] == '2026-07-29T21:56:20'


def test_nonfinal_stream_retry_does_not_mark_executor_drain(tmp_path):
    prediction = tmp_path / 'predictions/model/tmp_data.jsonl'
    prediction.parent.mkdir(parents=True)
    prediction.write_text('{}\n', encoding='utf-8')
    prediction_time = datetime(2026, 7, 29, 21, 56, 7).timestamp()
    os.utime(prediction, (prediction_time, prediction_time))

    log = tmp_path / 'logs/infer/model/data.out'
    log.parent.mkdir(parents=True)
    log.write_text(
        '07/29 21:56:20 - OpenCompass - ERROR - '
        'vLLM streaming chat request failed (attempt 1/3): timed out\n',
        encoding='utf-8')

    assert fatal_stream_error_after_last_prediction(
        tmp_path, [prediction], 2026) is None
