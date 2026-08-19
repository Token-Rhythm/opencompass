import csv
import json

from script.collect_qwen35_validation_results import (
    MMLU_PROX_LANGUAGES, final_answer_counts, longbenchv2_parse_counts,
    mmlu_prox_summary_score, prediction_count_for_summary, summary_score)


def test_summary_score_requires_exact_prediction_count(tmp_path, monkeypatch):
    import script.collect_qwen35_validation_results as module
    monkeypatch.setattr(module, 'REPO', tmp_path)
    run = tmp_path / 'outputs/example/20260725_000000'
    summary = run / 'summary/summary_20260725_000000.csv'
    summary.parent.mkdir(parents=True)
    with summary.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'metric', 'model'])
        writer.writerow(['example', 'accuracy', '42.5'])
    prediction = run / 'predictions/model/example.json'
    prediction.parent.mkdir(parents=True)
    prediction.write_text(json.dumps({'0': {}, '1': {}}), encoding='utf-8')

    assert prediction_count_for_summary(summary, 'example.json') == 2
    assert summary_score('outputs/example', ('example',), 'example.json', 3) \
        is None
    assert summary_score('outputs/example', ('example',), 'example.json', 2) \
        == (42.5, summary)
    assert final_answer_counts(summary, 'example.json') == {
        'blank_final_answers': 2,
        'nonempty_final_answers': 0,
    }


def test_longbenchv2_parse_counts_uses_official_patterns(tmp_path,
                                                        monkeypatch):
    import script.collect_qwen35_validation_results as module
    monkeypatch.setattr(module, 'REPO', tmp_path)
    run = tmp_path / 'outputs/example/20260725_000000'
    summary = run / 'summary/summary_20260725_000000.csv'
    summary.parent.mkdir(parents=True)
    summary.write_text('dataset,metric,model\n', encoding='utf-8')
    prediction = run / 'predictions/model/example.json'
    prediction.parent.mkdir(parents=True)
    prediction.write_text(json.dumps({
        '0': {'prediction': 'The correct answer is (A)'},
        '1': {'prediction': '**The correct answer is B**'},
        '2': {'prediction': 'I choose C'},
        '3': {'prediction': ''},
    }), encoding='utf-8')

    assert longbenchv2_parse_counts(summary, 'example.json') == {
        'parseable_final_answers': 2,
        'unparseable_final_answers': 2,
    }


def test_summary_score_pinned_run_never_falls_back_to_an_older_run(
        tmp_path, monkeypatch):
    import script.collect_qwen35_validation_results as module

    monkeypatch.setattr(module, 'REPO', tmp_path)
    old_run = tmp_path / 'outputs/example/20260725_000000'
    old_summary = old_run / 'summary/summary_20260725_000000.csv'
    old_summary.parent.mkdir(parents=True)
    old_summary.write_text(
        'dataset,metric,model\nexample,accuracy,99.0\n',
        encoding='utf-8')
    old_prediction = old_run / 'predictions/model/example.json'
    old_prediction.parent.mkdir(parents=True)
    old_prediction.write_text(json.dumps({'0': {}}), encoding='utf-8')

    assert summary_score(
        'outputs/example', ('example',), 'example.json', 1,
        run_id='20260725_010000') is None

    new_run = tmp_path / 'outputs/example/20260725_010000'
    new_summary = new_run / 'summary/summary_20260725_010000.csv'
    new_summary.parent.mkdir(parents=True)
    new_summary.write_text(
        'dataset,metric,model\nexample,accuracy,42.0\n',
        encoding='utf-8')
    new_prediction = new_run / 'predictions/model/example.json'
    new_prediction.parent.mkdir(parents=True)
    new_prediction.write_text(json.dumps({'0': {}}), encoding='utf-8')

    assert summary_score(
        'outputs/example', ('example',), 'example.json', 1,
        run_id='20260725_010000') == (42.0, new_summary)


def test_mmlu_prox_legacy_summary_requires_all_language_groups(
        tmp_path, monkeypatch):
    import script.collect_qwen35_validation_results as module
    monkeypatch.setattr(module, 'REPO', tmp_path)
    summary = (tmp_path / 'outputs/mmlu_prox/20260725_000000/summary/'
               'summary_20260725_000000.csv')
    summary.parent.mkdir(parents=True)
    with summary.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['dataset', 'metric', 'model'])
        for index, language in enumerate(MMLU_PROX_LANGUAGES):
            writer.writerow([
                f'mmlu_prox_5shot_{language}', 'accuracy', index
            ])
    expected = sum(range(len(MMLU_PROX_LANGUAGES))) / len(
        MMLU_PROX_LANGUAGES)
    assert mmlu_prox_summary_score('outputs/mmlu_prox') == (expected,
                                                            summary)

    lines = summary.read_text(encoding='utf-8').splitlines()
    summary.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    assert mmlu_prox_summary_score('outputs/mmlu_prox') is None


def test_collector_does_not_overclaim_32k_generation_as_official_protocol(
        tmp_path, monkeypatch):
    import script.collect_qwen35_validation_results as module

    monkeypatch.setattr(module, 'REPO', tmp_path)
    monkeypatch.setattr(module, 'latest_path', lambda pattern: None)
    monkeypatch.setattr(module, 'mmlu_prox_summary_score', lambda root: None)
    monkeypatch.setattr(
        module, 'final_answer_counts',
        lambda summary, filename: {
            'blank_final_answers': 1,
            'nonempty_final_answers': 1,
        })

    shared_hmmt_2025 = tmp_path / 'outputs/hmmt2025/summary.csv'
    evidence_by_root = {
        'outputs/hmmt_feb_2026_full_chat_qwen3.5_2b_thinking_stream33':
        tmp_path / 'outputs/hmmt2026/summary.csv',
        'outputs/livecodebench_v6_codegen_qwen3.5_2b_thinking_stream175':
        tmp_path / 'outputs/livecodebench/summary.csv',
    }

    def fake_summary_score(root, datasets, prediction_file=None,
                           expected_rows=None, run_id=None):
        if root in evidence_by_root:
            return 12.5, evidence_by_root[root]
        if root.endswith('hmmt_2025_full_chat_qwen3.5_2b_thinking_stream60'):
            return 12.5, shared_hmmt_2025
        return None

    monkeypatch.setattr(module, 'summary_score', fake_summary_score)
    state_path = tmp_path / 'state.json'
    keys = ('hmmt_feb_2026', 'livecodebench_v6', 'hmmt_feb_2025',
            'hmmt_nov_2025')
    state_path.write_text(
        json.dumps({'benchmarks': {key: {} for key in keys}}),
        encoding='utf-8')

    result = module.collect(state_path)

    for key in keys:
        assert result['benchmarks'][key]['result_type'] == (
            'official_dataset_scorer_32k_cap')


def test_collector_adds_longbench_and_aa_lcr_official_results(
        tmp_path, monkeypatch):
    import script.collect_qwen35_validation_results as module

    monkeypatch.setattr(module, 'REPO', tmp_path)
    monkeypatch.setattr(module, 'latest_path', lambda pattern: None)
    monkeypatch.setattr(module, 'mmlu_prox_summary_score', lambda root: None)

    fixtures = (
        ('outputs/longbenchv2_full_qwen3.5_2b_thinking_262144_32768_'
         'token_mid_stream16_batch16_idle600', '20260726_173404',
         'LongBenchv2',
         'LongBenchv2.json', 503, '38.7'),
        ('outputs/aa_lcr_full_qwen3.5_2b_thinking_262144_32768_'
         'stream8_batch16', '20260726_163358', 'aa_lcr', 'aa_lcr.json', 100,
         '25.6'),
    )
    for root, run_id, dataset, prediction_name, count, score in fixtures:
        run = tmp_path / root / run_id
        summary = run / 'summary' / f'summary_{run_id}.csv'
        summary.parent.mkdir(parents=True)
        summary.write_text(
            f'dataset,metric,model\n{dataset},accuracy,{score}\n',
            encoding='utf-8')
        prediction = run / 'predictions/Qwen3.5-2B-chat' / prediction_name
        prediction.parent.mkdir(parents=True)
        answer = ('The correct answer is (A)'
                  if dataset == 'LongBenchv2' else 'answer')
        prediction.write_text(
            json.dumps({str(index): {'prediction': answer}
                        for index in range(count)}),
            encoding='utf-8')

    state_path = tmp_path / 'state.json'
    state_path.write_text(json.dumps({'benchmarks': {
        'longbench_v2': {},
        'aa_lcr': {},
    }}), encoding='utf-8')

    result = module.collect(state_path)

    assert result['benchmarks']['longbench_v2']['score'] == 38.7
    assert result['benchmarks']['longbench_v2']['total'] == 503
    assert result['benchmarks']['longbench_v2'][
        'parseable_final_answers'] == 503
    assert result['benchmarks']['aa_lcr']['score'] == 25.6
    assert result['benchmarks']['aa_lcr']['result_type'] == (
        'official_equality_judge_qwen3_235b_non_thinking')
