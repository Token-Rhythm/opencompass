import json
import os
import subprocess
import sys
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / 'script/opencompass_agent_eval_adapter.py'
MODULE_SPEC = importlib.util.spec_from_file_location('agent_eval_adapter', ADAPTER)
ADAPTER_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(ADAPTER_MODULE)


def catalog_envelope():
    return {
        'source': 'd1',
        'data': {
            'systems': [{
                'id': 'sys_test_model',
                'name': 'test-model',
                'displayName': 'test-model',
                'scope': 'posttrained',
                'modelName': 'test-model',
                'modelVersion': 'revision-1',
            }],
            'benchmarks': [{
                'id': 'bmk_mmlu_pro',
                'slug': 'mmlu-pro',
                'name': 'MMLU-Pro',
                'displayName': 'MMLU-Pro',
            }],
            'benchmarkVersions': [{
                'id': 'bmv_mmlu_pro_2024',
                'benchmarkId': 'bmk_mmlu_pro',
                'version': '2024-5shot-cot',
                'split': 'test',
            }],
            'metrics': [{
                'id': 'met_accuracy',
                'key': 'accuracy',
                'displayName': 'Accuracy',
                'unit': 'percent',
                'scorerKey': 'accuracy',
            }],
        },
    }


def make_run(tmp_path):
    run_dir = tmp_path / 'mmlu_pro' / 'run-001'
    summary = run_dir / 'summary/summary_20260728_120000.csv'
    summary.parent.mkdir(parents=True)
    summary.write_text(
        'dataset,version,metric,mode,test-model\n'
        'mmlu_pro,abc123,accuracy,gen,61.25\n'
        'mmlu_pro_biology,abc123,accuracy,gen,70.00\n',
        encoding='utf-8')
    config = run_dir / 'configs/20260728_120000.py'
    config.parent.mkdir()
    config.write_text('models = []\n', encoding='utf-8')
    summary_timestamp = summary.stat().st_mtime
    os.utime(config, (summary_timestamp - 1, summary_timestamp - 1))
    predictions = run_dir / 'predictions/test-model-chat/mmlu_pro_biology.json'
    predictions.parent.mkdir(parents=True)
    predictions.write_text(json.dumps({'0': {}, '1': {}}), encoding='utf-8')
    return run_dir


def write_sampling_config(run_dir):
    config = next((run_dir / 'configs').glob('*.py'))
    config.write_text(
        """models = [
    dict(
        abbr='test-model-completions',
        path='test-model',
        summarizer_abbr='test-model',
        generation_endpoint='completions',
        temperature=0.2,
        max_out_len=4096,
        extra_body=dict(top_p=0.8),
    ),
    dict(
        abbr='test-model-chat',
        path='test-model',
        summarizer_abbr='test-model',
        generation_endpoint='chat',
        temperature=1.0,
        max_out_len=8192,
        extra_body=dict(
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            chat_template_kwargs=dict(enable_thinking=False),
            api_key='must-not-be-persisted',
        ),
        openai_extra_kwargs=dict(seed=7),
    ),
]
datasets = [
    dict(infer_cfg=dict(inferencer=dict(max_out_len=16384))),
]
""",
        encoding='utf-8')
    summary = next((run_dir / 'summary').glob('summary_*.csv'))
    summary_timestamp = summary.stat().st_mtime
    os.utime(config, (summary_timestamp - 1, summary_timestamp - 1))
    return config


def test_payload_only_extracts_primary_score_and_lineage(tmp_path):
    run_dir = make_run(tmp_path)
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps(catalog_envelope()), encoding='utf-8')
    result = subprocess.run([
        sys.executable, str(ADAPTER), '--benchmark', 'mmlu_pro',
        '--run-dir', str(run_dir), '--model', 'test-model',
        '--platform-url', 'https://47.88.93.207',
        '--evidence-url',
        'https://artifacts.example/{benchmark}/{run_id}/{summary_sha256}',
        '--evidence-root', str(tmp_path / 'evidence'),
        '--model-checkpoint-id', 'mcp_test_checkpoint',
        '--mode', 'review', '--catalog-file', str(catalog), '--payload-only',
    ], cwd=ROOT, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(
        (run_dir / 'agent_eval/mmlu_pro.json').read_text(encoding='utf-8'))
    submission = receipt['payload']['submission']
    assert submission['metric_value'] == 61.25
    assert submission['metric_unit'] == 'percent'
    assert submission['sample_count'] == 2
    assert submission['canonical_mapping'] == {
        'system_id': 'sys_test_model',
        'benchmark_version_id': 'bmv_mmlu_pro_2024',
        'metric_id': 'met_accuracy',
        'scorer_key': 'accuracy',
        'model_checkpoint_id': 'mcp_test_checkpoint',
    }
    published = (
        tmp_path / 'evidence/mmlu_pro/run-001/summary_20260728_120000.csv')
    assert published.read_bytes() == next(
        (run_dir / 'summary').glob('summary_*.csv')).read_bytes()
    assert submission['canonical_result']['dataset_snapshot'] == \
        'opencompass:abc123'
    assert submission['canonical_result']['config_hash'].startswith('sha256:')
    assert submission['canonical_evidence']['content_hash'].startswith('sha256:')


def test_payload_records_effective_generation_config_without_secrets(tmp_path):
    run_dir = make_run(tmp_path)
    write_sampling_config(run_dir)
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps(catalog_envelope()), encoding='utf-8')
    result = subprocess.run([
        sys.executable, str(ADAPTER), '--benchmark', 'mmlu_pro',
        '--run-dir', str(run_dir), '--model', 'test-model',
        '--platform-url', 'https://47.88.93.207',
        '--evidence-url', 'https://artifacts.example/mmlu-pro/run-001',
        '--mode', 'review', '--catalog-file', str(catalog), '--payload-only',
    ], cwd=ROOT, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    receipt_text = (run_dir / 'agent_eval/mmlu_pro.json').read_text(
        encoding='utf-8')
    assert 'must-not-be-persisted' not in receipt_text
    canonical_result = json.loads(receipt_text)['payload']['submission'][
        'canonical_result']
    assert canonical_result['temperature'] == 1.0
    assert canonical_result['top_p'] == 0.95
    assert canonical_result['top_k'] == 20
    assert canonical_result['min_p'] == 0.0
    assert canonical_result['seed'] == 7
    assert canonical_result['max_output_tokens'] == 16384
    assert canonical_result['thinking_enabled'] is False
    assert canonical_result['sampling_config'] == {
        'chat_template_kwargs': {'enable_thinking': False},
        'presence_penalty': 1.5,
        'repetition_penalty': 1.0,
    }


def test_include_does_not_report_unused_generation_config(tmp_path):
    run_dir = make_run(tmp_path)
    config = write_sampling_config(run_dir)

    extracted = ADAPTER_MODULE.generation_config(
        config, 'test-model', 'include')

    assert extracted == {}


def test_multichallenge_uses_axis_macro_score_and_pinned_catalog_version():
    spec = ADAPTER_MODULE.BENCHMARK_SPECS['multichallenge']

    assert spec['summary_datasets'] == ['multichallenge']
    assert spec['summary_metrics'] == ['overall_score']
    assert spec['benchmark_version'] == '2025-5ccefcca-gpt4o-judge'
    assert spec['split'] == 'test'
    assert spec['metric_aliases'] == ['accuracy']


def test_live_flow_catalog_dry_run_then_apply_and_never_persists_token(
        tmp_path, monkeypatch, capsys):
    run_dir = make_run(tmp_path)
    calls = []
    secret = 'aeh_pat_test_secret_that_must_not_be_written'

    def fake_request(platform_url, path, token, **kwargs):
        calls.append((kwargs.get('method', 'GET'), path, token,
                      kwargs.get('body')))
        if path == '/api/submissions/catalog':
            return 200, catalog_envelope()
        document = kwargs['body']
        if document['dry_run']:
            return 200, {
                'ok': True,
                'dry_run': True,
                'normalized': {
                    'idempotencyKey': document['idempotency_key'],
                },
            }
        return 201, {
            'ok': True,
            'dry_run': False,
            'replayed': False,
            'submission_id': 'sub_test',
            'review_id': 'rev_test',
            'status': 'pending_review',
        }

    monkeypatch.setenv('AGENT_EVAL_SUBMISSION_PAT', secret)
    monkeypatch.setattr(ADAPTER_MODULE, 'request_json', fake_request)
    result = ADAPTER_MODULE.main([
        '--benchmark', 'mmlu_pro', '--run-dir', str(run_dir),
        '--model', 'test-model', '--platform-url', 'http://127.0.0.1:3000',
        '--evidence-url', 'https://artifacts.example/mmlu-pro/run-001',
        '--mode', 'review',
    ])

    assert result == 0
    assert [call[:2] for call in calls] == [
        ('GET', '/api/submissions/catalog'),
        ('POST', '/api/submissions'),
        ('POST', '/api/submissions'),
    ]
    assert all(call[2] == secret for call in calls)
    assert calls[1][3]['dry_run'] is True
    assert calls[2][3]['dry_run'] is False
    assert calls[1][3]['idempotency_key'] == calls[2][3]['idempotency_key']
    receipt_text = (run_dir / 'agent_eval/mmlu_pro.json').read_text()
    assert secret not in receipt_text
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    receipt = json.loads(receipt_text)
    assert receipt['status'] == 'pending_review'
    assert receipt['submission_id'] == 'sub_test'


def test_direct_ingest_uses_separate_admin_token_and_skips_reviews(
        tmp_path, monkeypatch):
    run_dir = make_run(tmp_path)
    write_sampling_config(run_dir)
    calls = []
    catalog_secret = 'aeh_pat_catalog_lookup_secret'
    ingest_secret = 'ingest_admin_secret'

    def fake_request(platform_url, path, token, **kwargs):
        calls.append((kwargs.get('method', 'GET'), path, token,
                      kwargs.get('body')))
        if path == '/api/submissions/catalog':
            return 200, catalog_envelope()
        document = kwargs['body']
        if document['dry_run']:
            return 200, {
                'ok': True,
                'dry_run': True,
                'validation_mode': 'schema_only',
                'normalized': {'idempotencyKey': document['idempotency_key']},
            }
        return 201, {
            'ok': True,
            'dry_run': False,
            'idempotency_key': document['idempotency_key'],
            'replayed': False,
            'created': {'evidence': 1, 'runs': 1, 'samples': 0, 'results': 1},
            'ids': {
                'evidence': ['ev_test'],
                'runs': ['run_test'],
                'samples': [],
                'results': ['res_test'],
            },
        }

    monkeypatch.setenv('AGENT_EVAL_SUBMISSION_PAT', catalog_secret)
    monkeypatch.setenv('INGEST_ADMIN_TOKEN', ingest_secret)
    monkeypatch.setattr(ADAPTER_MODULE, 'request_json', fake_request)
    result = ADAPTER_MODULE.main([
        '--benchmark', 'mmlu_pro',
        '--run-dir', str(run_dir), '--model', 'test-model',
        '--platform-url', 'http://127.0.0.1:3000',
        '--evidence-url', 'https://artifacts.example/mmlu-pro/run-001',
    ])

    assert result == 0
    assert [call[:2] for call in calls] == [
        ('GET', '/api/submissions/catalog'),
        ('POST', '/api/ingest'),
        ('POST', '/api/ingest'),
    ]
    assert calls[0][2] == catalog_secret
    assert calls[1][2] == ingest_secret
    assert calls[2][2] == ingest_secret
    dry_payload = calls[1][3]
    assert dry_payload['runs'][0]['temperature'] == 1.0
    assert dry_payload['runs'][0]['top_p'] == 0.95
    assert dry_payload['runs'][0]['max_output_tokens'] == 16384
    assert dry_payload['results'][0]['value'] == 61.25
    receipt_path = run_dir / 'agent_eval/mmlu_pro.ingest.json'
    receipt_text = receipt_path.read_text(encoding='utf-8')
    assert catalog_secret not in receipt_text
    assert ingest_secret not in receipt_text
    receipt = json.loads(receipt_text)
    assert receipt['status'] == 'ingested'
    assert receipt['created'] == {
        'evidence': 1, 'runs': 1, 'samples': 0, 'results': 1,
    }
    assert receipt['ids']['runs'] == ['run_test']


def test_ifbench_metric_mapping_is_unambiguous():
    envelope = catalog_envelope()
    envelope['data']['benchmarks'][0].update({
        'id': 'bmk_ifbench',
        'slug': 'if-bench',
        'name': 'IFBench',
        'displayName': 'IFBench',
    })
    envelope['data']['benchmarkVersions'][0].update({
        'id': 'bmv_ifbench_2025',
        'benchmarkId': 'bmk_ifbench',
        'version': '2025-2e8a48de-zero-shot',
        'split': 'train',
    })
    envelope['data']['metrics'] = [
        {
            'id': 'met_accuracy_percent',
            'key': 'accuracy',
            'displayName': 'Accuracy',
            'unit': 'percent',
        },
        {
            'id': 'met_prompt_level_loose_accuracy_percent',
            'key': 'prompt_level_loose_accuracy',
            'displayName': 'Prompt-level-loose-accuracy',
            'unit': 'percent',
        },
    ]

    resolved = ADAPTER_MODULE.resolve_catalog(
        envelope, ADAPTER_MODULE.BENCHMARK_SPECS['ifbench'], 'test-model',
        'posttrained', '62b1bb')

    assert resolved['metric']['id'] == \
        'met_prompt_level_loose_accuracy_percent'


def test_ingest_receipt_rejects_mismatched_idempotency_key():
    try:
        ADAPTER_MODULE.validate_ingest_receipt({
            'ok': True,
            'dry_run': False,
            'idempotency_key': 'different-key',
            'replayed': True,
            'created': {
                'evidence': 0, 'runs': 0, 'samples': 0, 'results': 0,
            },
            'ids': {
                'evidence': ['ev_test'],
                'runs': ['run_test'],
                'samples': [],
                'results': ['res_test'],
            },
        }, dry_run=False, idempotency_key_value='expected-key')
    except ADAPTER_MODULE.AdapterError as error:
        assert 'different idempotency key' in str(error)
    else:
        raise AssertionError('mismatched idempotency key was accepted')


def test_missing_pinned_catalog_version_is_rejected(tmp_path):
    run_dir = make_run(tmp_path)
    envelope = catalog_envelope()
    envelope['data']['benchmarkVersions'][0]['version'] = '2025-01'
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps(envelope), encoding='utf-8')
    result = subprocess.run([
        sys.executable, str(ADAPTER), '--benchmark', 'mmlu_pro',
        '--run-dir', str(run_dir), '--model', 'test-model',
        '--platform-url', 'https://47.88.93.207',
        '--evidence-url', 'https://artifacts.example/run-001',
        '--catalog-file', str(catalog), '--payload-only',
    ], cwd=ROOT, text=True, capture_output=True)

    assert result.returncode == 2
    assert 'no exact match' in result.stderr


def test_ifeval_prefers_specific_metric_over_generic_accuracy():
    envelope = catalog_envelope()
    envelope['data']['benchmarks'] = [{
        'id': 'bmk_ifeval',
        'slug': 'if-eval',
        'name': 'IFEval',
        'displayName': 'IFEval',
    }]
    envelope['data']['benchmarkVersions'] = [{
        'id': 'bmv_ifeval_2023',
        'benchmarkId': 'bmk_ifeval',
        'version': '2023-966cd895-zero-shot',
        'split': 'train',
    }]
    envelope['data']['metrics'] = [
        {
            'id': 'met_accuracy_percent',
            'key': 'accuracy',
            'displayName': 'Accuracy',
            'unit': 'percent',
            'scorerKey': 'accuracy',
        },
        {
            'id': 'met_prompt_level_strict_accuracy_percent',
            'key': 'prompt_level_strict_accuracy',
            'displayName': 'Prompt-level-strict-accuracy',
            'unit': 'percent',
            'scorerKey': 'prompt_level_strict_accuracy',
        },
    ]

    resolved = ADAPTER_MODULE.resolve_catalog(
        envelope,
        ADAPTER_MODULE.BENCHMARK_SPECS['ifeval'],
        'test-model',
        'posttrained',
        '9a895f',
    )

    assert resolved['metric']['id'] == (
        'met_prompt_level_strict_accuracy_percent'
    )
