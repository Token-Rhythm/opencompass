import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'script/agent_eval_catalog_sync.py'
SPEC = importlib.util.spec_from_file_location('agent_eval_catalog_sync', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


EXISTING = {
    'mmlu-pro': ('bmk_mmlu_pro', 'cap_broad_knowledge', '2024', 'test'),
    'c-eval': ('bmk_ceval', 'cap_broad_knowledge', '2023', 'test'),
    'supergpqa': ('bmk_supergpqa', 'cap_broad_knowledge', '2025', 'test'),
    'mmmlu': ('bmk_mmmlu', 'cap_broad_knowledge', '2024', 'test'),
    'gpqa-diamond': (
        'bmk_gpqa', 'cap_scientific_reasoning', 'original-2023', 'diamond'),
    'live-code-bench': ('bmk_lcb', 'cap_coding', '2024', 'test'),
    'mmlu-redux-2-0': ('bmk_redux', 'cap_broad_knowledge', '2.0', 'test'),
    'include': ('bmk_include', 'cap_broad_knowledge', '2024', 'test'),
}


def production_catalog():
    benchmarks = []
    versions = []
    for slug, (benchmark_id, capability, version, split) in EXISTING.items():
        benchmarks.append({
            'id': benchmark_id,
            'slug': slug,
            'name': slug,
            'primaryCapabilityId': capability,
        })
        versions.append({
            'id': f'bmv_{benchmark_id}',
            'benchmarkId': benchmark_id,
            'version': version,
            'split': split,
            'sampleCount': None,
            'datasetRevision': None,
            'evaluatorRevision': None,
            'protocolUrl': None,
            'checksum': None,
        })
    return {
        'source': 'd1',
        'generatedAt': '2026-07-28T00:00:00Z',
        'data': {
            'benchmarks': benchmarks,
            'benchmarkVersions': versions,
        },
    }


def test_plan_contains_only_exact_missing_versions_and_families():
    proposals = MODULE.build_proposals(production_catalog())
    assert len(proposals) == 17
    creates = {
        item['benchmark']['slug'] for item in proposals
        if item['operation'] == 'create'
    }
    assert creates == {
        'ifeval', 'ifbench', 'longbench-v2', 'aime', 'hmmt', 'mmlu-prox',
        'global-piqa', 'multichallenge', 'aa-lcr',
    }
    updates = {
        item['_slug'] for item in proposals if item['operation'] == 'update'
    }
    assert updates == set(EXISTING)
    by_name = {
        MODULE.request_for(item)['submission']['benchmark_name']: item
        for item in proposals
    }
    assert by_name['GPQA-Diamond']['benchmark']['versions'][0][
        'sample_count'] == 792
    assert by_name['MMMLU']['benchmark']['versions'][0][
        'sample_count'] == 196588
    assert by_name['Global PIQA']['benchmark']['versions'][0][
        'sample_count'] == 67818
    multichallenge = by_name['MultiChallenge']['benchmark']['versions'][0]
    assert multichallenge['sample_count'] == 273
    assert multichallenge['dataset_revision'] == \
        '5ccefcca6a39020d66c1383c4e6a809cb07afa33'
    assert not MODULE.DEFERRED


def test_exact_existing_versions_are_idempotently_skipped():
    catalog = production_catalog()
    for desired in MODULE.FAMILIES:
        benchmark = next((item for item in catalog['data']['benchmarks']
                          if item['slug'] == desired['slug']), None)
        if benchmark is None:
            benchmark = {
                'id': f'bmk_{desired["slug"]}',
                'slug': desired['slug'],
                'name': desired['name'],
                'primaryCapabilityId': desired['primary_capability_id'],
            }
            catalog['data']['benchmarks'].append(benchmark)
        for item in desired['versions']:
            catalog['data']['benchmarkVersions'].append({
                'id': f'bmv_{desired["slug"]}_{item["version"]}',
                'benchmarkId': benchmark['id'],
                'version': item['version'],
                'split': item['split'],
                'sampleCount': item['sample_count'],
                'datasetRevision': item.get('dataset_revision'),
                'evaluatorRevision': item.get('evaluator_revision'),
                'protocolUrl': item.get('protocol_url'),
                'checksum': item.get('checksum'),
            })
    assert MODULE.build_proposals(catalog) == []


def test_immutable_version_collision_fails_closed():
    catalog = production_catalog()
    desired = MODULE.FAMILIES[0]['versions'][0]
    catalog['data']['benchmarkVersions'].append({
        'id': 'bmv_collision',
        'benchmarkId': 'bmk_mmlu_pro',
        'version': desired['version'],
        'split': desired['split'],
        'sampleCount': 1,
        'datasetRevision': desired['dataset_revision'],
        'evaluatorRevision': desired['evaluator_revision'],
        'protocolUrl': desired['protocol_url'],
        'checksum': desired.get('checksum'),
    })
    with pytest.raises(MODULE.CatalogSyncError, match='immutable version collision'):
        MODULE.build_proposals(catalog)


def test_plan_never_contains_a_token(tmp_path, monkeypatch):
    secret = 'aeh_pat_secret_must_not_be_persisted'
    monkeypatch.setenv(MODULE.TOKEN_ENV, secret)
    catalog = tmp_path / 'catalog.json'
    output = tmp_path / 'plan.json'
    catalog.write_text(json.dumps(production_catalog()), encoding='utf-8')
    assert MODULE.main([
        '--catalog-file', str(catalog), '--output', str(output),
    ]) == 0
    assert secret not in output.read_text(encoding='utf-8')
