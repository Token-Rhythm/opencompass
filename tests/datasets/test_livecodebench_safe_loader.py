from datasets import Dataset
from mmengine.config import Config

from opencompass.datasets.livecodebench import evaluator, livecodebench
from opencompass.openicl.icl_inferencer import ParallelGenInferencer


def test_codegen_loader_uses_safe_json_file(monkeypatch):
    source = Dataset.from_list([{
        'question_title': 'title',
        'question_content': 'problem',
        'question_id': 'id',
        'starter_code': '',
        'public_test_cases': '[{"input": "1", "output": "1"}]',
        'private_test_cases': '[{"input": "2", "output": "2"}]',
        'metadata': '{}',
        'contest_date': '2025-01-01',
    }])
    called = {}

    def fake_load_dataset(path, **kwargs):
        called.update(path=path, **kwargs)
        return source

    monkeypatch.setattr(livecodebench, 'load_dataset', fake_load_dataset)
    loaded = livecodebench.LCBCodeGenerationDataset.load(
        data_file='https://example.test/pinned.jsonl')

    assert called == {
        'path': 'json',
        'data_files': {'test': 'https://example.test/pinned.jsonl'},
        'split': 'test',
    }
    assert len(loaded['test']) == 1
    assert loaded['test'][0]['question_id'] == 'id'
    assert 'evaluation_sample' in loaded['test'].column_names


def test_v6_config_uses_official_code_generation_system_prompt():
    config = Config.fromfile(
        'opencompass/configs/datasets/livecodebench/'
        'livecodebench_v6_codegen.py')
    template = config.livecodebench_v6_codegen_datasets[0][
        'infer_cfg']['prompt_template']['template']

    assert template['begin'] == [{
        'role': 'SYSTEM',
        'fallback_role': 'HUMAN',
        'prompt': (
            'You are an expert Python programmer. You will be given a '
            'question (problem specification) and will generate a correct '
            'Python program that matches the specification and passes all '
            'tests.'),
    }]
    assert template['round'][0]['role'] == 'HUMAN'
    inferencer = config.livecodebench_v6_codegen_datasets[0][
        'infer_cfg']['inferencer']
    assert inferencer == {
        'type': ParallelGenInferencer,
        'save_every': 1,
    }


def test_codegen_global_timeout_returns_failure_without_metadata_crash(
        monkeypatch):
    class FakeManager:
        @staticmethod
        def list():
            return []

    class FakeProcess:
        killed = False
        joins = 0

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            self.joins += 1

        def is_alive(self):
            return True

        def kill(self):
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr(evaluator.multiprocessing, 'Manager', FakeManager)
    monkeypatch.setattr(
        evaluator.multiprocessing, 'Process', lambda *args, **kwargs: process)

    result, metadata = evaluator.codegen_check_correctness(
        {'input_output': '{"inputs": ["one", "two"]}'},
        'print(1)',
        timeout=1,
        debug=False,
    )

    assert result == [-1, -1]
    assert metadata == {}
    assert process.killed is True
    assert process.joins == 2
