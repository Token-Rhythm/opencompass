from opencompass.datasets.hmmt_2025 import MathArenaEvaluator
from opencompass.configs.datasets.hmmt_2025.hmmt_2025_matharena_gen import (
    hmmt_2025_datasets,
)
from opencompass.openicl.icl_inferencer import ParallelGenInferencer


def test_matharena_evaluator_rejects_length_mismatch_before_scoring():
    result = MathArenaEvaluator().score([], ['1'])
    assert result == {
        'error': 'predictions and references have different lengths'
    }


def test_hmmt_2025_uses_rolling_checkpoint_inference():
    for dataset in hmmt_2025_datasets:
        inferencer = dataset['infer_cfg']['inferencer']
        assert inferencer['type'] is ParallelGenInferencer
        assert inferencer['save_every'] == 1
