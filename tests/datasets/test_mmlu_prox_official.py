from opencompass.configs.datasets.mmlu_prox.mmlu_prox_0shot_cot_gen import (
    mmlu_prox_0shot_datasets,
)
from opencompass.configs.datasets.mmlu_prox.mmlu_prox_5shot_cot_gen import (
    mmlu_prox_5shot_datasets,
)
from opencompass.configs.datasets.mmlu_prox.mmlu_prox_lite_0shot_cot_gen import (
    mmlu_prox_lite_0shot_datasets,
)
from opencompass.configs.datasets.mmlu_prox.mmlu_prox_lite_5shot_cot_gen import (
    mmlu_prox_lite_5shot_datasets,
)
from opencompass.configs.summarizers.groups.mmlu_prox import (
    mmlu_prox_summary_groups,
)
from opencompass.configs.summarizers.mmlu_prox import summarizer
from opencompass.datasets.mmlu_prox import MMLUProXEvaluator
from opencompass.openicl.icl_inferencer import ParallelGenInferencer


def test_mmlu_prox_configs_do_not_stop_on_question_markers():
    variants = (
        mmlu_prox_5shot_datasets,
        mmlu_prox_0shot_datasets,
        mmlu_prox_lite_5shot_datasets,
        mmlu_prox_lite_0shot_datasets,
    )
    for datasets in variants:
        assert 'stopping_criteria' not in datasets[0]['infer_cfg']['inferencer']


def test_mmlu_prox_has_29_language_overall_group():
    overall = next(group for group in mmlu_prox_summary_groups
                   if group['name'] == 'mmlu_prox')
    assert len(overall['subsets']) == 29 * 14
    assert len(overall['weights']) == 29 * 14
    assert summarizer['dataset_abbrs'][-1] == 'mmlu_prox'


def test_mmlu_prox_full_5shot_uses_rolling_checkpoint_inference():
    inferencer = mmlu_prox_5shot_datasets[0]['infer_cfg']['inferencer']
    assert inferencer['type'] is ParallelGenInferencer
    assert inferencer['save_every'] == 1


def test_mmlu_prox_evaluator_rejects_length_mismatch():
    result = MMLUProXEvaluator().score([], ['A'], [])
    assert result == {
        'error': 'predictions, references, and test_set have different lengths'
    }
