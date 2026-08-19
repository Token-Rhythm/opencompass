from collections import Counter

from opencompass.datasets.gpqa import (GPQADataset,
                                       GPQASimpleEvalDataset,
                                       GPQA_Simple_Eval_postprocess)
from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import (
    gpqa_datasets,
)


def test_gpqa_uses_rolling_parallel_inference():
    inferencer = gpqa_datasets[0]['infer_cfg']['inferencer']

    assert inferencer['type'].__name__ == 'ParallelGenInferencer'
    assert inferencer['save_every'] == 1


def test_gpqa_simple_eval_uses_four_valid_permutations():
    path = './data/gpqa/'
    original = GPQADataset.load(path, 'gpqa_diamond.csv')
    repeated = GPQASimpleEvalDataset.load(path, 'gpqa_diamond.csv')

    assert len(original) == 198
    assert len(repeated) == 4 * len(original)
    assert set(Counter(row['question'] for row in repeated).values()) == {4}

    for row in repeated:
        answer_index = ord(row['answer']) - ord('A')
        assert row['permutation'][answer_index] == 0
        assert row[row['answer']] == row['options'][answer_index]


def test_gpqa_simple_eval_postprocess_matches_official_answer_pattern():
    assert GPQA_Simple_Eval_postprocess('Answer: A') == 'A'
    assert GPQA_Simple_Eval_postprocess('ANSWER:\t$B$') == 'B'
    assert GPQA_Simple_Eval_postprocess('reasoning only') is None
