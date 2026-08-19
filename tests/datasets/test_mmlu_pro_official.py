from opencompass.datasets.mmlu_pro import (_format_fewshot,
                                           _format_question,
                                           MMLUProEvaluator)
from opencompass.configs.datasets.mmlu_pro.mmlu_pro_5shot_cot_gen import \
    mmlu_pro_5shot_cot_datasets


def _row():
    return {
        'question': 'What is 1 + 1?',
        'options': [' 1 ', ' 2 ', 'N/A'],
        'answer': 'B',
        'cot_content': (
            "A: Let's think step by step. One plus one is two. "
            'The answer is (B).'),
    }


def test_mmlu_pro_question_matches_current_reference_format():
    assert _format_question(_row()) == (
        'Question:\nWhat is 1 + 1?\nOptions:\nA. 1\nB. 2\n'
        "Answer: Let's think step by step.")


def test_mmlu_pro_fewshot_keeps_cot_and_answer_suffix():
    prompt = _format_fewshot(_row())
    assert "Answer: Let's think step by step. One plus one is two." in prompt
    assert prompt.endswith('The answer is (B).\n\n')


def test_mmlu_pro_evaluator_uses_official_answer_is_suffix():
    result = MMLUProEvaluator().score(
        ['Reasoning. The answer is (B).', 'ANSWER: C'], ['B', 'C'])
    assert result['accuracy'] == 50
    assert result['details'][0]['parsed'] == 'B'
    assert result['details'][1]['parsed'] == ''


def test_mmlu_pro_thinking_task_has_no_question_stop_sequence():
    inferencer = mmlu_pro_5shot_cot_datasets[0]['infer_cfg']['inferencer']
    assert 'stopping_criteria' not in inferencer
