"""Official MultiChallenge multi-turn conversation adapter."""

import re
from collections import defaultdict

from datasets import DatasetDict, load_dataset

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET

from .base import BaseDataset

MULTICHALLENGE_REVISION = '5ccefcca6a39020d66c1383c4e6a809cb07afa33'
MULTICHALLENGE_URL = (
    'https://raw.githubusercontent.com/ekwinox117/multi-challenge/'
    f'{MULTICHALLENGE_REVISION}/data/benchmark_questions.jsonl')


def _format_multichallenge(row):
    # ChatInferencer(infer_mode='last') expects a final assistant turn holding
    # the gold value. It removes that turn before model inference.
    dialogue = list(row['CONVERSATION']) + [{
        'role': 'assistant',
        'content': row['PASS_CRITERIA'],
    }]
    return {
        'question_id': row['QUESTION_ID'],
        'axis': row['AXIS'],
        'dialogue': dialogue,
        'target_question': row['TARGET_QUESTION'],
        'pass_criteria': row['PASS_CRITERIA'],
    }


@LOAD_DATASET.register_module()
class MultiChallengeDataset(BaseDataset):
    """Load the data file from a pinned official GitHub commit."""

    @staticmethod
    def load(path=MULTICHALLENGE_URL):
        split = load_dataset('json', data_files=path, split='train')
        split = split.map(_format_multichallenge,
                          remove_columns=split.column_names)
        return DatasetDict({'test': split})


def _extract_multichallenge_verdict(judgement):
    matches = re.findall(r'\b(YES|NO)\b', str(judgement).upper())
    return matches[-1] if matches else None


def _score_multichallenge(judgements, rows):
    axis_results = defaultdict(list)
    details = []
    for judgement, row in zip(judgements, rows):
        verdict = _extract_multichallenge_verdict(judgement)
        passed = verdict == row['pass_criteria']
        axis_results[row['axis']].append(float(passed))
        details.append({
            'question_id': row['question_id'],
            'axis': row['axis'],
            'judge_response': judgement,
            'verdict': verdict,
            'pass_criteria': row['pass_criteria'],
            'passed': passed,
        })
    axis_scores = {
        axis: 100 * sum(values) / len(values)
        for axis, values in axis_results.items()
    }
    result = {
        'overall_score':
        (sum(axis_scores.values()) / len(axis_scores) if axis_scores else 0),
        'details':
        details,
    }
    result.update({
        f'axis/{axis}': score
        for axis, score in axis_scores.items()
    })
    return result


@DICT_POSTPROCESSORS.register_module()
def multichallenge_postprocess(output, output_path, dataset):
    rows = dataset.reader.dataset['test']
    ordered = sorted(output.items(), key=lambda item: int(item[0]))
    judgements = [value['prediction'] for _, value in ordered]
    selected_rows = [rows[int(index)] for index, _ in ordered]
    return _score_multichallenge(judgements, selected_rows)
