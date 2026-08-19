"""MMLU-Pro dataset adapters and answer evaluation."""

import re

from datasets import load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset

MMLU_PRO_PATH = 'TIGER-Lab/MMLU-Pro'
MMLU_PRO_REVISION = 'b189ec765aa7ed75c8acfea42df31fdae71f97be'
CHOICES = list('ABCDEFGHIJKLMNOP')


def _format_question(item):
    """Render a question in the current lm-eval MMLU-Pro format."""
    lines = ['Question:', item['question'], 'Options:']
    for index, option in enumerate(item['options'][:10]):
        if option == 'N/A':
            continue
        lines.append(f'{CHOICES[index]}. {option.strip()}')
    lines.append("Answer: Let's think step by step.")
    return '\n'.join(lines)


def _format_fewshot(item):
    question = _format_question(item)
    cot = item['cot_content'].strip()
    prefix = "A: Let's think step by step."
    if cot.startswith(prefix):
        cot = cot[len(prefix):].lstrip()
    return f'{question} {cot}\n\n'


def _parse(item):
    s = ''
    item['answer_string'] = ''
    for i, opt in enumerate(item['options']):
        if opt == 'N/A':
            continue
        option = '{}. {}\n'.format(CHOICES[i], opt.strip())
        s += option
        if item['answer'] == CHOICES[i]:
            item['answer_string'] = option

    item['options_str'] = s.strip()
    # Keep legacy fields intact while exposing fully rendered official prompt
    # fragments for the 5-shot CoT chat task.
    item['question_prompt'] = _format_question(item)
    item['fewshot_prompt'] = _format_fewshot(item)
    return item


@LOAD_DATASET.register_module()
class MMLUProDataset(BaseDataset):

    @staticmethod
    def load(path: str = MMLU_PRO_PATH,
             category: str = None,
             revision: str = MMLU_PRO_REVISION):
        # The canonical dataset is a Hub ID. Preserve get_data_path for the
        # legacy ``opencompass/mmlu_pro`` local archive configs.
        if path != MMLU_PRO_PATH:
            path = get_data_path(path)
        load_kwargs = {'revision': revision} if revision else {}
        mmlu_pro = load_dataset(path, **load_kwargs)
        if category is not None:
            mmlu_pro = mmlu_pro.filter(lambda x: x['category'] == category)
        mmlu_pro = mmlu_pro.map(_parse)
        return mmlu_pro


class MMLUProEvaluator(BaseEvaluator):
    """Extract the official ``the answer is (X)`` suffix and score EM."""

    answer_pattern = re.compile(r'answer is \(?([A-J])\)?', re.IGNORECASE)

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        correct = 0
        details = []
        for prediction, reference in zip(predictions, references):
            matches = self.answer_pattern.findall(str(prediction))
            parsed = matches[-1].upper() if matches else ''
            answer = str(reference).upper()
            is_correct = parsed == answer
            correct += is_correct
            details.append({
                'pred': prediction,
                'parsed': parsed,
                'answer': answer,
                'correct': is_correct,
            })
        return {
            'accuracy': 100 * correct / len(references) if references else 0,
            'details': details,
        }


class MMLUProBaseEvaluator(BaseEvaluator):

    def is_equal(self, pred, refer):
        try:
            refer_option, refer_string = refer.split('. ')
            if pred in CHOICES and refer_option == pred:
                return True
            elif refer_string.strip() == pred:
                return True
            else :
                return False
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            i = i.split('\n')[0].strip()
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equal(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
