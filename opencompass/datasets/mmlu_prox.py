"""MMLU-ProX adapter for the official full/lite and zero/five-shot tasks."""

import re

from datasets import DatasetDict, load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset
from .mmlu_prox_lang_libs import LANG_LIBS, LANG_SUBJECTS

MMLU_PROX_PATH = 'li-lab/MMLU-ProX'
MMLU_PROX_REVISION = '8e6106a6c6ce1c5027e66cc338143cf997b2aa09'
MMLU_PROX_LITE_PATH = 'li-lab/MMLU-ProX-Lite'
MMLU_PROX_LITE_REVISION = 'e82aafb9460529687d3c7e51b401d8dd1dd309dd'
MMLU_PROX_LANGUAGES = tuple(LANG_LIBS)
MMLU_PROX_CATEGORIES = ('biology', 'business', 'chemistry', 'computer science',
                        'economics', 'engineering', 'health', 'history', 'law',
                        'math', 'other', 'philosophy', 'physics', 'psychology')
CHOICES = 'ABCDEFGHIJKLMNOP'


def _format_example(row, lang, category):
    strings = LANG_LIBS[lang]
    prompt = f'{strings[0]}\n{row["question"]}\n{strings[1]}\n'
    for index in range(10):
        option = row.get(f'option_{index}')
        if option is not None:
            prompt += f'{CHOICES[index]}. {option}\n'

    cot = row['cot_content'].replace(strings[4], strings[2])
    suffix = strings[5].format('X')
    # The dataset labels this category with a space, while the official
    # localization table uses identifier-style keys.
    subject = LANG_SUBJECTS[lang][category.replace(' ', '_')]
    description = strings[3].format(subject=subject, ans_suffix=suffix)
    return {
        'language_config': lang,
        'description': description + '\n\n',
        'question_prompt': prompt + strings[2],
        # Split fields preserve the official raw prompt when their contents
        # are concatenated, while also giving chat endpoints proper
        # user/assistant few-shot turns.
        'fewshot_question_prompt': prompt,
        'fewshot_answer_prompt': cot + '\n\n',
        'fewshot_prompt': prompt + cot + '\n\n',
        'answer_letter': row['answer'],
    }


@LOAD_DATASET.register_module()
class MMLUProXDataset(BaseDataset):

    @staticmethod
    def load(path=MMLU_PROX_PATH,
             lang=None,
             category=None,
             revision=MMLU_PROX_REVISION):
        if lang not in LANG_LIBS:
            raise ValueError(f'Unsupported MMLU-ProX language: {lang!r}')
        if category not in MMLU_PROX_CATEGORIES:
            raise ValueError(f'Unsupported MMLU-ProX category: {category!r}')
        dataset = load_dataset(path, lang, revision=revision)
        dataset = dataset.filter(lambda row: row['category'] == category)
        dataset = dataset.map(lambda row: _format_example(row, lang, category))
        return DatasetDict({
            'validation': dataset['validation'],
            'test': dataset['test'],
        })


class MMLUProXEvaluator(BaseEvaluator):
    """Apply the official answer-is regex and punctuation-insensitive EM."""

    @staticmethod
    def _pattern(lang):
        template = LANG_LIBS[lang][5]
        prefix, suffix = template.split('{}')
        if prefix.endswith('(') and suffix.startswith(')'):
            prefix, suffix = prefix[:-1], suffix[1:]
        return re.compile(
            re.escape(prefix) + r'\(?([ABCDEFGHIJ])\)?' + re.escape(suffix),
            re.IGNORECASE)

    def score(self, predictions, references, test_set):
        if not (len(predictions) == len(references) == len(test_set)):
            return {
                'error': 'predictions, references, and test_set have '
                'different lengths'
            }
        details = []
        correct = 0
        for prediction, reference, row in zip(predictions, references,
                                              test_set):
            match = self._pattern(row['language_config']).search(
                str(prediction))
            parsed = match.group(1).upper() if match else ''
            is_correct = parsed == str(reference).upper()
            correct += is_correct
            details.append({
                'pred': prediction,
                'parsed': parsed,
                'answer': reference,
                'language': row['language_config'],
                'correct': is_correct,
            })
        return {
            'accuracy': 100 * correct / len(references) if references else 0,
            'details': details,
        }
