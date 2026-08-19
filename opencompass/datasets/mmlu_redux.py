"""MMLU-Redux 2.0 adapter matching the official generative protocol."""

import re
from collections import defaultdict

from datasets import DatasetDict, concatenate_datasets, load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

MMLU_REDUX_PATH = 'edinburgh-dawg/mmlu-redux-2.0'
MMLU_REDUX_REVISION = '372ea425445d51e1ba1188c56e5e893f8138621f'
CHOICES = 'ABCD'
MMLU_REDUX_DESCRIPTION = (
    'The following are multiple choice questions (with answers) about '
    '{subject}.\n\n')
MMLU_REDUX_CATEGORIES = {
    'humanities':
    ('formal_logic', 'high_school_european_history', 'high_school_us_history',
     'high_school_world_history', 'international_law', 'jurisprudence',
     'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
     'prehistory', 'professional_law', 'world_religions'),
    'other': ('business_ethics', 'clinical_knowledge', 'college_medicine',
              'global_facts', 'human_aging', 'management', 'marketing',
              'medical_genetics', 'miscellaneous', 'nutrition',
              'professional_accounting', 'professional_medicine', 'virology'),
    'social_sciences':
    ('econometrics', 'high_school_geography',
     'high_school_government_and_politics', 'high_school_macroeconomics',
     'high_school_microeconomics', 'high_school_psychology', 'human_sexuality',
     'professional_psychology', 'public_relations', 'security_studies',
     'sociology', 'us_foreign_policy'),
    'stem':
    ('abstract_algebra', 'anatomy', 'astronomy', 'college_biology',
     'college_chemistry', 'college_computer_science', 'college_mathematics',
     'college_physics', 'computer_security', 'conceptual_physics',
     'electrical_engineering', 'elementary_mathematics', 'high_school_biology',
     'high_school_chemistry', 'high_school_computer_science',
     'high_school_mathematics', 'high_school_physics',
     'high_school_statistics', 'machine_learning'),
}
MMLU_REDUX_SUBJECTS = tuple(subject
                            for subjects in MMLU_REDUX_CATEGORIES.values()
                            for subject in subjects)
SUBJECT_TO_CATEGORY = {
    subject: category
    for category, subjects in MMLU_REDUX_CATEGORIES.items()
    for subject in subjects
}


def _mmlu_redux_description(subject):
    """Return lm-eval's per-subject MMLU-Redux task description."""
    return MMLU_REDUX_DESCRIPTION.format(subject=subject.replace('_', ' '))


def _format_mmlu_redux(row):
    prompt = row['question'].strip() + '\n'
    prompt += '\n'.join(f'{letter}. {choice}'
                        for letter, choice in zip(CHOICES, row['choices']))
    prompt += ('\nPlease respond with the correct letter (A, B, C or D) '
               'without any additional comments, only the correct letter:')
    return {
        'prompt': prompt,
        'answer_letter': CHOICES[int(row['answer'])],
    }


@LOAD_DATASET.register_module()
class MMLUReduxDataset(BaseDataset):
    """Load all subjects and retain samples marked ``error_type=ok``.

    The official lm-evaluation-harness task uses a derived ``-ok`` dataset.
    Filtering the version-pinned official source here is equivalent and keeps
    provenance tied to the benchmark authors' dataset. ``subject`` remains an
    optional compatibility/debug selector; the official config loads all 57.
    """

    @staticmethod
    def load(path=MMLU_REDUX_PATH, subject=None, revision=MMLU_REDUX_REVISION):
        if subject is not None and subject not in MMLU_REDUX_SUBJECTS:
            raise ValueError(f'Unsupported MMLU-Redux subject: {subject!r}')
        subjects = (subject, ) if subject else MMLU_REDUX_SUBJECTS
        subsets = []
        for name in subjects:
            split = load_dataset(path, name, revision=revision, split='test')
            split = split.filter(lambda row: row['error_type'] == 'ok')
            split = split.map(
                lambda row, current=name: {
                    **_format_mmlu_redux(row),
                    'description':
                    _mmlu_redux_description(current),
                    'subject':
                    current,
                    'category':
                    SUBJECT_TO_CATEGORY[current],
                })
            split = split.select_columns([
                'description', 'prompt', 'answer_letter', 'subject', 'category'
            ])
            subsets.append(split)
        return DatasetDict({'test': concatenate_datasets(subsets)})


class MMLUReduxEvaluator(BaseEvaluator):
    """Official first-capital-letter extraction plus exact match."""

    pattern = re.compile(r'([ABCD])')

    def score(self, predictions, references, test_set):
        details = []
        correct = 0
        by_subject = defaultdict(list)
        by_category = defaultdict(list)
        for prediction, reference, row in zip(predictions, references,
                                              test_set):
            match = self.pattern.search(str(prediction))
            parsed = match.group(1) if match else ''
            is_correct = parsed.lower() == str(reference).lower()
            correct += is_correct
            by_subject[row['subject']].append(float(is_correct))
            by_category[row['category']].append(float(is_correct))
            details.append({
                'pred': prediction,
                'parsed': parsed,
                'answer': reference,
                'subject': row['subject'],
                'category': row['category'],
                'correct': is_correct,
            })
        result = {
            'accuracy': 100 * correct / len(references) if references else 0,
            'details': details,
        }
        result.update({
            f'category/{name}': 100 * sum(scores) / len(scores)
            for name, scores in by_category.items()
        })
        result.update({
            f'subject/{name}': 100 * sum(scores) / len(scores)
            for name, scores in by_subject.items()
        })
        return result
