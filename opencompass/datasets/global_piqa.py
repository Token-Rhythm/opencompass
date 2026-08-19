"""Global PIQA generation adapter following lm-evaluation-harness v1.0."""

import re
from collections import defaultdict

from datasets import (DatasetDict, concatenate_datasets,
                      get_dataset_config_names, load_dataset)

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

GLOBAL_PIQA_SOURCES = {
    'nonparallel': ('mrlbenchmarks/global-piqa-nonparallel',
                    '6777742fa3634c0583cda3b7f8a482ea7b1b0937'),
    'parallel': ('mrlbenchmarks/global-piqa-parallel',
                 'b0b18516a8bc2cb1106bce3dd4db32848ca715ea'),
}
LETTERS = 'ABCD'


def _format_global_piqa(row, variant, language):
    if variant == 'nonparallel':
        prompt = (
            'Given the following situation, which option is more likely to '
            'be correct?\n\n'
            f'Situation:\n{row["prompt"]} ...\n\n'
            f'Option A: {row["solution0"]}\n\n'
            f'Option B: {row["solution1"]}\n\n'
            'Your response should end with "The best answer is: '
            '[answer_letter]" where [answer_letter] is one of A or B.')
    else:
        prompt = (
            f'{row["prompt"]}\n\n'
            f'Option A: {row["solution0"]}\n\n'
            f'Option B: {row["solution1"]}\n\n'
            f'Option C: {row["solution2"]}\n\n'
            f'Option D: {row["solution3"]}\n\n'
            'Your response should end with "The best answer is: '
            '[answer_letter]" where [answer_letter] is one of A, B, C, or D.')
    return {
        'question_prompt': prompt,
        'answer_letter': LETTERS[int(row['label'])],
        'variant': variant,
        'language_config': language,
    }


@LOAD_DATASET.register_module()
class GlobalPIQADataset(BaseDataset):
    """Load every official language config from both benchmark variants."""

    @staticmethod
    def load(sources=GLOBAL_PIQA_SOURCES, samples_per_config=None):
        if (samples_per_config is not None
                and (not isinstance(samples_per_config, int)
                     or isinstance(samples_per_config, bool)
                     or samples_per_config <= 0)):
            raise ValueError('samples_per_config must be a positive integer')
        subsets = []
        for variant, (path, revision) in sources.items():
            configs = get_dataset_config_names(path, revision=revision)
            for language in configs:
                split = load_dataset(path,
                                     language,
                                     revision=revision,
                                     split='test')
                if samples_per_config is not None:
                    split = split.select(
                        range(min(samples_per_config, len(split))))
                split = split.map(lambda row, v=variant, lang=language:
                                  _format_global_piqa(row, v, lang))
                split = split.select_columns([
                    'question_prompt', 'answer_letter', 'variant',
                    'language_config'
                ])
                subsets.append(split)
        if not subsets:
            raise RuntimeError('Global PIQA returned no language configs.')
        return DatasetDict({'test': concatenate_datasets(subsets)})


class GlobalPIQAEvaluator(BaseEvaluator):
    """Strict extraction and two-level unweighted language/variant macro."""

    patterns = {
        'nonparallel':
        re.compile(r'[Tt]he (?:[Bb]est [Aa]nswer|[Ff]inal [Aa]nswer|[Aa]nswer)'
                   r'[^A-B]*([A-B])|[Aa]nswer\s*:[^A-B]*([A-B])|'
                   r'\\boxed\{([A-B])\}'),
        'parallel':
        re.compile(r'[Tt]he (?:[Bb]est [Aa]nswer|[Ff]inal [Aa]nswer|[Aa]nswer)'
                   r'[^A-D]*([A-D])|[Aa]nswer\s*:[^A-D]*([A-D])|'
                   r'\\boxed\{([A-D])\}'),
    }

    def score(self, predictions, references, test_set):
        if not (len(predictions) == len(references) == len(test_set)):
            return {
                'error': 'predictions, references, and test_set have '
                'different lengths'
            }
        by_language = defaultdict(list)
        details = []
        for prediction, reference, row in zip(predictions, references,
                                              test_set):
            matches = self.patterns[row['variant']].findall(str(prediction))
            parsed = ''
            if matches:
                parsed = next((value for value in matches[-1] if value), '')
            is_correct = parsed.lower() == str(reference).lower()
            key = (row['variant'], row['language_config'])
            by_language[key].append(float(is_correct))
            details.append({
                'pred': prediction,
                'parsed': parsed,
                'answer': reference,
                'variant': row['variant'],
                'language': row['language_config'],
                'correct': is_correct,
            })

        language_scores = {
            key: sum(values) / len(values)
            for key, values in by_language.items()
        }
        variant_scores = {}
        for variant in GLOBAL_PIQA_SOURCES:
            scores = [
                value for (name, _), value in language_scores.items()
                if name == variant
            ]
            variant_scores[variant] = sum(scores) / len(
                scores) if scores else 0
        overall = sum(variant_scores.values()) / len(variant_scores)
        result = {
            'accuracy': 100 * overall,
            'nonparallel_accuracy': 100 * variant_scores['nonparallel'],
            'parallel_accuracy': 100 * variant_scores['parallel'],
            'details': details,
        }
        result.update({
            f'{variant}/{language}': 100 * score
            for (variant, language), score in language_scores.items()
        })
        return result
