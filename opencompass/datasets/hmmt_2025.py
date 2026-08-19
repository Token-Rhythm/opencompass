"""Official MathArena adapters for HMMT February/November 2025."""

from datasets import DatasetDict, load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

HMMT_DATASETS = {
    'feb':
    ('MathArena/hmmt_feb_2025', '6fdc4277120810ff75aa22d2d5489b91f7a262a1'),
    'nov':
    ('MathArena/hmmt_nov_2025', '118dbfb45c4c9467c672268ed55166642897aa46'),
}

MATHARENA_2026_DATASETS = {
    'aime_2026':
    ('MathArena/aime_2026', 'd2de22f3c656b4f56cf8981212186377d1e23bc3'),
    'hmmt_feb_2026':
    ('MathArena/hmmt_feb_2026',
     '02fba4f74d8e68e73e66a02d540fd979c05c274c'),
}


def _format_hmmt(row):
    return {
        'prompt': ('Put your final answer within \\boxed{}.\n\n' +
                   row['problem']).strip(),
        'answer':
        str(row['answer']),
    }


def _repeat_train_split(dataset, num_repeats):
    if not isinstance(num_repeats, int) or isinstance(num_repeats, bool):
        raise TypeError('num_repeats must be an integer')
    if num_repeats < 1:
        raise ValueError('num_repeats must be at least 1')
    source = dataset['train']
    indices = list(range(len(source))) * num_repeats
    return source.select(indices)


@LOAD_DATASET.register_module()
class HMMT2025Dataset(BaseDataset):

    @staticmethod
    def load(competition=None, num_repeats=1):
        if competition not in HMMT_DATASETS:
            raise ValueError('competition must be either "feb" or "nov"')
        path, revision = HMMT_DATASETS[competition]
        dataset = load_dataset(path, revision=revision)
        repeated = _repeat_train_split(dataset, num_repeats)
        return DatasetDict({'test': repeated.map(_format_hmmt)})


@LOAD_DATASET.register_module()
class MathArena2026Dataset(BaseDataset):
    """Load a pinned 2026 MathArena competition dataset."""

    @staticmethod
    def load(competition=None, num_repeats=1):
        if competition not in MATHARENA_2026_DATASETS:
            supported = ', '.join(sorted(MATHARENA_2026_DATASETS))
            raise ValueError(
                f'competition must be one of: {supported}')
        path, revision = MATHARENA_2026_DATASETS[competition]
        dataset = load_dataset(path, revision=revision)
        repeated = _repeat_train_split(dataset, num_repeats)
        return DatasetDict({'test': repeated.map(_format_hmmt)})


class MathArenaEvaluator(BaseEvaluator):
    """Use the pinned official MathArena parser and answer checker."""

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different lengths'
            }
        try:
            from ._matharena import check_answers, extract_answer, parse_answer
        except ImportError as error:
            raise ImportError(
                'HMMT 2025 evaluation requires the OpenCompass extra '
                'dependencies sympy, regex and '
                'antlr4-python3-runtime==4.11.') \
                from error

        details = []
        correct = 0
        for prediction, reference in zip(predictions, references):
            list_answer = ',' in str(reference)
            parsed, warning = extract_answer(str(prediction),
                                             strict_parsing=False,
                                             parse=True,
                                             list_answer=list_answer,
                                             typed_delimiters=True)
            gold, _ = parse_answer(str(reference),
                                   list_answer=list_answer,
                                   typed_delimiters=True)
            is_correct = bool(check_answers(parsed, gold))
            correct += is_correct
            details.append({
                'pred': prediction,
                'parsed': str(parsed),
                'answer': reference,
                'correct': is_correct,
                'warning': warning.value,
            })
        return {
            'accuracy': 100 * correct / len(references) if references else 0,
            'details': details,
        }
