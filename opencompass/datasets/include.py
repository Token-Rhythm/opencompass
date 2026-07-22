"""INCLUDE base-44 adapter for the official zero-shot protocol."""

from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

INCLUDE_PATH = 'CohereLabs/include-base-44'
INCLUDE_REVISION = 'd2e1f6015f67a43c02a9a68db98e2298e2d6a660'
INCLUDE_LANGUAGES = ('Albanian', 'Arabic', 'Armenian', 'Azerbaijani', 'Basque',
                     'Belarusian', 'Bengali', 'Bulgarian', 'Chinese',
                     'Croatian', 'Dutch', 'Estonian', 'Finnish', 'French',
                     'Georgian', 'German', 'Greek', 'Hebrew', 'Hindi',
                     'Hungarian', 'Indonesian', 'Italian', 'Japanese',
                     'Kazakh', 'Korean', 'Lithuanian', 'Malay', 'Malayalam',
                     'Nepali', 'North Macedonian', 'Persian', 'Polish',
                     'Portuguese', 'Russian', 'Serbian', 'Spanish', 'Tagalog',
                     'Tamil', 'Telugu', 'Turkish', 'Ukrainian', 'Urdu',
                     'Uzbek', 'Vietnamese')


def _format_include(row):
    prompt = (f'{row["question"].strip()}\n'
              f'A. {row["option_a"]}\n'
              f'B. {row["option_b"]}\n'
              f'C. {row["option_c"]}\n'
              f'D. {row["option_d"]}\nAnswer:')
    return {'prompt': prompt, 'answer': int(row['answer'])}


@LOAD_DATASET.register_module()
class INCLUDEDataset(BaseDataset):
    """Load one of the 44 official language configurations."""

    @staticmethod
    def load(path=INCLUDE_PATH, lang=None, revision=INCLUDE_REVISION):
        if lang not in INCLUDE_LANGUAGES:
            raise ValueError(f'Unsupported INCLUDE language: {lang!r}')
        dataset = load_dataset(path, lang, revision=revision)
        return DatasetDict({
            'test': dataset['test'].map(_format_include),
        })
