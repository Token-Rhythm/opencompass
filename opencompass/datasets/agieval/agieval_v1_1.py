"""Local AGIEval v1.1 with single-pass generation and legacy scoring.

Data is pinned to AGIEVAL_V1_1_REVISION. Nothing is
downloaded by the loader, so inference can run entirely offline.
"""

import json
from pathlib import Path

from datasets import Dataset

from opencompass.registry import LOAD_DATASET

from ..base import BaseDataset
from . import dataset_loader as legacy
from .agieval import _format_label
from .agieval_v1_1_postprocess import normalize_mathqa_label

AGIEVAL_V1_1_REVISION = '84ab72d94318290aad2e4ec820d535a95a1f7552'
AGIEVAL_V1_1_PATH = './data/AGIEval/data/v1_1'
AGIEVAL_V1_1_TASKS = (
    'gaokao-chinese', 'gaokao-english', 'gaokao-geography',
    'gaokao-history', 'gaokao-biology', 'gaokao-chemistry',
    'gaokao-physics', 'gaokao-mathqa', 'logiqa-zh',
    'lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math',
    'sat-en', 'sat-en-without-passage', 'aqua-rat', 'jec-qa-kd',
    'jec-qa-ca', 'gaokao-mathcloze', 'math',
)
AGIEVAL_V1_1_CLOZE = ('gaokao-mathcloze', 'math')
AGIEVAL_V1_1_EN_MC = (
    'lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math',
    'sat-en', 'sat-en-without-passage', 'aqua-rat',
)
AGIEVAL_V1_1_SETTINGS = ('zero-shot', 'zero-shot-CoT')


def single_choice_label(label):
    """Normalize v1.1 singleton labels; reject accidentally supplied v1 data."""
    if isinstance(label, list) and len(label) != 1:
        raise ValueError('AGIEval v1.1 requires exactly one reference option')
    value = _format_label(label)
    if not isinstance(value, str) or len(value) != 1 or value not in 'ABCDEFG':
        raise ValueError(f'Invalid AGIEval v1.1 single-choice label: {label!r}')
    return value


@LOAD_DATASET.register_module()
class AGIEvalV11Dataset(BaseDataset):
    """Read v1.1 JSONL and build a single model request per test question."""

    @staticmethod
    def load(path=AGIEVAL_V1_1_PATH, name=None, setting_name='zero-shot',
             chat_mode=True):
        if name not in AGIEVAL_V1_1_TASKS:
            raise ValueError(f'Unknown AGIEval v1.1 task: {name!r}')
        if setting_name not in AGIEVAL_V1_1_SETTINGS:
            raise ValueError(f'Unsupported AGIEval setting: {setting_name!r}')
        if chat_mode is not True:
            raise ValueError('AGIEval v1.1 supports only chat_mode=True')

        data_path = Path(path).expanduser()
        filename = data_path / f'{name}.jsonl'
        if not filename.is_file():
            raise FileNotFoundError(
                f'Missing local AGIEval v1.1 data: {filename}. '
                'Download the pinned v1.1 data before inference.')
        with filename.open(encoding='utf-8-sig') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows:
            raise ValueError(f'Empty AGIEval dataset: {filename}')

        records = []
        for index, row in enumerate(rows):
            if name in AGIEVAL_V1_1_CLOZE:
                label = _format_label(row['answer'])
            elif name == 'gaokao-mathqa':
                label = normalize_mathqa_label(row['label'])
            else:
                label = single_choice_label(row['label'])
            if not isinstance(label, str) or not label:
                raise ValueError(f'Missing answer in {name} row {index}')
            if setting_name == 'zero-shot':
                prompt = legacy.convert_zero_shot(row, name)
            else:
                prompt = legacy.convert_zero_shot_CoT_stage1(row, name)
            messages = (prompt if isinstance(prompt, list) else
                        [dict(role='user', content=prompt)])
            # RawPromptTemplate expands these messages without formatting
            # their contents again; braces in math/code remain untouched.
            records.append(dict(id=index, messages=messages, label=label,
                                num_shots=0))
        return Dataset.from_list(records)
