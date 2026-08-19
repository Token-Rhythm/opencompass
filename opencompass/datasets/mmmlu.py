# flake8: noqa
# yapf: disable

import json
import os.path as osp
import re

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


MMMLU_QUERY_TEMPLATE = """\
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}

B) {B}

C) {C}

D) {D}"""

# Kept in the same order as openai/simple-evals. The reference evaluator stops
# at the first answer-prefix family that matches the normalized response.
MMMLU_ANSWER_PREFIXES = (
    r'Answer\s*:',
    r'Answer\s*:',  # duplicated upstream for a Korean invisible-char case
    r'উত্তর\s*:',
    r'उत्तर\s*:',
    r'উত্তরঃ',
    r'উত্তর\s*:',
    r'Antwort\s*:',
    r'답변\s*:',
    r'정답\s*:',
    r'답\s*:',
    r'答案\s*：',
    r'答案\s*:',
    r'答\s*：',
    r'答\s*:',
    r'答复\s*：',
    r'答曰\s*：',
    r'الإجابة:',
    r'الجواب:',
    r'إجابة:',
    r'الإجابة النهائية:',
    r'الإجابة الصحيحة:',
    r'الإجابة الصحيحة هي:',
    r'الإجابة هي:',
    r'الجواب النهائي:',
    r'Respuesta\s*:',
    r'Risposta\s*:',
    r'答え\s*:',
    r'答え\s*：',
    r'回答\s*:',
    r'回答\s*：',
    r'解答\s*:',
    r'Jawaban\s*:',
    r'Réponse\s*:',
    r'Resposta\s*:',
    r'Jibu\s*:',
    r'Idahun\s*:',
    r'Ìdáhùn\s*:',
    r'Idáhùn\s*:',
    r'Àmọ̀nà\s*:',
    r'Àdáhùn\s*:',
    r'Ànúgọ\s*:',
    r'Àṣàyàn\s*:',
)
MMMLU_LOCALIZED_CHOICES = str.maketrans({
    'أ': 'A',
    'ب': 'B',
    'ج': 'C',
    'د': 'D',
    'অ': 'A',
    'ব': 'B',
    'ড': 'C',
    'ঢ': 'D',
    'Ａ': 'A',
    'Ｂ': 'B',
    'Ｃ': 'C',
    'Ｄ': 'D',
})


def format_mmmlu_question(row) -> str:
    """Reproduce openai/simple-evals' zero-shot multilingual MMLU prompt."""
    return MMMLU_QUERY_TEMPLATE.format(**row)


@LOAD_DATASET.register_module()
class MMMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, hf_revision: str = None):
        dataset = DatasetDict()
        subset = name.split('_')[1].replace('-', '_')
        for split in ['test']:
            data = load_dataset(path=path,
                                name=subset,
                                split=split,
                                revision=hf_revision)
            dataset_list = []
            for item in data:
                dataset_list.append({
                    'question_prompt': format_mmmlu_question(item),
                    'target': item['Answer'],
                    'subject': item['Subject'].replace('_', ' ')
                })
            dataset[split] = Dataset.from_list(dataset_list)
        return dataset


def mmmlu_answer_postprocess(text: str) -> str:
    """Reproduce simple-evals' normalized multilingual answer extraction."""
    normalized = (str(text).replace('**', '').replace('$\\boxed{', '')
                  .replace('}$', '').replace('\\$', '').replace('$\\text{', '')
                  .replace('$', '').replace('\\mathrm{', '')
                  .replace('\\{', '').replace('\\text', '')
                  .replace('\\(', '').replace('\\mathbf{', '')
                  .replace('{', '').replace('\\boxed', ''))
    choice_pattern = r'([A-D]|[أ-د]|[অবডঢ]|[Ａ-Ｄ])'
    for prefix in MMMLU_ANSWER_PREFIXES:
        match = re.search(f'(?i){prefix}[ \\t]*{choice_pattern}', normalized)
        if match:
            return match.group(1).translate(MMMLU_LOCALIZED_CHOICES).upper()
    return ''

@LOAD_DATASET.register_module()
class MMMLULiteDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path, local_mode=False)
        dataset = DatasetDict()
        name = name.split('_')[-1]
        raw_data = []
        filename = osp.join(path, name, 'test.jsonl')
        with open(filename, encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f.readlines()]
        dataset['test'] = Dataset.from_list(raw_data)
        return dataset
