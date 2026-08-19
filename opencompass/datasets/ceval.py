import csv
import json
import os.path as osp
import re
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


def ceval_answer_postprocess(text: str) -> str:
    """Extract the final A-D choice from a generated C-Eval answer.

    C-Eval generations may contain Chinese explanations or Latin acronyms.
    Taking the first capital letter therefore mis-scores outputs such as
    ``根据 TCP 协议，答案是 B`` as ``T``.  Prefer explicit answer markers and
    only fall back to a standalone choice letter.
    """
    if not isinstance(text, str):
        return ''
    text = text.strip()
    if not text:
        return ''

    patterns = [
        r'["\']answer["\']\s*:\s*["\']?([A-D])["\']?',
        r'(?:最终答案|正确答案|答案|故选|选择)\s*(?:应该|应当)?\s*'
        r'(?:是|为|选|：|:)?\s*(?:选项)?\s*[\(（\[]?([A-D])',
        r'(?i:(?:final\s+answer|correct\s+(?:answer|option)|answer)\s*'
        r'(?:is|:)?\s*[\(\[]?([A-D]))',
        r'\\boxed\s*\{\s*([A-D])\s*\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()

    # Models often put a bare choice on the final line.
    last_line = next(
        (line.strip() for line in reversed(text.splitlines()) if line.strip()),
        '',
    )
    match = re.fullmatch(r'[\s\*`$\(（\[]*([A-D])[\s\*`$\.。\)）\]]*',
                         last_line,
                         flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    standalone = re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', text,
                            flags=re.IGNORECASE)
    return standalone[-1].upper() if standalone else ''


def ceval_evalscope_answer_postprocess(text: str) -> str:
    """Match EvalScope's native C-Eval answer extractor exactly.

    EvalScope only scores an answer emitted as ``答案：LETTER`` and uses the
    last such marker.  Keep this stricter parser separate from the permissive
    legacy OpenCompass parser so official-protocol reports do not accept
    alternative formats that EvalScope would reject.
    """
    if not isinstance(text, str):
        return ''
    matches = re.findall(r'答案：([A-D])', text)
    return matches[-1] if matches else ''


@LOAD_DATASET.register_module()
class CEvalDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, local_mode: bool = False):
        path = get_data_path(path, local_mode=local_mode)
        dataset = {}
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=path, subset_name=name)
        else:
            for split in ['dev', 'val', 'test']:
                filename = osp.join(path, split, f'{name}_{split}.csv')
                with open(filename, encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row in reader:
                        item = dict(zip(header, row))
                        item.setdefault('explanation', '')
                        item.setdefault('answer', '')
                        dataset.setdefault(split, []).append(item)
            dataset = DatasetDict(
                {i: Dataset.from_list(dataset[i])
                 for i in dataset})
        return dataset


def build_ceval_evalscope_prompt(subject: str, dev_rows: list,
                                 test_row: dict) -> str:
    """Build EvalScope's official single-user-message 5-shot prompt."""

    def choices(row: dict) -> str:
        return '\n'.join(f'{letter}. {row[letter]}' for letter in 'ABCD')

    examples = []
    for row in dev_rows[:5]:
        examples.append(
            f'问题：{row["question"]}\n'
            f'选项：\n{choices(row)}\n'
            f'解析：{row.get("explanation", "")}\n'
            f'答案：{row["answer"]}')
    fewshot = '\n\n'.join(examples)
    return (
        f'以下是一些示例问题：\n\n{fewshot}\n\n\n'
        f'以下是中国关于{subject}的单项选择题，请选出其中的正确答案。'
        '你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），'
        '其中 [LETTER] 是 A、B、C、D 中的一个。\n\n'
        f'问题：{test_row["question"]}\n'
        f'选项：\n{choices(test_row)}\n')


@LOAD_DATASET.register_module()
class CEvalEvalScopeDataset(BaseDataset):
    """C-Eval with EvalScope's 5-shot examples in one user message."""

    @staticmethod
    def load(path: str,
             name: str,
             subject: str,
             local_mode: bool = False):
        source = CEvalDataset.load(path, name, local_mode=local_mode)
        dev_rows = [dict(row) for row in source['dev']][:5]
        val_rows = []
        for raw_row in source['val']:
            row = dict(raw_row)
            row['prompt'] = build_ceval_evalscope_prompt(
                subject, dev_rows, row)
            val_rows.append(row)
        return DatasetDict({'val': Dataset.from_list(val_rows)})


class CEvalDatasetClean(BaseDataset):

    # load the contamination annotations of CEval from
    # https://github.com/liyucheng09/Contamination_Detector
    @staticmethod
    def load_contamination_annotations(path, split='val'):
        import requests

        assert split == 'val', 'Now we only have annotations for val set'
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope.utils.config_ds import MS_DATASETS_CACHE
            annotation_cache_path = osp.join(
                MS_DATASETS_CACHE, 'ceval_contamination_annotations.json')
            link_of_annotations = 'https://modelscope.cn/datasets/opencompass/Contamination_Detector/resolve/master/ceval_annotations.json'  # noqa
        else:
            annotation_cache_path = osp.join(
                path, split, 'ceval_contamination_annotations.json')
            link_of_annotations = 'https://github.com/liyucheng09/Contamination_Detector/releases/download/v0.1.1rc/ceval_annotations.json'  # noqa

        if osp.exists(annotation_cache_path):
            with open(annotation_cache_path, 'r') as f:
                annotations = json.load(f)
            return annotations
        annotations = json.loads(requests.get(link_of_annotations).text)
        with open(annotation_cache_path, 'w') as f:
            json.dump(annotations, f)
        return annotations

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        dataset = {}
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=path, subset_name=name)
            # 向该数据添加 'is_clean' 字段
            annotations = CEvalDatasetClean.load_contamination_annotations(
                path, 'val')
            val = dataset['val']
            val_data = []
            for index in range(val.num_rows):
                row = val[index]
                row_id = f'{name}-{index}'
                row.update({
                    'is_clean':
                    annotations[row_id][0]
                    if row_id in annotations else 'not labeled'
                })
                val_data.append(row)
            dataset['val'] = Dataset.from_list(val_data)
        else:
            for split in ['dev', 'val', 'test']:
                if split == 'val':
                    annotations = \
                        CEvalDatasetClean.load_contamination_annotations(
                            path, split)
                filename = osp.join(path, split, f'{name}_{split}.csv')
                with open(filename, encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row_index, row in enumerate(reader):
                        item = dict(zip(header, row))
                        item.setdefault('explanation', '')
                        item.setdefault('answer', '')
                        if split == 'val':
                            row_id = f'{name}-{row_index}'
                            if row_id in annotations:
                                item['is_clean'] = annotations[row_id][0]
                            else:
                                item['is_clean'] = 'not labeled'
                        dataset.setdefault(split, []).append(item)
            dataset = DatasetDict(
                {i: Dataset.from_list(dataset[i])
                 for i in dataset})
        return dataset
