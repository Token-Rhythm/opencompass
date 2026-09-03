"""Basic Inferencer."""
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from mmengine.dist import is_main_process
from torch.utils.data import DataLoader

from opencompass.utils import get_logger

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever


class BaseInferencer:
    """Base Inferencer class for all evaluation Inferencer.

    Attributes:
        model (:obj:`BaseModel`, optional): The module to inference.
        max_model_token_num (:obj:`int`, optional): Maximum number of
            tokenized words allowed by the LM.
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output
            `JSON` file.
    """
    model = None

    def __init__(
        self,
        model,
        max_seq_len: Optional[int] = None,
        batch_size: Optional[int] = 1,
        output_json_filepath: Optional[str] = './icl_inference_output',
        output_json_filename: Optional[str] = 'predictions',
        fix_id_list: Optional[List[int]] = None,
        **kwargs,
    ) -> None:

        if fix_id_list:
            raise ValueError('Passing fix_id_list to Inferencer is no longer '
                             'allowed. Please pass it to FixKRetriever '
                             'instead.')

        self.model = model

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename
        self.is_main_process = is_main_process()
        os.makedirs(self.output_json_filepath, exist_ok=True)

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        """Perform In-Context Inference given a retriever and optional
        templates.

        Args:
            retriever (:obj:`BaseRetriever`): An instance of a Retriever class
                that will be used to retrieve in-context examples
            ice_template (:obj:`PromptTemplate`, optional): A template for
                generating the in-context examples prompt. Defaults to None.
            prompt_template (:obj:`PromptTemplate`, optional): A template for
                generating the final prompt. Defaults to None.
            output_json_filepath (:obj:`str`, optional): The file path to save
                the results as a `JSON` file. Defaults to None.
            output_json_filename (:obj:`str`, optional): The file name to save
                the results as a `JSON` file. Defaults to None.

        Raises:
            NotImplementedError: If the function is not implemented in the
                subclass.

        Returns:
            :obj:`List:` A list of string, each representing the results of one
                inference.
        """
        raise NotImplementedError("Method hasn't been implemented yet")

    @staticmethod
    def get_dataloader(datalist: List[List],
                       batch_size: Optional[int]) -> DataLoader:
        """Return a dataloader of the input data list."""
        if batch_size is None:
            batch_size = 1
        dataloader = DataLoader(datalist,
                                batch_size=batch_size,
                                collate_fn=lambda x: x)
        return dataloader


def dump_results_dict(results_dict, filename):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(results_dict, json_file, indent=4, ensure_ascii=False)


def split_prediction_fields(prediction):
    """Split structured model output into answer and reasoning fields.

    Models traditionally return a string. Reasoning APIs may instead return
    ``{'content': ..., 'reasoning_content': ...}``. The latter is normalized
    here so inference files retain both fields while the legacy ``prediction``
    field contains only the final answer.
    """
    if isinstance(prediction, dict) and ('content' in prediction
                                         or 'reasoning_content' in prediction):
        content = prediction.get('content', '')
        reasoning_content = prediction.get('reasoning_content', '')
        return (content if content is not None else '',
                reasoning_content if reasoning_content is not None else '',
                True)

    if (isinstance(prediction, list) and prediction and all(
            isinstance(item, dict) and
        ('content' in item or 'reasoning_content' in item)
            for item in prediction)):
        contents = []
        reasoning_contents = []
        for item in prediction:
            content = item.get('content', '')
            reasoning_content = item.get('reasoning_content', '')
            contents.append(content if content is not None else '')
            reasoning_contents.append(
                reasoning_content if reasoning_content is not None else '')
        return contents, reasoning_contents, True

    return prediction, None, False


def prediction_content(prediction):
    """Return only the final-answer content from a generated output."""
    return split_prediction_fields(prediction)[0]


def _next_jsonl_backup_path(path: Path) -> Path:
    """Return a backup path without overwriting an earlier recovery file."""
    backup = path.with_name(path.name + '.bak')
    suffix = 1
    while backup.exists():
        backup = path.with_name(f'{path.name}.bak.{suffix}')
        suffix += 1
    return backup


def _restore_results_from_jsonl(path: Path) -> dict:
    """Load every intact JSONL record and quarantine malformed records.

    A process can be interrupted while appending its final record.  The old
    behavior moved the entire temporary file aside when *one* line was
    malformed, which discarded all completed requests on resume.  Preserve
    every independently valid row, keep the original bytes in a backup for
    diagnosis, and rewrite a clean checkpoint containing only those rows.
    Missing/corrupt rows are then naturally submitted again by the inferencer.
    """
    logger = get_logger()
    result_dict = {}
    invalid_lines = []
    # Iterate over physical LF-delimited records. ``str.splitlines()`` also
    # splits at Unicode separators such as U+0085 and U+2028, which are legal
    # inside a JSON string and occur in some LongBench documents. Treating
    # those characters as JSONL boundaries corrupts otherwise valid rows.
    with path.open('r', encoding='utf-8', newline='') as json_file:
        for line_number, line in enumerate(json_file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or 'idx' not in item:
                    raise ValueError('JSONL record must be an object with idx')
                idx = item.pop('idx')
                result_dict[idx] = item
            except (json.JSONDecodeError, TypeError, ValueError):
                invalid_lines.append(line_number)

    if invalid_lines:
        backup = _next_jsonl_backup_path(path)
        shutil.move(path, backup)
        if result_dict:
            with path.open('x', encoding='utf-8') as json_file:
                for idx, result in result_dict.items():
                    json_file.write(
                        json.dumps({
                            'idx': idx,
                            **result,
                        }, ensure_ascii=False) + '\n')
            logger.warning(
                f'Recovered {len(result_dict)} valid entries from {path}; '
                f'quarantined malformed lines {invalid_lines} in {backup}.')
        else:
            logger.warning(
                f'No valid entries could be recovered from {path}; moved '
                f'the original file to {backup}.')
    return result_dict


class GenInferencerOutputHandler:
    origin_prompt_dict = {}
    output_dict = {}
    prediction_dict = {}
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}
        self.dumped_indices = set()

    def write_to_jsonl(self, save_dir: str, filename: str):
        filename = Path(save_dir) / filename
        with open(filename, 'a', encoding='utf-8') as json_file:
            new_lines = []
            for idx, result in self.results_dict.items():
                if idx in self.dumped_indices:
                    continue
                new_lines.append(
                    json.dumps({
                        'idx': idx,
                        **result
                    }, ensure_ascii=False))
                self.dumped_indices.add(idx)
            if new_lines:
                json_file.write('\n'.join(new_lines) + '\n')

    def restore_from_jsonl(self, save_dir: str, filename: str) -> dict:
        path = Path(save_dir) / filename
        if path.exists():
            result_dict = _restore_results_from_jsonl(path)
            self.results_dict = result_dict
            self.dumped_indices.update(result_dict.keys())
        return self.results_dict

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, Path(save_dir) / filename)

    def save_results(self,
                     origin_prompt,
                     prediction,
                     idx,
                     gold=None,
                     res_length=None,
                     input_length=None,
                     inference_error=None):
        structured_error = (prediction.get('inference_error')
                            if isinstance(prediction, dict) else None)
        content, reasoning_content, is_structured = split_prediction_fields(
            prediction)
        self.results_dict[str(idx)] = {
            'origin_prompt': origin_prompt,
            # Keep prediction as a compatibility alias. It must never include
            # reasoning, because existing evaluators consume this field.
            'prediction': content,
        }
        if is_structured:
            self.results_dict[str(idx)]['content'] = content
            self.results_dict[str(
                idx)]['reasoning_content'] = reasoning_content
        if gold:
            self.results_dict[str(idx)]['gold'] = gold
        if res_length:
            self.results_dict[str(idx)]['res_length'] = res_length
        if input_length:
            self.results_dict[str(idx)]['all_input_length'] = input_length
        inference_error = (inference_error
                           if inference_error is not None else structured_error)
        if inference_error is not None:
            self.results_dict[str(idx)]['inference_error'] = inference_error


class ChatOutputHandler:

    def __init__(self) -> None:
        self.results_dict = {}
        self.dumped_indices = set()

    def write_to_jsonl(self, save_dir: str, filename: str):
        filename = Path(save_dir) / filename
        with open(filename, 'a', encoding='utf-8') as json_file:
            new_lines = []
            for idx, result in self.results_dict.items():
                if idx in self.dumped_indices:
                    continue
                new_lines.append(
                    json.dumps({
                        'idx': idx,
                        **result
                    }, ensure_ascii=False))
                self.dumped_indices.add(idx)
            if new_lines:
                json_file.write('\n'.join(new_lines) + '\n')

    def restore_from_jsonl(self, save_dir: str, filename: str) -> dict:
        path = Path(save_dir) / filename
        if path.exists():
            result_dict = _restore_results_from_jsonl(path)
            self.results_dict = result_dict
            self.dumped_indices.update(result_dict.keys())
        return self.results_dict

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, Path(save_dir) / filename)

    def save_results(self,
                     origin_prompt: list,
                     prediction: str,
                     idx: int,
                     gold: str = None):
        result_dict = {}
        if gold:
            result_dict['gold'] = gold
        content, reasoning_content, is_structured = split_prediction_fields(
            prediction)
        result_dict.update({
            'prediction': content,
            'origin_prompt': origin_prompt,
        })
        if is_structured:
            result_dict['content'] = content
            result_dict['reasoning_content'] = reasoning_content
        if isinstance(prediction, dict) and prediction.get(
                'inference_error') is not None:
            result_dict['inference_error'] = prediction['inference_error']
        self.results_dict[str(idx)] = result_dict

    def save_multiround_results(self,
                                origin_prompt: list,
                                prediction: str,
                                idx: int,
                                gold: str = None):
        result_dict = self.results_dict.get(str(idx), {
            'gold': [],
            'prediction': [],
            'origin_prompt': [],
        })
        result_dict['gold'].append(gold)
        content, reasoning_content, is_structured = split_prediction_fields(
            prediction)
        result_dict['prediction'].append(content)
        if is_structured:
            result_dict.setdefault('content', []).append(content)
            result_dict.setdefault('reasoning_content',
                                   []).append(reasoning_content)
        round_error = (prediction.get('inference_error')
                       if isinstance(prediction, dict) else None)
        if round_error is not None or 'inference_error' in result_dict:
            errors = result_dict.setdefault(
                'inference_error',
                [None] * (len(result_dict['prediction']) - 1))
            errors.append(round_error)
        result_dict['origin_prompt'].append(origin_prompt)
        self.results_dict[str(idx)] = result_dict


class PPLInferencerOutputHandler:
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, Path(save_dir) / filename)

    def save_ice(self, ice):
        for idx, example in enumerate(ice):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['in-context examples'] = example

    def save_predictions(self, predictions):
        for idx, prediction in enumerate(predictions):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['prediction'] = prediction

    def save_prompt_and_ppl(self, label, input, prompt, ppl, idx):
        if str(idx) not in self.results_dict.keys():
            self.results_dict[str(idx)] = {}
        if 'origin_prompt' not in self.results_dict[str(idx)]:
            self.results_dict[str(idx)]['origin_prompt'] = input
        if 'label: ' + str(label) not in self.results_dict[str(idx)].keys():
            self.results_dict[str(idx)]['label: ' + str(label)] = {}
        self.results_dict[str(idx)]['label: ' +
                                    str(label)]['testing input'] = input
        self.results_dict[str(idx)]['label: ' + str(label)]['prompt'] = prompt
        self.results_dict[str(idx)]['label: ' + str(label)]['PPL'] = ppl

    def save_golds(self, golds):
        for idx, gold in enumerate(golds):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['gold'] = gold


class CLPInferencerOutputHandler:
    results_dict = {}

    def __init__(self) -> None:
        self.results_dict = {}

    def write_to_json(self, save_dir: str, filename: str):
        """Dump the result to a json file."""
        dump_results_dict(self.results_dict, Path(save_dir) / filename)

    def save_ice(self, ice):
        for idx, example in enumerate(ice):
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['in-context examples'] = example

    def save_prompt_and_condprob(self,
                                 input,
                                 prompt,
                                 cond_prob,
                                 idx,
                                 choices,
                                 gold=None):
        if str(idx) not in self.results_dict.keys():
            self.results_dict[str(idx)] = {}
        # TODO:
        # for single token situation, the input will always be yes currently
        self.results_dict[str(idx)]['testing input'] = input
        self.results_dict[str(idx)]['prompt'] = prompt
        # TODO: hard code here
        self.results_dict[str(idx)]['choices'] = choices
        # For calculate auc scores, set scores as prediction
        self.results_dict[str(idx)]['prediction'] = cond_prob
        # set pred label in case needed
        self.results_dict[str(idx)]['pred_label'] = int(np.argmax(cond_prob))
        self.results_dict[str(idx)]['gold'] = gold
