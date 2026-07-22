"""Artificial Analysis Long Context Reasoning (AA-LCR) adapter."""

import csv
import unicodedata
import zipfile

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

AA_LCR_PATH = 'ArtificialAnalysis/AA-LCR'
AA_LCR_REVISION = 'bdae010bbce259820c0e34c1d7cce210d966fb75'


def _build_aa_lcr_prompt(documents, question):
    documents_text = '\n\n'.join(
        f'BEGIN DOCUMENT {index + 1}:\n{document}\nEND DOCUMENT {index + 1}'  # noqa: E231,E501
        for index, document in enumerate(documents))
    return f'''BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
'''


def _resolve_archive_member(archive, member):
    """Handle UTF-8 names stored without the ZIP UTF-8 filename flag."""
    candidates = [member]
    for normalization in ('NFC', 'NFD'):
        normalized = unicodedata.normalize(normalization, member)
        candidates.append(normalized)
        try:
            candidates.append(normalized.encode('utf-8').decode('cp437'))
        except UnicodeError:
            pass
    for candidate in candidates:
        if candidate in archive.NameToInfo:
            return candidate
    raise KeyError(f'Document is missing from the AA-LCR archive: {member}')


@LOAD_DATASET.register_module()
class AALCRDataset(BaseDataset):
    """Load the 100 official questions and their ordered document sets."""

    @staticmethod
    def load(path=AA_LCR_PATH, revision=AA_LCR_REVISION):
        csv_path = hf_hub_download(path,
                                   'AA-LCR_Dataset.csv',
                                   repo_type='dataset',
                                   revision=revision)
        archive_path = hf_hub_download(
            path,
            'extracted_text/AA-LCR_extracted-text.zip',
            repo_type='dataset',
            revision=revision)

        rows = []
        with open(csv_path, encoding='utf-8') as stream, zipfile.ZipFile(
                archive_path) as archive:
            for raw in csv.DictReader(stream):
                filenames = raw['data_source_filenames'].split(';')
                category = raw['document_category'].replace(' ', '_')
                documents = []
                for filename in filenames:
                    member = (f'lcr/{category}/{raw["document_set_id"]}/'
                              f'{filename}')
                    with archive.open(_resolve_archive_member(
                            archive, member)) as document:
                        documents.append(document.read().decode('utf-8'))
                rows.append({
                    'question_id':
                    raw['question_id'],
                    'question':
                    raw['question'],
                    # The official loader treats semicolon-separated entries
                    # as independent answer criteria before interpolating them
                    # into the equality-checker prompt.
                    'answer':
                    raw['answer'].split(';'),
                    'prompt':
                    _build_aa_lcr_prompt(documents, raw['question']),
                    'document_category':
                    raw['document_category'],
                    'document_set_id':
                    raw['document_set_id'],
                    'input_tokens':
                    int(raw['input_tokens']),
                })
        return DatasetDict({'test': Dataset.from_list(rows)})
