import os

from opencompass.datasets import (AALCRDataset,
                                  generic_llmjudge_postprocess)
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.models import OpenAISDK
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

AA_LCR_GRADER = '''Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {reference}
CANDIDATE ANSWER TO ASSESS: {prediction}

Reply only with CORRECT or INCORRECT.
'''

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])
aa_lcr_judge_cfg = dict(
    abbr='qwen3-235b-a22b-instruct-2507',
    type=OpenAISDK,
    path='Qwen/Qwen3-235B-A22B-Instruct-2507',
    key='ENV',
    openai_api_base=os.environ.get('OPENAI_BASE_URL',
                                   'https://api.openai.com/v1/'),
    meta_template=api_meta_template,
    temperature=0,
    max_seq_len=131072,
    max_out_len=128,
    batch_size=16)

aa_lcr_reader_cfg = dict(input_columns=['prompt', 'question'],
                         output_column='answer',
                         test_split='test')
aa_lcr_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,
                    max_seq_len=131072,
                    max_out_len=8192))
aa_lcr_eval_cfg = dict(evaluator=dict(
    type=GenericLLMEvaluator,
    prompt_template=dict(type=PromptTemplate,
                         template=dict(round=[
                             dict(role='HUMAN', prompt=AA_LCR_GRADER)
                         ])),
    dataset_cfg=dict(type=AALCRDataset, reader_cfg=aa_lcr_reader_cfg),
    judge_cfg=aa_lcr_judge_cfg,
    dict_postprocessor=dict(type=generic_llmjudge_postprocess,
                            true_tag='CORRECT',
                            false_tag='INCORRECT')))

aa_lcr_datasets = [
    dict(abbr='aa_lcr',
         type=AALCRDataset,
         reader_cfg=aa_lcr_reader_cfg,
         infer_cfg=aa_lcr_infer_cfg,
         eval_cfg=aa_lcr_eval_cfg,
         mode='singlescore')
]
