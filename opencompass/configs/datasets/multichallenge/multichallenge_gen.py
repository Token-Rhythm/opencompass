import os

from opencompass.datasets import (MultiChallengeDataset,
                                  multichallenge_postprocess)
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.models import OpenAISDK
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{prediction}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{target_question}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".'''

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])
multichallenge_judge_cfg = dict(
    abbr='gpt-4o-2024-08-06',
    type=OpenAISDK,
    path='gpt-4o-2024-08-06',
    key='ENV',
    openai_api_base=os.environ.get('OPENAI_BASE_URL',
                                   'https://api.openai.com/v1/'),
    meta_template=api_meta_template,
    temperature=0,
    max_seq_len=16384,
    max_out_len=4096,
    batch_size=16,
    openai_extra_kwargs=dict(response_format=dict(
        type='json_schema',
        json_schema=dict(name='JudgeResponse',
                         strict=True,
                         schema=dict(
                             type='object',
                             properties=dict(
                                 reasoning=dict(type='string'),
                                 verdict=dict(type='string',
                                              enum=['YES', 'NO']),
                             ),
                             required=['reasoning', 'verdict'],
                             additionalProperties=False,
                         )))))

multichallenge_reader_cfg = dict(input_columns=['dialogue'],
                                 output_column='pass_criteria',
                                 train_split='test',
                                 test_split='test')
multichallenge_infer_cfg = dict(ice_template=dict(type=PromptTemplate,
                                                  template=''),
                                retriever=dict(type=ZeroRetriever),
                                inferencer=dict(type=ChatInferencer,
                                                infer_mode='last',
                                                max_out_len=4096))
multichallenge_eval_cfg = dict(evaluator=dict(
    type=GenericLLMEvaluator,
    prompt_template=dict(type=PromptTemplate,
                         template=dict(
                             round=[dict(role='HUMAN', prompt=JUDGE_PROMPT)])),
    dataset_cfg=dict(type=MultiChallengeDataset,
                     reader_cfg=multichallenge_reader_cfg),
    judge_cfg=multichallenge_judge_cfg,
    dict_postprocessor=dict(type=multichallenge_postprocess)))

multichallenge_datasets = [
    dict(abbr='multichallenge',
         type=MultiChallengeDataset,
         reader_cfg=multichallenge_reader_cfg,
         infer_cfg=multichallenge_infer_cfg,
         eval_cfg=multichallenge_eval_cfg,
         mode='singlescore')
]
