"""Official MMLU-ProX Full zero-shot CoT evaluation."""

from mmengine.config import read_base

from opencompass.datasets import MMLUProXDataset, MMLUProXEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from .mmlu_prox_5shot_cot_gen import categories, question_stops

mmlu_prox_0shot_datasets = []
for lang, question_stop in question_stops.items():
    for category in categories:
        mmlu_prox_0shot_datasets.append(
            dict(abbr=f'mmlu_prox_0shot_{lang}_{category.replace(" ", "_")}',
                 type=MMLUProXDataset,
                 path='li-lab/MMLU-ProX',
                 revision='8e6106a6c6ce1c5027e66cc338143cf997b2aa09',
                 lang=lang,
                 category=category,
                 reader_cfg=dict(
                     input_columns=['description', 'question_prompt'],
                     output_column='answer_letter',
                     train_split='validation',
                     test_split='test'),
                 infer_cfg=dict(prompt_template=dict(
                     type=RawPromptTemplate,
                     messages=[
                         dict(role='system', content='{description}'),
                         dict(role='user', content='{question_prompt}'),
                     ]),
                                retriever=dict(type=ZeroRetriever),
                                inferencer=dict(type=GenInferencer,
                                                max_out_len=2048,
                                                generation_kwargs=dict(
                                                    do_sample=False,
                                                    temperature=0.0))),
                 eval_cfg=dict(evaluator=dict(type=MMLUProXEvaluator))))
