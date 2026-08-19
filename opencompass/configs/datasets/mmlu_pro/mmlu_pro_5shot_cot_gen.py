"""Official MMLU-Pro 5-shot chain-of-thought generation task."""

from mmengine.config import read_base

from opencompass.datasets import MMLUProDataset, MMLUProEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever

with read_base():
    from .mmlu_pro_categories import categories


mmlu_pro_5shot_cot_datasets = []
for category in categories:
    description = (
        'The following are multiple choice questions (with answers) about '
        f'{category}.\nThink step by step and then output the answer in the '
        'format of "The answer is (X)" at the end.\n\n')
    mmlu_pro_5shot_cot_datasets.append(
        dict(
            abbr=f'mmlu_pro_{category.replace(" ", "_")}',
            type=MMLUProDataset,
            path='TIGER-Lab/MMLU-Pro',
            revision='b189ec765aa7ed75c8acfea42df31fdae71f97be',
            category=category,
            reader_cfg=dict(
                input_columns=['question_prompt', 'fewshot_prompt'],
                output_column='answer',
                train_split='validation',
                test_split='test',
            ),
            infer_cfg=dict(
                ice_template=dict(
                    type=PromptTemplate,
                    template='{fewshot_prompt}',
                ),
                prompt_template=dict(
                    type=PromptTemplate,
                    template=description + '</E>{question_prompt}',
                    ice_token='</E>',
                ),
                retriever=dict(
                    type=FixKRetriever,
                    fix_id_list=[0, 1, 2, 3, 4],
                    ice_separator='',
                    ice_eos_token='',
                ),
                inferencer=dict(
                    type=GenInferencer,
                    max_out_len=8192,
                ),
            ),
            eval_cfg=dict(evaluator=dict(type=MMLUProEvaluator)),
        ))

# Keep the conventional variable name available to simple launchers.
mmlu_pro_datasets = mmlu_pro_5shot_cot_datasets
