from opencompass.datasets import GlobalPIQADataset, GlobalPIQAEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

global_piqa_reader_cfg = dict(input_columns=['question_prompt'],
                              output_column='answer_letter',
                              test_split='test')
global_piqa_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template='{question_prompt}'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,
                    max_out_len=2048,
                    generation_kwargs=dict(do_sample=True,
                                           temperature=0.8,
                                           top_p=0.95)))
global_piqa_eval_cfg = dict(evaluator=dict(type=GlobalPIQAEvaluator))

global_piqa_datasets = [
    dict(abbr='global_piqa_generation',
         type=GlobalPIQADataset,
         reader_cfg=global_piqa_reader_cfg,
         infer_cfg=global_piqa_infer_cfg,
         eval_cfg=global_piqa_eval_cfg)
]
