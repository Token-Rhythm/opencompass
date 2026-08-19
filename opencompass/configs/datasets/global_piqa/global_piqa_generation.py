from opencompass.datasets import GlobalPIQADataset, GlobalPIQAEvaluator
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

global_piqa_reader_cfg = dict(input_columns=['question_prompt'],
                              output_column='answer_letter',
                              train_split='test',
                              test_split='test')
global_piqa_infer_cfg = dict(prompt_template=dict(
    type=RawPromptTemplate,
    messages=[dict(role='user', content='{question_prompt}')]),
                             retriever=dict(type=ZeroRetriever),
                             inferencer=dict(type=ParallelGenInferencer,
                                             max_out_len=2048,
                                             save_every=1))
global_piqa_eval_cfg = dict(evaluator=dict(type=GlobalPIQAEvaluator))

global_piqa_datasets = [
    dict(abbr='global_piqa_generation',
         type=GlobalPIQADataset,
         reader_cfg=global_piqa_reader_cfg,
         infer_cfg=global_piqa_infer_cfg,
         eval_cfg=global_piqa_eval_cfg)
]
