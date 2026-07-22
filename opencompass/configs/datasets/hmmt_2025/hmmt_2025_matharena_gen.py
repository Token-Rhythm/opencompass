from opencompass.datasets import HMMT2025Dataset, MathArenaEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

hmmt_2025_datasets = []
for competition in ('feb', 'nov'):
    reader_cfg = dict(input_columns=['prompt'],
                      output_column='answer',
                      test_split='test')
    infer_cfg = dict(prompt_template=dict(type=PromptTemplate,
                                          template='{prompt}'),
                     retriever=dict(type=ZeroRetriever),
                     inferencer=dict(type=GenInferencer, max_out_len=32768))
    eval_cfg = dict(evaluator=dict(type=MathArenaEvaluator))
    hmmt_2025_datasets.append(
        dict(abbr=f'hmmt_{competition}_2025',
             type=HMMT2025Dataset,
             competition=competition,
             reader_cfg=reader_cfg,
             infer_cfg=infer_cfg,
             eval_cfg=eval_cfg))
