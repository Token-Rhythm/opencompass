from opencompass.datasets.decision_benchmarks import KevDataset, KevEvaluator
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

reader = dict(input_columns=['system_prompt', 'question_prompt'], output_column='reference',
              train_split='test', test_split='test')
infer = dict(prompt_template=dict(type=RawPromptTemplate, messages=[
    dict(role='system', content='{system_prompt}'),
    dict(role='user', content='{question_prompt}')]),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ParallelGenInferencer, max_out_len=4096, save_every=1))
kev_datasets = [dict(abbr='kev_' + suite.replace('-', '_') + '_dev', type=KevDataset, suite=suite, partition='development', reader_cfg=reader, infer_cfg=infer, eval_cfg=dict(evaluator=dict(type=KevEvaluator))) for suite in ('decision-v7', 'transfer-v4', 'transfer-v9')]
