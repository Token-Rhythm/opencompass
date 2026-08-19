from opencompass.datasets import HMMT2025Dataset, MathArenaEvaluator
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

hmmt_2025_datasets = []
for competition in ('feb', 'nov'):
    reader_cfg = dict(input_columns=['prompt'],
                      output_column='answer',
                      train_split='test',
                      test_split='test')
    infer_cfg = dict(prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(role='user', content='{prompt}')]),
                     retriever=dict(type=ZeroRetriever),
                     inferencer=dict(type=ParallelGenInferencer,
                                     max_out_len=32768,
                                     save_every=1))
    eval_cfg = dict(evaluator=dict(type=MathArenaEvaluator))
    hmmt_2025_datasets.append(
        dict(abbr=f'hmmt_{competition}_2025',
             type=HMMT2025Dataset,
             competition=competition,
             # MathArena's official runner defaults to four runs per problem.
             num_repeats=4,
             reader_cfg=reader_cfg,
             infer_cfg=infer_cfg,
             eval_cfg=eval_cfg))
