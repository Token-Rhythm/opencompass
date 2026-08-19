from opencompass.datasets import MathArena2026Dataset, MathArenaEvaluator
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


aime2026_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answer',
    train_split='test',
    test_split='test',
)
aime2026_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[dict(role='user', content='{prompt}')],
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ParallelGenInferencer, save_every=1),
)
aime2026_eval_cfg = dict(evaluator=dict(type=MathArenaEvaluator))

aime2026_datasets = [
    dict(
        type=MathArena2026Dataset,
        abbr='aime2026',
        competition='aime_2026',
        # Match MathArena's official default of four runs per problem.
        num_repeats=4,
        reader_cfg=aime2026_reader_cfg,
        infer_cfg=aime2026_infer_cfg,
        eval_cfg=aime2026_eval_cfg,
    )
]
