from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


aime2025_reader_cfg = dict(input_columns=['question'], output_column='answer')
aime2025_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{question}\nPlease reason step by step, and put your '
                'final answer within \\boxed{}.',
            )
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ParallelGenInferencer, save_every=1),
)
aime2025_eval_cfg = dict(evaluator=dict(type=MATHVerifyEvaluator))

aime2025_datasets = [
    dict(
        type=CustomDataset,
        abbr='aime2025',
        path='opencompass/aime2025',
        reader_cfg=aime2025_reader_cfg,
        infer_cfg=aime2025_infer_cfg,
        eval_cfg=aime2025_eval_cfg,
    )
]
