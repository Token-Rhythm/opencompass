from opencompass.datasets.supergpqa.supergpqa import (
    SuperGPQADataset,
    SuperGPQAEvaluator,
)
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


# Reader configuration
reader_cfg = dict(
    input_columns=[
        'question',
        'options',
        'discipline',
        'field',
        'subfield',
        'difficulty',
        'infer_prompt',
        'prompt_mode',
    ],
    output_column='answer_letter',
)

# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{infer_prompt}',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    # SuperGPQA contains 26k+ examples.  Keep a rolling API worker pool so a
    # single long-thinking response does not hold an entire inference batch,
    # and persist every completed response for safe resume.
    inferencer=dict(type=ParallelGenInferencer, save_every=1),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=SuperGPQAEvaluator),
    pred_role='BOT',
)
supergpqa_dataset = dict(
    type=SuperGPQADataset,
    abbr='supergpqa',
    path='m-a-p/SuperGPQA',
    hf_revision='4430d4458112c7d4497fdcf94d7cc223313d6acf',
    prompt_mode='zero-shot',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
)

supergpqa_datasets = [supergpqa_dataset]
