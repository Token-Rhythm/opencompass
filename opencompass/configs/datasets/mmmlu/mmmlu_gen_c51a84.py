from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import MMMLUDataset
from opencompass.datasets.mmmlu import mmmlu_answer_postprocess


mmmlu_reader_cfg = dict(
    input_columns=['question_prompt'],
    output_column='target',
    train_split='test')

mmmlu_all_sets = [
    'mmlu_AR-XY',
    'mmlu_BN-BD',
    'mmlu_DE-DE',
    'mmlu_ES-LA',
    'mmlu_FR-FR',
    'mmlu_HI-IN',
    'mmlu_ID-ID',
    'mmlu_IT-IT',
    'mmlu_JA-JP',
    'mmlu_KO-KR',
    'mmlu_PT-BR',
    'mmlu_SW-KE',
    'mmlu_YO-NG',
    'mmlu_ZH-CN',
]

mmmlu_datasets = []
for _name in mmmlu_all_sets:
    mmmlu_infer_cfg = dict(
        prompt_template=dict(
            type=RawPromptTemplate,
            messages=[dict(role='user', content='{question_prompt}')],
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=ParallelGenInferencer, save_every=1),
    )

    mmmlu_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=mmmlu_answer_postprocess))

    mmmlu_datasets.append(
        dict(
            abbr=f'openai_m{_name}',
            type=MMMLUDataset,
            path='openai/MMMLU',
            hf_revision='325a01dc3e173cac1578df94120499aaca2e2504',
            name=_name,
            reader_cfg=mmmlu_reader_cfg,
            infer_cfg=mmmlu_infer_cfg,
            eval_cfg=mmmlu_eval_cfg,
        ))

del _name
