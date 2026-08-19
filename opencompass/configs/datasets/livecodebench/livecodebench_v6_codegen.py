from opencompass.datasets import (LCBCodeGenerationDataset,
                                  LCBCodeGenerationEvaluator)
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# Official v6-only data (175 problems), pinned to the current upstream commit.
# It is intentionally loaded as JSONL instead of executing the dataset repo's
# custom Python loading script.
LCB_V6_DATA_FILE = (
    'https://huggingface.co/datasets/livecodebench/code_generation_lite/'
    'resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl')

lcb_codegen_reader_cfg = dict(
    input_columns=['question_content', 'format_prompt'],
    output_column='question_id',
)

prompt_template = (
    '### Question:\n{question_content}\n\n{format_prompt}'
    '### Answer: (use the provided format with backticks)\n\n')

lcb_codegen_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt=(
                        'You are an expert Python programmer. You will be '
                        'given a question (problem specification) and will '
                        'generate a correct Python program that matches the '
                        'specification and passes all tests.'),
                ),
            ],
            round=[dict(role='HUMAN', prompt=prompt_template)],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ParallelGenInferencer, save_every=1),
)

lcb_codegen_eval_cfg = dict(
    evaluator=dict(
        type=LCBCodeGenerationEvaluator,
        release_version='v6',
        data_file=LCB_V6_DATA_FILE,
        extractor_version='v2',
        num_process_evaluate=16,
        timeout=6,
    ),
    pred_role='BOT',
)

livecodebench_v6_codegen_datasets = [
    dict(
        type=LCBCodeGenerationDataset,
        abbr='livecodebench_v6_codegen',
        path='livecodebench/code_generation_lite',
        release_version='v6',
        data_file=LCB_V6_DATA_FILE,
        reader_cfg=lcb_codegen_reader_cfg,
        infer_cfg=lcb_codegen_infer_cfg,
        eval_cfg=lcb_codegen_eval_cfg,
    )
]
