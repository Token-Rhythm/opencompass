from opencompass.datasets import MMLUReduxDataset, MMLUReduxEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

mmlu_redux_reader_cfg = dict(input_columns=['prompt'],
                             output_column='answer_letter',
                             test_split='test')
mmlu_redux_infer_cfg = dict(prompt_template=dict(type=PromptTemplate,
                                                 template='{prompt}'),
                            retriever=dict(type=ZeroRetriever),
                            inferencer=dict(type=GenInferencer,
                                            max_out_len=256,
                                            stopping_criteria=['</s>']))
mmlu_redux_eval_cfg = dict(evaluator=dict(type=MMLUReduxEvaluator))

mmlu_redux_datasets = [
    dict(abbr='mmlu_redux',
         type=MMLUReduxDataset,
         reader_cfg=mmlu_redux_reader_cfg,
         infer_cfg=mmlu_redux_infer_cfg,
         eval_cfg=mmlu_redux_eval_cfg)
]
