from opencompass.datasets import PolyMathDataset, PolyMathEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

languages = ('en', 'zh', 'ar', 'bn', 'de', 'es', 'fr', 'id', 'it', 'ja',
             'ko', 'ms', 'pt', 'ru', 'sw', 'te', 'th', 'vi')
levels = ('low', 'medium', 'high', 'top')

polymath_datasets = []
for lang in languages:
    for level in levels:
        reader_cfg = dict(input_columns=['prompt'],
                          output_column='answer',
                          test_split='test')
        infer_cfg = dict(
            prompt_template=dict(type=PromptTemplate, template='{prompt}'),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer,
                            max_seq_len=65536,
                            max_out_len=65536))
        eval_cfg = dict(evaluator=dict(type=PolyMathEvaluator))
        polymath_datasets.append(
            dict(abbr=f'polymath_{lang}_{level}',
                 type=PolyMathDataset,
                 lang=lang,
                 level=level,
                 reader_cfg=reader_cfg,
                 infer_cfg=infer_cfg,
                 eval_cfg=eval_cfg))
