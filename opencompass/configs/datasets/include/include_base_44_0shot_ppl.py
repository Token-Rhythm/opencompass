from opencompass.datasets import INCLUDEDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

languages = ('Albanian', 'Arabic', 'Armenian', 'Azerbaijani', 'Basque',
             'Belarusian', 'Bengali', 'Bulgarian', 'Chinese', 'Croatian',
             'Dutch', 'Estonian', 'Finnish', 'French', 'Georgian', 'German',
             'Greek', 'Hebrew', 'Hindi', 'Hungarian', 'Indonesian', 'Italian',
             'Japanese', 'Kazakh', 'Korean', 'Lithuanian', 'Malay',
             'Malayalam', 'Nepali', 'North Macedonian', 'Persian', 'Polish',
             'Portuguese', 'Russian', 'Serbian', 'Spanish', 'Tagalog', 'Tamil',
             'Telugu', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese')

include_datasets = []
for lang in languages:
    reader_cfg = dict(input_columns=['prompt'],
                      output_column='answer',
                      test_split='test')
    infer_cfg = dict(prompt_template=dict(type=PromptTemplate,
                                          template={
                                              0: '{prompt} A',
                                              1: '{prompt} B',
                                              2: '{prompt} C',
                                              3: '{prompt} D',
                                          }),
                     retriever=dict(type=ZeroRetriever),
                     inferencer=dict(type=PPLInferencer))
    eval_cfg = dict(evaluator=dict(type=AccEvaluator))
    include_datasets.append(
        dict(abbr=f'include_base_44_{lang.lower().replace(" ", "_")}',
             type=INCLUDEDataset,
             lang=lang,
             reader_cfg=reader_cfg,
             infer_cfg=infer_cfg,
             eval_cfg=eval_cfg))
