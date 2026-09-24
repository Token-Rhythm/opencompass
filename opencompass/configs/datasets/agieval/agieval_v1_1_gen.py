"""AGIEval v1.1, zero-shot, one model response per question."""

from opencompass.datasets.agieval.agieval_v1_1_cloze import AGIEvalV11ClozeEvaluator
from opencompass.datasets.agieval.agieval_v1_1 import AGIEvalV11Dataset
from opencompass.datasets.agieval.agieval_v1_1_postprocess import agieval_mathqa_postprocess
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets.agieval.agieval_v1_1_postprocess import agieval_single_choice_postprocess

_tasks = [
    'gaokao-chinese', 'gaokao-english', 'gaokao-geography',
    'gaokao-history', 'gaokao-biology', 'gaokao-chemistry',
    'gaokao-physics', 'gaokao-mathqa', 'logiqa-zh',
    'lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math',
    'sat-en', 'sat-en-without-passage', 'aqua-rat', 'jec-qa-kd',
    'jec-qa-ca', 'gaokao-mathcloze', 'math',
]
_cloze = ['gaokao-mathcloze', 'math']
_english_mc = [
    'lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math',
    'sat-en', 'sat-en-without-passage', 'aqua-rat',
]
_group = 'agieval_v1_1_zero_shot_chat'
agieval_v1_1_datasets = []
for _name in _tasks:
    if _name in _cloze:
        _evaluation = dict(evaluator=dict(type=AGIEvalV11ClozeEvaluator))
    elif _name == 'gaokao-mathqa':
        _evaluation = dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=agieval_mathqa_postprocess))
    else:
        _evaluation = dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=agieval_single_choice_postprocess,
                                    options='ABCDE'))
    agieval_v1_1_datasets.append(dict(
        type=AGIEvalV11Dataset, path='./data/AGIEval/data/v1_1',
        name=_name, setting_name='zero-shot', chat_mode=True,
        abbr=f'{_group}_{_name}',
        reader_cfg=dict(input_columns=['messages'], output_column='label',
                        train_split='test', test_split='test'),
        infer_cfg=dict(
            prompt_template=dict(
                type=RawPromptTemplate,
                messages=[
                    dict(role='system',
                         content='You are a helpful AI assistant.'),
                    dict(expand_column='messages'),
                ]),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=4096,
                            save_every=1)),
        eval_cfg=_evaluation,
    ))

_chinese_mc = [name for name in _tasks
               if name not in _english_mc and name not in _cloze]
agieval_v1_1_summary_groups = [
    dict(name=_group, subsets=[f'{_group}_{name}' for name in _tasks]),
    dict(name=f'{_group}_en',
         subsets=[f'{_group}_{name}' for name in _english_mc]),
    dict(name=f'{_group}_zh',
         subsets=[f'{_group}_{name}' for name in _chinese_mc]),
    dict(name=f'{_group}_cloze',
         subsets=[f'{_group}_{name}' for name in _cloze]),
]
del _tasks, _cloze, _english_mc, _chinese_mc, _group, _name, _evaluation
