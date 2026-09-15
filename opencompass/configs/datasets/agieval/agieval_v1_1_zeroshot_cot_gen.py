"""AGIEval v1.1: zero-shot-CoT, chat_mode=True."""

from mmengine.config import read_base

with read_base():
    from .agieval_v1_1_gen import (agieval_v1_1_datasets,
                                   agieval_v1_1_summary_groups)

_old = 'agieval_v1_1_zero_shot_chat'
_new = 'agieval_v1_1_zero_shot_cot_chat'
for _dataset in agieval_v1_1_datasets:
    _dataset['setting_name'] = 'zero-shot-CoT'
    _dataset['chat_mode'] = True
    _dataset['abbr'] = _dataset['abbr'].replace(_old, _new, 1)
for _group in agieval_v1_1_summary_groups:
    _group['name'] = _group['name'].replace(_old, _new, 1)
    _group['subsets'] = [name.replace(_old, _new, 1)
                         for name in _group['subsets']]
del _old, _new, _dataset, _group
