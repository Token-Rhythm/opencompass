from mmengine.config import read_base

with read_base():
    from .groups.mmlu_prox import (category_sizes, languages,
                                   mmlu_prox_summary_groups)

summarizer = dict(dataset_abbrs=[
    f'mmlu_prox_{lang}_{category}' for lang in languages
    for category in category_sizes
] + [f'mmlu_prox_{lang}' for lang in languages],
                  summary_groups=mmlu_prox_summary_groups)
