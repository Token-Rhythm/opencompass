from mmengine.config import read_base

with read_base():
    from .groups.mmlu_prox import (languages, lite_category_sizes,
                                   mmlu_prox_lite_0shot_summary_groups)

summarizer = dict(dataset_abbrs=[
    f'mmlu_prox_lite_0shot_{lang}_{category}' for lang in languages
    for category in lite_category_sizes
] + [f'mmlu_prox_lite_0shot_{lang}' for lang in languages],
                  summary_groups=mmlu_prox_lite_0shot_summary_groups)
