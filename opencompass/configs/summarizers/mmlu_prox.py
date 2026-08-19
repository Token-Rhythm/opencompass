from mmengine.config import read_base

with read_base():
    from .groups.mmlu_prox import (category_sizes, languages,
                                   mmlu_prox_summary_groups)

# Keep the official weighted overall visible in the CSV as well as in the
# internal group metrics.
summarizer = dict(dataset_abbrs=[
    f'mmlu_prox_5shot_{lang}_{category}' for lang in languages
    for category in category_sizes
] + [f'mmlu_prox_5shot_{lang}' for lang in languages] + ['mmlu_prox'],
                  summary_groups=mmlu_prox_summary_groups)
