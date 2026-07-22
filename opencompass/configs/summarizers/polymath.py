from mmengine.config import read_base

with read_base():
    from .groups.polymath import (languages, level_weights,
                                  polymath_summary_groups)

summarizer = dict(
    dataset_abbrs=[
        f'polymath_{lang}_{level}' for lang in languages
        for level in level_weights
    ] + [f'polymath_{lang}' for lang in languages] + ['polymath'],
    summary_groups=polymath_summary_groups)
