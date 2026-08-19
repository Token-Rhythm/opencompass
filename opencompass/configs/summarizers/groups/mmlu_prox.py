languages = ('en', 'ja', 'zh', 'ko', 'fr', 'de', 'es', 'pt', 'zu', 'sw', 'wo',
             'yo', 'th', 'ar', 'hi', 'bn', 'mr', 'ne', 'af', 'te', 'ur', 'ru',
             'id', 'vi', 'cs', 'hu', 'it', 'sr', 'uk')
category_sizes = {
    'biology': 717,
    'business': 789,
    'chemistry': 1132,
    'computer_science': 410,
    'economics': 844,
    'engineering': 969,
    'health': 687,
    'history': 381,
    'law': 959,
    'math': 1351,
    'other': 924,
    'philosophy': 499,
    'physics': 1299,
    'psychology': 798,
}

lite_category_sizes = {
    'biology': 36,
    'business': 40,
    'chemistry': 56,
    'computer_science': 20,
    'economics': 42,
    'engineering': 48,
    'health': 35,
    'history': 19,
    'law': 48,
    'math': 68,
    'other': 46,
    'philosophy': 25,
    'physics': 65,
    'psychology': 40,
}


def build_mmlu_prox_summary_groups(dataset_prefix, group_prefix,
                                   dataset_category_sizes):
    groups = []
    for lang in languages:
        subsets = [
            f'{dataset_prefix}_{lang}_{category}'
            for category in dataset_category_sizes
        ]
        weights = {
            f'{dataset_prefix}_{lang}_{category}': size
            for category, size in dataset_category_sizes.items()
        }
        groups.append(
            dict(name=f'{group_prefix}_{lang}',
                 subsets=subsets,
                 weights=weights))
    return groups


mmlu_prox_summary_groups = build_mmlu_prox_summary_groups(
    'mmlu_prox_5shot', 'mmlu_prox_5shot', category_sizes)
mmlu_prox_summary_groups.append(
    dict(
        name='mmlu_prox',
        subsets=[
            f'mmlu_prox_5shot_{lang}_{category}'
            for lang in languages for category in category_sizes
        ],
        # Every language contains the same category sizes, so these weights
        # are exactly equivalent to averaging the 29 language accuracies.
        weights={
            f'mmlu_prox_5shot_{lang}_{category}': size
            for lang in languages for category, size in category_sizes.items()
        },
    ))
mmlu_prox_0shot_summary_groups = build_mmlu_prox_summary_groups(
    'mmlu_prox_0shot', 'mmlu_prox_0shot', category_sizes)
mmlu_prox_lite_summary_groups = build_mmlu_prox_summary_groups(
    'mmlu_prox_lite_5shot', 'mmlu_prox_lite_5shot', lite_category_sizes)
mmlu_prox_lite_0shot_summary_groups = build_mmlu_prox_summary_groups(
    'mmlu_prox_lite_0shot', 'mmlu_prox_lite_0shot', lite_category_sizes)
