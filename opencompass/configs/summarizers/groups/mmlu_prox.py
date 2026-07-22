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

mmlu_prox_summary_groups = []
for lang in languages:
    subsets = [f'mmlu_prox_{lang}_{category}' for category in category_sizes]
    weights = {
        f'mmlu_prox_{lang}_{category}': size
        for category, size in category_sizes.items()
    }
    mmlu_prox_summary_groups.append(
        dict(name=f'mmlu_prox_{lang}', subsets=subsets, weights=weights))
