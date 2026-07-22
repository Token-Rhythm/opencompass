languages = ('en', 'zh', 'ar', 'bn', 'de', 'es', 'fr', 'id', 'it', 'ja',
             'ko', 'ms', 'pt', 'ru', 'sw', 'te', 'th', 'vi')
level_weights = {'low': 1, 'medium': 2, 'high': 4, 'top': 8}

polymath_summary_groups = []
for lang in languages:
    subsets = [f'polymath_{lang}_{level}' for level in level_weights]
    weights = {
        f'polymath_{lang}_{level}': weight
        for level, weight in level_weights.items()
    }
    polymath_summary_groups.append(
        dict(name=f'polymath_{lang}', subsets=subsets, weights=weights))

polymath_summary_groups.append(
    dict(name='polymath', subsets=[f'polymath_{lang}' for lang in languages]))
