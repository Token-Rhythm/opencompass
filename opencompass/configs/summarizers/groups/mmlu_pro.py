categories = ['math', 'physics', 'chemistry', 'law', 'engineering', 'other', 'economics', 'health', 'psychology', 'business', 'biology', 'philosophy', 'computer science', 'history']

category_sizes = {
    'math': 1351,
    'physics': 1299,
    'chemistry': 1132,
    'law': 1101,
    'engineering': 969,
    'other': 924,
    'economics': 844,
    'health': 818,
    'psychology': 798,
    'business': 789,
    'biology': 717,
    'philosophy': 499,
    'computer_science': 410,
    'history': 381,
}

subsets = ['mmlu_pro_' + c.replace(' ', '_') for c in categories]

mmlu_pro_summary_groups = [
    {
        'name': 'mmlu_pro',
        'subsets': subsets,
        'weights': {
            subset: category_sizes[category.replace(' ', '_')]
            for subset, category in zip(subsets, categories)
        },
    },
]
