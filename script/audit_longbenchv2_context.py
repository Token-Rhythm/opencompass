#!/usr/bin/env python3
"""Audit LongBench v2 prompts against an OpenCompass API context budget.

The official LongBench v2 runner keeps the beginning and end when a prompt is
longer than the configured input window. This script tokenizes the exact local
prompt and reports which samples require that official-style mid truncation
after reserving the requested generation and protocol overhead budgets.
"""

import argparse
import json
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer


REVISION = '2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'
PROMPT = (
    'Please read the following text and answer the questions below.\n'
    ' <text> \n {context} \n </text> \n \n What is the correct answer to '
    'this question: {question} \n \n Choices: \n (A) {choice_A} \n '
    '(B) {choice_B} \n (C) {choice_C} \n (D) {choice_D} \n Let’s '
    'think step by step. Based on the above, what is the single, most likely '
    'answer choice? Format your response as follows: "The correct answer is '
    '(insert answer here)".'
)


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='zai-org/LongBench-v2')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--tokenizer', default='Qwen/Qwen3.5-2B')
    parser.add_argument('--max-seq-len', type=int, default=65536)
    parser.add_argument('--max-out-len', type=int, default=8192)
    # OpenAISDK reserves 100 tokens for chat-template and protocol overhead.
    parser.add_argument('--overhead-reserve', type=int, default=100)
    args = parser.parse_args()

    if args.max_seq_len <= 0 or args.max_out_len <= 0:
        parser.error('context and output limits must be positive')
    input_budget = (args.max_seq_len - args.max_out_len -
                    args.overhead_reserve)
    if input_budget <= 0:
        parser.error('max-out-len must be smaller than max-seq-len')

    dataset = load_dataset(args.dataset,
                           split='train',
                           revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    token_lengths = []
    over_input_budget = []
    over_context_alone = []
    length_labels = Counter()
    for index, row in enumerate(dataset):
        prompt = PROMPT.format(**row)
        length = len(tokenizer.encode(prompt, add_special_tokens=False))
        token_lengths.append(length)
        length_labels[str(row.get('length', 'unknown'))] += 1
        if length > input_budget:
            over_input_budget.append(index)
        if length > args.max_seq_len:
            over_context_alone.append(index)

    ordered = sorted(token_lengths)
    report = {
        'dataset': args.dataset,
        'revision': args.revision,
        'tokenizer': args.tokenizer,
        'samples': len(token_lengths),
        'max_seq_len': args.max_seq_len,
        'max_out_len': args.max_out_len,
        'overhead_reserve': args.overhead_reserve,
        'input_budget': input_budget,
        'token_lengths': {
            'min': ordered[0] if ordered else 0,
            'p50': percentile(ordered, 0.50),
            'p90': percentile(ordered, 0.90),
            'p95': percentile(ordered, 0.95),
            'p99': percentile(ordered, 0.99),
            'max': ordered[-1] if ordered else 0,
        },
        'length_labels': dict(sorted(length_labels.items())),
        'over_input_budget_count': len(over_input_budget),
        'over_input_budget_indices': over_input_budget,
        'over_context_alone_count': len(over_context_alone),
        'over_context_alone_indices': over_context_alone,
        'requires_mid_truncation': bool(over_input_budget),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
