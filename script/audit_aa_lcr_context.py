#!/usr/bin/env python3
"""Audit AA-LCR prompts against an OpenCompass API context budget."""

import argparse
import json

from transformers import AutoTokenizer

from opencompass.datasets.aa_lcr import (AA_LCR_PATH, AA_LCR_REVISION,
                                         AALCRDataset)


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0
    return sorted_values[round((len(sorted_values) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=AA_LCR_PATH)
    parser.add_argument('--revision', default=AA_LCR_REVISION)
    parser.add_argument('--tokenizer', default='Qwen/Qwen3.5-2B')
    parser.add_argument('--max-seq-len', type=int, default=262144)
    parser.add_argument('--max-out-len', type=int, default=32768)
    # OpenAISDK reserves 100 tokens for chat-template and protocol overhead.
    parser.add_argument('--overhead-reserve', type=int, default=100)
    args = parser.parse_args()

    if min(args.max_seq_len, args.max_out_len) <= 0:
        parser.error('context and output limits must be positive')
    input_budget = (args.max_seq_len - args.max_out_len -
                    args.overhead_reserve)
    if input_budget <= 0:
        parser.error('output and overhead must be smaller than context')

    rows = AALCRDataset.load(path=args.dataset,
                             revision=args.revision)['test']
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                              trust_remote_code=True)
    token_lengths = [
        len(tokenizer.encode(row['prompt'], add_special_tokens=False))
        for row in rows
    ]
    ordered = sorted(token_lengths)
    over_input_budget = [
        index for index, length in enumerate(token_lengths)
        if length > input_budget
    ]
    over_context = [
        index for index, length in enumerate(token_lengths)
        if length > args.max_seq_len
    ]
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
        'over_input_budget_count': len(over_input_budget),
        'over_input_budget_indices': over_input_budget,
        'over_context_alone_count': len(over_context),
        'over_context_alone_indices': over_context,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
