#!/usr/bin/env python3
"""Stage review-gated Agent Eval catalog changes for OpenCompass benchmarks.

This script never writes the Agent Eval deployment or source tree.  It compares
the canonical submission catalog with frozen OpenCompass protocols, validates
all proposed mutations locally, and (with ``--apply``) submits ``catalog_change``
requests through ``/api/submissions``.  The PAT is read with ``getpass`` when it
is not already present in ``AGENT_EVAL_SUBMISSION_PAT``; it is never accepted on
the command line or stored in receipts.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


TOKEN_ENV = 'AGENT_EVAL_SUBMISSION_PAT'
DEFAULT_PLATFORM_URL = 'https://47.88.93.207'
PRIMARY_METRIC = {'primaryMetric': 'Accuracy', 'metricUnit': 'percent'}


class CatalogSyncError(RuntimeError):
    """A safe, actionable catalog synchronization failure."""


def task(name: str, sample_count: int, capability_id: str) -> dict[str, Any]:
    return {
        'task_key': 'overall',
        'name': name,
        'sample_count': sample_count,
        'difficulty': 'mixed',
        'capability_id': capability_id,
    }


def version(*, name: str, split: str, sample_count: int,
            capability_id: str, dataset_revision: str | None,
            evaluator_hash: str, dataset_uri: str, protocol_url: str,
            config_module: str, protocol: str, is_current: bool,
            checksum: str | None = None, source_sample_count: int | None = None,
            repeats: int = 1, few_shot: int = 0,
            primary_metric: dict[str, str] = PRIMARY_METRIC) -> dict[str, Any]:
    opencompass = {
        'config_module': config_module,
        'protocol': protocol,
        'few_shot': few_shot,
        'repeats_per_source_sample': repeats,
    }
    if source_sample_count is not None:
        opencompass['source_sample_count'] = source_sample_count
    value: dict[str, Any] = {
        'version': name,
        'split': split,
        'is_current': is_current,
        'sample_count': sample_count,
        'dataset_revision': dataset_revision,
        'evaluator_revision': f'opencompass-local:sha256:{evaluator_hash}',
        'protocol_url': protocol_url,
        'dataset_uri': dataset_uri,
        'config': {
            'feishu': dict(primary_metric),
            'opencompass': opencompass,
        },
        'tasks': [task('Overall', sample_count, capability_id)],
    }
    if checksum:
        value['checksum'] = checksum
    return value


def family(*, slug: str, name: str, task_domain: str, description: str,
           homepage_url: str, paper_url: str | None,
           capability_id: str, versions: list[dict[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {
        'slug': slug,
        'name': name,
        'display_name': name,
        'task_domain': task_domain,
        'description': description,
        'homepage_url': homepage_url,
        'primary_capability_id': capability_id,
        'contamination_risk': 'unknown',
        'status': 'active',
        'capabilities': [{
            'capability_id': capability_id,
            'coverage_weight': 1,
            'is_primary': True,
        }],
        'versions': versions,
    }
    if paper_url:
        value['paper_url'] = paper_url
    return value


CAP_BROAD = 'cap_broad_knowledge'
CAP_SCIENCE = 'cap_scientific_reasoning'
CAP_REASONING = 'cap_reasoning'
CAP_MATH = 'cap_mathematical_reasoning'
CAP_CODING = 'cap_coding'
CAP_INSTRUCTION = 'cap_instruction_following'


FAMILIES: list[dict[str, Any]] = [
    family(
        slug='mmlu-pro', name='MMLU-Pro', task_domain='通用知识/高难',
        description='MMLU-Pro under the pinned official 5-shot chain-of-thought protocol.',
        homepage_url='https://github.com/TIGER-AI-Lab/MMLU-Pro',
        paper_url='https://arxiv.org/abs/2406.01574', capability_id=CAP_BROAD,
        versions=[version(
            name='2024-5shot-cot', split='test', sample_count=12032,
            capability_id=CAP_BROAD,
            dataset_revision='b189ec765aa7ed75c8acfea42df31fdae71f97be',
            evaluator_hash='650c689217fa6b7dae63ffbcf671cea39bfcbab9c9adc852ba0c4d57f60a3a30',
            dataset_uri='https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro',
            protocol_url='https://github.com/TIGER-AI-Lab/MMLU-Pro',
            config_module='opencompass.configs.datasets.mmlu_pro.mmlu_pro_5shot_cot_gen',
            protocol='official 5-shot CoT generation; category macro accuracy',
            few_shot=5, is_current=True)],
    ),
    family(
        slug='c-eval', name='C-Eval', task_domain='中文考试',
        description='C-Eval labelled validation split using EvalScope-compatible 5-shot prompts.',
        homepage_url='https://github.com/hkust-nlp/ceval',
        paper_url='https://arxiv.org/abs/2305.08322', capability_id=CAP_BROAD,
        versions=[version(
            name='2023-evalscope-5shot', split='val', sample_count=1346,
            capability_id=CAP_BROAD, dataset_revision=None,
            evaluator_hash='64011010128f1219a3be1f712f3515ae0b59808482febd02a2aee93cb09b0408',
            dataset_uri='http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ceval.zip',
            protocol_url='https://github.com/modelscope/evalscope',
            config_module='opencompass.configs.datasets.ceval.ceval_evalscope_gen',
            protocol='EvalScope-compatible single-context 5-shot; strict 答案：LETTER extraction',
            checksum='md5:3889202085aebd28b596b98bee43e1aa', few_shot=5,
            is_current=True)],
    ),
    family(
        slug='supergpqa', name='SuperGPQA', task_domain='高难综合知识',
        description='Full SuperGPQA test set under its pinned zero-shot generation protocol.',
        homepage_url='https://supergpqa.github.io/',
        paper_url='https://arxiv.org/abs/2502.14739', capability_id=CAP_BROAD,
        versions=[version(
            name='2025-opencompass-zero-shot', split='train', sample_count=26529,
            capability_id=CAP_BROAD,
            dataset_revision='4430d4458112c7d4497fdcf94d7cc223313d6acf',
            evaluator_hash='255f8df5a90cfef0b2a8703293d9a8ed724b80a81eafc2c45a5a19928120f585',
            dataset_uri='https://huggingface.co/datasets/m-a-p/SuperGPQA',
            protocol_url='https://github.com/SuperGPQA/SuperGPQA',
            config_module='opencompass.configs.datasets.supergpqa.supergpqa_gen',
            protocol='official zero-shot prompt and hierarchical macro accuracy',
            is_current=True)],
    ),
    family(
        slug='ifeval', name='IFEval', task_domain='可验证指令遵循',
        description=('Google IFEval under its official zero-shot, rule-based '
                     'strict instruction-following protocol.'),
        homepage_url='https://huggingface.co/datasets/google/IFEval',
        paper_url='https://arxiv.org/abs/2311.07911',
        capability_id=CAP_INSTRUCTION,
        versions=[version(
            name='2023-966cd895-zero-shot', split='train', sample_count=541,
            capability_id=CAP_INSTRUCTION,
            dataset_revision='966cd89545d6b6acfd7638bc708b98261ca58e84',
            evaluator_hash='05300816b9a238af5a9dfd18b4093bb782617ccec775497db7c7d61399441722',
            dataset_uri='https://huggingface.co/datasets/google/IFEval',
            protocol_url=('https://github.com/google-research/google-research/'
                          'tree/master/instruction_following_eval'),
            config_module='opencompass.configs.datasets.IFEval.IFEval_gen',
            protocol=('official zero-shot generation; prompt-level strict '
                      'accuracy'),
            primary_metric={
                'primaryMetric': 'Prompt-Level Strict Accuracy',
                'metricUnit': 'percent',
            },
            is_current=True)],
    ),
    family(
        slug='ifbench', name='IFBench', task_domain='可验证指令遵循泛化',
        description=('AllenAI IFBench single-turn test set under the official '
                     'zero-shot prompt-level loose evaluation protocol.'),
        homepage_url='https://github.com/allenai/IFBench',
        paper_url='https://arxiv.org/abs/2507.02833',
        capability_id=CAP_INSTRUCTION,
        versions=[version(
            name='2025-2e8a48de-zero-shot', split='train', sample_count=300,
            capability_id=CAP_INSTRUCTION,
            dataset_revision='2e8a48de45ff3bf41242f927254ca81b59ca3ae2',
            evaluator_hash='0d3ec1731c1109e2229b43c3886d69695d759e498f138f56c7cf73c67d4c0227',
            dataset_uri='https://huggingface.co/datasets/allenai/IFBench_test',
            protocol_url='https://github.com/allenai/IFBench',
            config_module='opencompass.configs.datasets.IFBench.IFBench_gen',
            protocol=('official single-turn zero-shot generation; prompt-level '
                      'loose accuracy'),
            primary_metric={
                'primaryMetric': 'Prompt-Level Loose Accuracy',
                'metricUnit': 'percent',
            },
            is_current=True)],
    ),
    family(
        slug='mmmlu', name='MMMLU', task_domain='多语言知识',
        description='OpenAI MMMLU across all 14 translated test locales using simple-evals prompting.',
        homepage_url='https://huggingface.co/datasets/openai/MMMLU',
        paper_url=None, capability_id=CAP_BROAD,
        versions=[version(
            name='2024-simple-evals-zero-shot', split='test', sample_count=196588,
            capability_id=CAP_BROAD,
            dataset_revision='325a01dc3e173cac1578df94120499aaca2e2504',
            evaluator_hash='baf6f32c5263eb7a2275f2ee2021ff1db9f5fd2c34838f3df65a92c90036019c',
            dataset_uri='https://huggingface.co/datasets/openai/MMMLU',
            protocol_url='https://github.com/openai/simple-evals',
            config_module='opencompass.configs.datasets.mmmlu.mmmlu_gen_c51a84',
            protocol='simple-evals zero-shot multilingual CoT; 14-locale macro accuracy',
            is_current=True)],
    ),
    family(
        slug='gpqa-diamond', name='GPQA-Diamond', task_domain='研究生科学推理',
        description='GPQA Diamond under OpenAI simple-evals with four seeded option shuffles.',
        homepage_url='https://github.com/idavidrein/gpqa',
        paper_url='https://arxiv.org/abs/2311.12022', capability_id=CAP_SCIENCE,
        versions=[version(
            name='original-2023-simple-evals-4x', split='diamond', sample_count=792,
            source_sample_count=198, repeats=4, capability_id=CAP_SCIENCE,
            dataset_revision='gpqa-diamond-original',
            evaluator_hash='95b7a92c9d03fca7febb81648b879d1b273afc39be56bff132d9b435bc672fa7',
            dataset_uri='https://github.com/idavidrein/gpqa',
            protocol_url='https://github.com/openai/simple-evals',
            config_module='opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece',
            protocol='simple-evals zero-shot CoT; four seeded option permutations',
            checksum='sha256:41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305',
            is_current=True)],
    ),
    family(
        slug='longbench-v2', name='LongBench v2', task_domain='长上下文理解与推理',
        description='503 realistic long-context multiple-choice reasoning tasks.',
        homepage_url='https://longbench2.github.io/',
        paper_url='https://arxiv.org/abs/2412.15204', capability_id=CAP_REASONING,
        versions=[version(
            name='2.0-opencompass-zero-shot', split='train', sample_count=503,
            capability_id=CAP_REASONING,
            dataset_revision='2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9',
            evaluator_hash='b1d64f638fb3a2a5dbee8db1de3bdfda779de9c5dfbaea7ea5635e09cf4854af',
            dataset_uri='https://huggingface.co/datasets/zai-org/LongBench-v2',
            protocol_url='https://github.com/THUDM/LongBench',
            config_module='opencompass.configs.datasets.longbenchv2.longbenchv2_gen_bd9437',
            protocol='official zero-shot multiple-choice prompt and exact answer extraction',
            is_current=True)],
    ),
    family(
        slug='aime', name='AIME', task_domain='竞赛数学',
        description='American Invitational Mathematics Examination final-answer evaluations.',
        homepage_url='https://artofproblemsolving.com/wiki/index.php/American_Invitational_Mathematics_Examination',
        paper_url=None, capability_id=CAP_MATH,
        versions=[
            version(
                name='2024-opencompass', split='test', sample_count=30,
                capability_id=CAP_MATH, dataset_revision='opencompass-aime-2024',
                evaluator_hash='aa42c76e51339c365da4cc9050036553de514fd3eddf3a88f3910e36d7b6de56',
                dataset_uri='http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip',
                protocol_url='https://github.com/open-compass/opencompass',
                config_module='opencompass.configs.datasets.aime2024.aime2024_gen_17d799',
                protocol='OpenCompass zero-shot final-answer math verification',
                checksum='md5:fbe2d0577fc210962a549f8cea1a00c8', is_current=False),
            version(
                name='2025-opencompass', split='test', sample_count=30,
                capability_id=CAP_MATH,
                dataset_revision='a6ad95f611d72cf628a80b58bd0432ef6638f958',
                evaluator_hash='02f2e59c6dfe5fb198d570a75888c4527200f937ba1c0cb1644eea1767cc32fe',
                dataset_uri='https://huggingface.co/datasets/opencompass/AIME2025',
                protocol_url='https://github.com/open-compass/opencompass',
                config_module='opencompass.configs.datasets.aime2025.aime2025_gen',
                protocol='OpenCompass zero-shot final-answer math verification',
                checksum='md5:aa18cd5d2e2de246c5397f5eb1e61004', is_current=False),
            version(
                name='2026-matharena-4x', split='train', sample_count=120,
                source_sample_count=30, repeats=4, capability_id=CAP_MATH,
                dataset_revision='d2de22f3c656b4f56cf8981212186377d1e23bc3',
                evaluator_hash='1937be076aef11c372fe029283a51aefe9ba42a30d11c37e055cd016a3cb1c04',
                dataset_uri='https://huggingface.co/datasets/MathArena/aime_2026',
                protocol_url='https://github.com/eth-sri/matharena',
                config_module='opencompass.configs.datasets.aime2026.aime2026_gen',
                protocol='MathArena final-answer parser; four runs per problem',
                is_current=True),
        ],
    ),
    family(
        slug='hmmt', name='HMMT', task_domain='竞赛数学',
        description='Harvard-MIT Mathematics Tournament final-answer evaluations using MathArena.',
        homepage_url='https://www.hmmt.org/',
        paper_url='https://arxiv.org/abs/2605.00674', capability_id=CAP_MATH,
        versions=[
            version(
                name='2025-02-matharena-4x', split='train', sample_count=120,
                source_sample_count=30, repeats=4, capability_id=CAP_MATH,
                dataset_revision='6fdc4277120810ff75aa22d2d5489b91f7a262a1',
                evaluator_hash='9b7d8e36f819f8a66f2f9e5a5143a0df546a154ebd372fbf3ed319b53fbe4887',
                dataset_uri='https://huggingface.co/datasets/MathArena/hmmt_feb_2025',
                protocol_url='https://github.com/eth-sri/matharena',
                config_module='opencompass.configs.datasets.hmmt_2025.hmmt_2025_matharena_gen',
                protocol='MathArena final-answer parser; four runs per problem',
                is_current=False),
            version(
                name='2025-11-matharena-4x', split='train', sample_count=120,
                source_sample_count=30, repeats=4, capability_id=CAP_MATH,
                dataset_revision='118dbfb45c4c9467c672268ed55166642897aa46',
                evaluator_hash='9b7d8e36f819f8a66f2f9e5a5143a0df546a154ebd372fbf3ed319b53fbe4887',
                dataset_uri='https://huggingface.co/datasets/MathArena/hmmt_nov_2025',
                protocol_url='https://github.com/eth-sri/matharena',
                config_module='opencompass.configs.datasets.hmmt_2025.hmmt_2025_matharena_gen',
                protocol='MathArena final-answer parser; four runs per problem',
                is_current=False),
            version(
                name='2026-02-matharena-4x', split='train', sample_count=132,
                source_sample_count=33, repeats=4, capability_id=CAP_MATH,
                dataset_revision='02fba4f74d8e68e73e66a02d540fd979c05c274c',
                evaluator_hash='1937be076aef11c372fe029283a51aefe9ba42a30d11c37e055cd016a3cb1c04',
                dataset_uri='https://huggingface.co/datasets/MathArena/hmmt_feb_2026',
                protocol_url='https://github.com/eth-sri/matharena',
                config_module='opencompass.configs.datasets.hmmt2026.hmmt2026_gen',
                protocol='MathArena final-answer parser; four runs per problem',
                is_current=True),
        ],
    ),
    family(
        slug='live-code-bench', name='LiveCodeBench', task_domain='code',
        description='LiveCodeBench code-generation release v6 with official execution tests.',
        homepage_url='https://livecodebench.github.io/',
        paper_url='https://arxiv.org/abs/2403.07974', capability_id=CAP_CODING,
        versions=[version(
            name='v6', split='test', sample_count=175,
            capability_id=CAP_CODING,
            dataset_revision='0fe84c3912ea0c4d4a78037083943e8f0c4dd505',
            evaluator_hash='ff4d84e2e145d6b8c6273b2197649dd9f2a7462fb973328bc103f92429f23dc5',
            dataset_uri='https://huggingface.co/datasets/livecodebench/code_generation_lite',
            protocol_url='https://github.com/LiveCodeBench/LiveCodeBench',
            config_module='opencompass.configs.datasets.livecodebench.livecodebench_v6_codegen',
            protocol='release v6, extractor v2, execution pass@1',
            primary_metric={'primaryMetric': 'pass@1', 'metricUnit': 'percent'},
            is_current=True)],
    ),
    family(
        slug='humaneval', name='HumanEval', task_domain='code',
        description=('OpenAI HumanEval using the OpenCompass zero-shot code '
                     'generation prompt and native execution pass@1 scorer.'),
        homepage_url='https://github.com/openai/human-eval',
        paper_url='https://arxiv.org/abs/2107.03374',
        capability_id=CAP_CODING,
        versions=[version(
            name='openai-sample-evals-zero-shot', split='test',
            sample_count=164, capability_id=CAP_CODING,
            dataset_revision=None,
            evaluator_hash='f2bd37e80ff4fd8fce276993b94743911409bb4e7818793ed9317d4b2833ec28',
            dataset_uri='https://huggingface.co/datasets/opencompass/humaneval',
            protocol_url='https://github.com/openai/human-eval',
            config_module=('opencompass.configs.datasets.humaneval.'
                           'humaneval_gen'),
            protocol=('OpenCompass OpenAI sample-evals zero-shot prompt; '
                      'native execution pass@1'),
            primary_metric={
                'primaryMetric': 'pass@1',
                'metricUnit': 'percent',
            },
            is_current=True)],
    ),
    family(
        slug='mmlu-redux-2-0', name='MMLU-Redux-2.0', task_domain='通用知识',
        description='MMLU-Redux 2.0 questions marked error_type=ok under the OpenCompass chat prompt.',
        homepage_url='https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0',
        paper_url='https://arxiv.org/abs/2406.04127', capability_id=CAP_BROAD,
        versions=[version(
            name='2.0-ok-opencompass-chat', split='test', sample_count=5330,
            capability_id=CAP_BROAD,
            dataset_revision='372ea425445d51e1ba1188c56e5e893f8138621f',
            evaluator_hash='b757bb7e77a2f52f734707887e6f83363a9e329e38e9654eeabf3412842ded1c',
            dataset_uri='https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0',
            protocol_url='https://github.com/EleutherAI/lm-evaluation-harness',
            config_module='opencompass.configs.datasets.mmlu_redux.mmlu_redux_gen',
            protocol='error_type=ok filter; system+user OpenCompass chat prompt; first A-D extraction',
            is_current=True)],
    ),
    family(
        slug='mmlu-prox', name='MMLU-ProX', task_domain='多语言高难知识推理',
        description='Full 29-language MMLU-ProX under the official 5-shot CoT protocol.',
        homepage_url='https://github.com/weihao1115/MMLU-ProX',
        paper_url='https://arxiv.org/abs/2503.10497', capability_id=CAP_BROAD,
        versions=[version(
            name='2025.05-full-5shot-cot', split='test', sample_count=341011,
            capability_id=CAP_BROAD,
            dataset_revision='8e6106a6c6ce1c5027e66cc338143cf997b2aa09',
            evaluator_hash='ee6d0b12630b3a796050ea0c678da02a0a4d5d10f362a0de4f5e9547658e0604',
            dataset_uri='https://huggingface.co/datasets/li-lab/MMLU-ProX',
            protocol_url='https://github.com/weihao1115/MMLU-ProX',
            config_module='opencompass.configs.datasets.mmlu_prox.mmlu_prox_5shot_cot_gen',
            protocol='official full 29-language 5-shot CoT; language/category macro accuracy',
            few_shot=5, is_current=True)],
    ),
    family(
        slug='global-piqa', name='Global PIQA', task_domain='多语言常识推理',
        description='Combined Global PIQA non-parallel and parallel variants across all language configs.',
        homepage_url='https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel',
        paper_url='https://arxiv.org/abs/2510.24081', capability_id=CAP_SCIENCE,
        versions=[version(
            name='2026-opencompass-zero-shot', split='test', sample_count=67818,
            capability_id=CAP_SCIENCE,
            dataset_revision='nonparallel@6777742fa3634c0583cda3b7f8a482ea7b1b0937;parallel@b0b18516a8bc2cb1106bce3dd4db32848ca715ea',
            evaluator_hash='eba400738c48ee224474cfcb71d7f59aeb4936ae4790e9d56212bf65ccad78b4',
            dataset_uri='https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel',
            protocol_url='https://arxiv.org/abs/2510.24081',
            config_module='opencompass.configs.datasets.global_piqa.global_piqa_generation',
            protocol='zero-shot; macro by language within variant, then equal macro across variants',
            is_current=True)],
    ),
    family(
        slug='include', name='INCLUDE', task_domain='多语言/低资源',
        description='INCLUDE base-44 zero-shot log-likelihood over 44 official language configurations.',
        homepage_url='https://huggingface.co/datasets/CohereLabs/include-base-44',
        paper_url='https://arxiv.org/abs/2411.19799', capability_id=CAP_BROAD,
        versions=[version(
            name='2024-d2e1f601-zero-shot', split='test', sample_count=22639,
            capability_id=CAP_BROAD,
            dataset_revision='d2e1f6015f67a43c02a9a68db98e2298e2d6a660',
            evaluator_hash='948212d7f82485380023520c34e9147f80f795f8691898cb55fbeded92f9102c',
            dataset_uri='https://huggingface.co/datasets/CohereLabs/include-base-44',
            protocol_url='https://arxiv.org/abs/2411.19799',
            config_module='opencompass.configs.datasets.include.include_base_44_0shot_ppl',
            protocol='official zero-shot continuation log-likelihood; 44-language macro accuracy',
            is_current=True)],
    ),
    family(
        slug='multichallenge', name='MultiChallenge',
        task_domain='multi-turn instruction following',
        description=(
            'Official 273-example MultiChallenge test set scored by the '
            'pinned GPT-4o-2024-08-06 judge and macro-averaged across axes.'),
        homepage_url='https://github.com/ekwinox117/multi-challenge',
        paper_url='https://arxiv.org/abs/2501.17399',
        capability_id=CAP_INSTRUCTION,
        versions=[version(
            name='2025-5ccefcca-gpt4o-judge', split='test',
            sample_count=273, capability_id=CAP_INSTRUCTION,
            dataset_revision='5ccefcca6a39020d66c1383c4e6a809cb07afa33',
            evaluator_hash='e55eedc034416612c0f064d425ae1dbb6ec73e95091037ead21aa9015be7fbcc',
            dataset_uri=(
                'https://raw.githubusercontent.com/ekwinox117/'
                'multi-challenge/5ccefcca6a39020d66c1383c4e6a809cb07afa33/'
                'data/benchmark_questions.jsonl'),
            protocol_url='https://github.com/ekwinox117/multi-challenge',
            config_module=(
                'opencompass.configs.datasets.multichallenge.'
                'multichallenge_gen'),
            protocol=(
                'official multi-turn last-response generation; '
                'GPT-4o-2024-08-06 strict YES/NO judge; axis macro score'),
            is_current=True)],
    ),
    family(
        slug='aa-lcr', name='AA-LCR', task_domain='长上下文多文档推理',
        description='Artificial Analysis Long Context Reasoning with its official equality-judge rubric.',
        homepage_url='https://artificialanalysis.ai/articles/announcing-aa-lcr',
        paper_url=None, capability_id=CAP_REASONING,
        versions=[version(
            name='2025-bdae010', split='test', sample_count=100,
            capability_id=CAP_REASONING,
            dataset_revision='bdae010bbce259820c0e34c1d7cce210d966fb75',
            evaluator_hash='742e70ccae51565b943dec4a5644e3b224026a60140b42c607d1ad0b87aec349',
            dataset_uri='https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR',
            protocol_url='https://artificialanalysis.ai/articles/announcing-aa-lcr',
            config_module='opencompass.configs.datasets.aa_lcr.aa_lcr_gen',
            protocol='official equality rubric judged by Qwen3-235B-A22B-Instruct-2507 at temperature 0',
            is_current=True)],
    ),
]


DEFERRED: dict[str, str] = {}


def catalog_data(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict) or envelope.get('source') != 'd1':
        raise CatalogSyncError('catalog must be a canonical D1 envelope')
    data = envelope.get('data')
    if not isinstance(data, dict):
        raise CatalogSyncError('catalog envelope has no data object')
    for key in ('benchmarks', 'benchmarkVersions'):
        if not isinstance(data.get(key), list):
            raise CatalogSyncError(f'catalog has no {key} array')
    return data


def validate_uri(value: str, path: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise CatalogSyncError(f'{path} must be an absolute HTTP(S) URI')


def validate_family(value: dict[str, Any]) -> None:
    slug = value['slug']
    if not re.fullmatch(r'[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?', slug):
        raise CatalogSyncError(f'invalid benchmark slug: {slug}')
    if not value.get('versions') or len(value['versions']) > 5:
        raise CatalogSyncError(f'{slug}: create needs 1-5 versions')
    if sum(len(item.get('tasks') or []) for item in value['versions']) > 20:
        raise CatalogSyncError(f'{slug}: more than 20 task slices')
    if sum(item.get('is_current') is True for item in value['versions']) != 1:
        raise CatalogSyncError(f'{slug}: exactly one desired version must be current')
    validate_uri(value['homepage_url'], f'{slug}.homepage_url')
    if value.get('paper_url'):
        validate_uri(value['paper_url'], f'{slug}.paper_url')
    seen = set()
    for item in value['versions']:
        key = (item['version'], item['split'])
        if key in seen:
            raise CatalogSyncError(f'{slug}: duplicate version/split {key}')
        seen.add(key)
        validate_uri(item['dataset_uri'], f'{slug}.{key}.dataset_uri')
        validate_uri(item['protocol_url'], f'{slug}.{key}.protocol_url')
        if len(item.get('tasks') or []) != 1:
            raise CatalogSyncError(f'{slug}.{key}: exactly one overall task required')
        if item['tasks'][0]['sample_count'] != item['sample_count']:
            raise CatalogSyncError(f'{slug}.{key}: task/version sample counts differ')


def production_version(value: dict[str, Any]) -> dict[str, Any]:
    return {
        'version': value.get('version'),
        'split': value.get('split'),
        'sample_count': value.get('sampleCount'),
        'dataset_revision': value.get('datasetRevision'),
        'evaluator_revision': value.get('evaluatorRevision'),
        'protocol_url': value.get('protocolUrl'),
        'checksum': value.get('checksum'),
    }


def assert_immutable_match(slug: str, desired: dict[str, Any],
                           current: dict[str, Any]) -> None:
    existing = production_version(current)
    for key in ('sample_count', 'dataset_revision', 'evaluator_revision',
                'protocol_url', 'checksum'):
        expected = desired.get(key)
        if expected is not None and existing.get(key) != expected:
            raise CatalogSyncError(
                f'{slug} {desired["version"]}/{desired["split"]}: existing '
                f'{key}={existing.get(key)!r}, expected {expected!r}; immutable '
                'version collision, choose a new version label')


def build_proposals(envelope: Any) -> list[dict[str, Any]]:
    data = catalog_data(envelope)
    by_slug = {item.get('slug'): item for item in data['benchmarks']}
    if len(by_slug) != len(data['benchmarks']):
        raise CatalogSyncError('production catalog contains duplicate slugs')
    versions_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for item in data['benchmarkVersions']:
        versions_by_benchmark.setdefault(str(item.get('benchmarkId')), []).append(item)

    proposals = []
    for desired in FAMILIES:
        validate_family(desired)
        existing_family = by_slug.get(desired['slug'])
        if existing_family is None:
            proposals.append({'operation': 'create', 'benchmark': desired})
            continue
        if existing_family.get('primaryCapabilityId') != desired['primary_capability_id']:
            raise CatalogSyncError(
                f'{desired["slug"]}: production primary capability '
                f'{existing_family.get("primaryCapabilityId")!r} differs from '
                f'{desired["primary_capability_id"]!r}')
        current_versions = versions_by_benchmark.get(str(existing_family.get('id')), [])
        by_key = {(item.get('version'), item.get('split')): item
                  for item in current_versions}
        missing = []
        for item in desired['versions']:
            current = by_key.get((item['version'], item['split']))
            if current is None:
                missing.append(item)
            else:
                assert_immutable_match(desired['slug'], item, current)
        if missing:
            proposals.append({
                'operation': 'update',
                'benchmark_id': existing_family['id'],
                'capability_mode': 'merge',
                'benchmark': {'versions': missing},
                '_display_name': desired['name'],
                '_slug': desired['slug'],
                '_source_url': desired['homepage_url'],
            })
    return proposals


def canonical_proposal(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith('_')}


def request_for(value: dict[str, Any]) -> dict[str, Any]:
    proposal = canonical_proposal(value)
    benchmark = proposal['benchmark']
    name = value.get('_display_name') or benchmark['name']
    slug = value.get('_slug') or benchmark['slug']
    source_url = value.get('_source_url') or benchmark['homepage_url']
    versions = benchmark['versions']
    current = next((item for item in versions if item.get('is_current')), versions[-1])
    digest = hashlib.sha256(
        json.dumps(proposal, ensure_ascii=False, sort_keys=True,
                   separators=(',', ':')).encode()).hexdigest()[:16]
    return {
        'idempotency_key': f'opencompass.catalog.{proposal["operation"]}.{slug}.{digest}',
        'dry_run': True,
        'submission': {
            'submission_kind': 'catalog_change',
            'provenance': 'internal_private',
            'source_url': source_url,
            'evidence_title': f'OpenCompass benchmark catalog proposal: {name}',
            'benchmark_name': name,
            'benchmark_version': f'{current["version"]} / {current["split"]}',
            'sample_count': current.get('sample_count'),
            'notes': ('Frozen OpenCompass protocol metadata. Review dataset revision, '
                      'split, sample count, evaluator hash, repetitions and primary metric.'),
            'catalog_change': proposal,
        },
    }


def request_json(platform_url: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        f'{platform_url}/api/submissions', data=body, method='POST',
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
        })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode('utf-8')
            result = json.loads(raw or '{}')
    except urllib.error.HTTPError as error:
        raw = error.read().decode('utf-8', errors='replace')
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = {'message': raw[:500]}
        raise CatalogSyncError(
            f'Agent Eval HTTP {error.code}: {detail.get("error")}; '
            f'{detail.get("message")}; issues={detail.get("issues")!r}') from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise CatalogSyncError(f'Agent Eval transport failure: {error}') from error
    if not isinstance(result, dict) or result.get('ok') is not True:
        raise CatalogSyncError(f'Agent Eval returned an invalid receipt: {result!r}')
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-file', required=True, type=Path,
                        help='Export from /api/submissions/catalog')
    parser.add_argument('--platform-url', default=DEFAULT_PLATFORM_URL)
    parser.add_argument('--output', type=Path,
                        default=Path('/data2/liyulong/tmp/agent-eval-catalog-change-plan.json'))
    parser.add_argument('--apply', action='store_true',
                        help='Dry-run every request, then create pending reviews')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        envelope = json.loads(args.catalog_file.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogSyncError(f'cannot read catalog file: {args.catalog_file}') from error
    parsed = urlsplit(args.platform_url.rstrip('/'))
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise CatalogSyncError('--platform-url must be an absolute HTTP(S) URL')
    platform_url = args.platform_url.rstrip('/')
    proposals = build_proposals(envelope)
    requests = [request_for(value) for value in proposals]
    plan = {
        'catalog_generated_at': envelope.get('generatedAt'),
        'platform_url': platform_url,
        'proposal_count': len(requests),
        'deferred': DEFERRED,
        'requests': requests,
    }
    atomic_json(args.output, plan)
    print(f'Catalog plan: {len(requests)} proposal(s) -> {args.output}')
    if DEFERRED:
        print('Deferred: ' + ', '.join(DEFERRED))
    if not args.apply:
        print('No network writes performed. Re-run with --apply after reviewing the plan.')
        return 0
    if not requests:
        print('Catalog is already aligned; nothing to submit.')
        return 0
    token = os.environ.get(TOKEN_ENV, '').strip()
    if not token:
        token = getpass.getpass('Agent Eval submission PAT: ').strip()
    if not token:
        raise CatalogSyncError('submission PAT is required')

    dry_receipts = []
    for index, payload in enumerate(requests, 1):
        receipt = request_json(platform_url, token, dict(payload, dry_run=True))
        if receipt.get('dry_run') is not True or not isinstance(receipt.get('normalized'), dict):
            raise CatalogSyncError(f'dry-run {index} returned an invalid receipt')
        dry_receipts.append(receipt)
        print(f'Dry-run {index}/{len(requests)} passed')
    apply_receipts = []
    for index, payload in enumerate(requests, 1):
        receipt = request_json(platform_url, token, dict(payload, dry_run=False))
        if receipt.get('status') != 'pending_review':
            raise CatalogSyncError(f'apply {index} did not create a pending review')
        apply_receipts.append(receipt)
        print(f'Applied {index}/{len(requests)}: review={receipt.get("review_id")}')
    receipt_path = args.output.with_name(args.output.stem + '-receipts.json')
    atomic_json(receipt_path, {
        'platform_url': platform_url,
        'dry_run_receipts': dry_receipts,
        'apply_receipts': apply_receipts,
    })
    print(f'Receipts: {receipt_path}')
    print('All changes are pending review; no benchmark is canonical until approved and promoted.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except CatalogSyncError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2)
