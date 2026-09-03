"""Official-prompt SciCode with scientist-annotated background.

This config is independent of the legacy ``scicode_gen`` and
``scicode_wbg_gen`` configs.
"""

from opencompass.datasets.scicode_official import (
    OfficialExtractionSciCodeEvaluator,
    OfficialSciCodeDataset,
    OfficialSciCodeWithBackgroundInferencer,
)
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

SciCode_official_reader_cfg = {
    "input_columns": ["problem_id"],
    "output_column": None,
}

SciCode_official_infer_cfg = {
    "ice_template": {"type": PromptTemplate, "template": ""},
    "retriever": {"type": ZeroRetriever},
    "inferencer": {
        "type": OfficialSciCodeWithBackgroundInferencer,
        "save_every": 1,
        # Each problem remains sequential; different problems may run in
        # parallel, matching the dependency structure of the official runner.
        "max_infer_workers": 8,
        "official_prompt_template": "./data/scicode_official/multistep_template.txt",
        "official_special_code_dir": "./data/scicode_official/special_steps",
    },
}

SciCode_official_eval_cfg = {
    "evaluator": {
        # Only answer extraction is aligned with upstream. Test construction,
        # isolation and the default 120-second timeout reuse the local scorer.
        "type": OfficialExtractionSciCodeEvaluator,
        "dataset_path": "./data/scicode",
        "with_bg": True,
    }
}

SciCode_datasets = [
    {
        "abbr": "SciCode_official_with_background_sandboxed",
        "type": OfficialSciCodeDataset,
        "path": "./data/scicode_official",
        "with_background": True,
        "reader_cfg": SciCode_official_reader_cfg,
        "infer_cfg": SciCode_official_infer_cfg,
        "eval_cfg": SciCode_official_eval_cfg,
    }
]
