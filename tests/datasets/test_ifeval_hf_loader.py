from datasets import Dataset

from opencompass.datasets.IFEval import ifeval
from opencompass.datasets.IFEval.instructions import ResponseLanguageChecker


def test_ifeval_hf_loader_preserves_reference(monkeypatch):
    source = Dataset.from_list([
        {
            'key': '0',
            'prompt': 'Follow this instruction.',
            'instruction_id_list': ['length_constraints:number_words'],
            'kwargs': [{'num_words': 3}],
        }
    ])
    called = {}

    def fake_load_dataset(path, split, revision):
        called.update(path=path, split=split, revision=revision)
        return source

    monkeypatch.setattr(ifeval, 'load_dataset', fake_load_dataset)
    loaded = ifeval.IFEvalDataset.load(
        'google/IFEval', hf_revision='pinned', hf_split='train')

    assert called == {
        'path': 'google/IFEval',
        'split': 'train',
        'revision': 'pinned',
    }
    assert loaded[0]['prompt'] == source[0]['prompt']
    assert loaded[0]['reference'] == source[0]


def test_ifeval_language_constraint_has_official_dependency():
    checker = ResponseLanguageChecker('language:response_language')
    checker.build_description(language='en')

    assert checker.check_following(
        'This response is written entirely in English for the evaluation.')
    assert not checker.check_following('这是一段完全使用中文写成的回复。')
