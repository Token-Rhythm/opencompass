from types import SimpleNamespace

from opencompass.datasets.IFBench import ifbench
from opencompass.datasets.IFBench import instructions
from opencompass.datasets.IFEval.ifeval import IFEvaluator


def _evaluation_result(followed):
    return SimpleNamespace(
        follow_instruction_list=[followed],
        instruction_id_list=['test:instruction'],
    )


def test_ifbench_headline_score_is_prompt_level_loose(monkeypatch):
    monkeypatch.setattr(
        ifbench, 'test_instruction_following_strict',
        lambda *_: _evaluation_result(False))
    monkeypatch.setattr(
        ifbench, 'test_instruction_following_loose',
        lambda *_: _evaluation_result(True))
    reference = {
        'key': 1,
        'instruction_id_list': ['test:instruction'],
        'prompt': 'prompt',
        'kwargs': [{}],
    }

    scores = ifbench.IFBenchEvaluator().score(
        ['response'], [reference], ['prompt'])

    assert scores['score'] == 100
    assert scores['Prompt-level-loose-accuracy'] == 100
    assert scores['Prompt-level-strict-accuracy'] == 0
    assert scores['average_4_metrics'] == 50


def test_instruction_following_evaluators_reject_length_mismatch():
    expected_error = {
        'error': 'predictions, references, and origin_prompt have different '
                 'lengths'
    }
    assert ifbench.IFBenchEvaluator().score([], [{}], ['prompt']) == expected_error
    assert IFEvaluator().score([], [{}], ['prompt']) == expected_error


def test_ifbench_position_and_character_span_checks_match_official():
    position = instructions.KeywordSpecificPositionChecker(
        'words:keywords_specific_position')
    position.build_description(keyword='target', n=1, m=2)
    assert position.check_following('First, TARGET comes here.')

    edge_position = instructions.WordsPositionChecker('words:words_position')
    edge_position.build_description(keyword='echo')
    assert edge_position.check_following('Start ECHO echo.')

    span = instructions.RepeatSpanChecker('repeat:repeat_span')
    span.build_description(prompt_to_repeat='abcdef', n_start=1, n_end=3)
    assert span.check_following('bcd')


def test_ifbench_official_optional_constraints_are_runnable():
    emoji_checker = instructions.EmojiSentenceChecker('format:emoji')
    emoji_checker.build_description()
    assert emoji_checker.check_following('Done! 😊')

    syllable_checker = instructions.AlternateParitySyllablesChecker(
        'words:odd_even_syllables')
    syllable_checker.build_description()
    assert isinstance(syllable_checker.check_following('cat banana'), bool)
