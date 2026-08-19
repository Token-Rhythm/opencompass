import json

from opencompass.datasets.livecodebench.evaluator import \
    codegen_check_correctness


def test_codegen_runtime_fallback_call_based():
    sample = {
        'input_output': json.dumps({
            'fn_name': 'add',
            'inputs': ['1\n2', '40\n2'],
            'outputs': ['3', '42'],
        })
    }

    results, metadata = codegen_check_correctness(
        sample, 'def add(a, b):\n    return a + b', timeout=2, debug=False)

    assert results == [True, True]
    assert metadata == {}


def test_codegen_runtime_fallback_standard_input():
    sample = {
        'input_output': json.dumps({
            'fn_name': None,
            'inputs': ['2 3'],
            'outputs': ['5'],
        })
    }
    program = 'a, b = map(int, input().split())\nprint(a + b)'

    results, metadata = codegen_check_correctness(
        sample, program, timeout=2, debug=False)

    assert results == [True]
    assert metadata == {}
