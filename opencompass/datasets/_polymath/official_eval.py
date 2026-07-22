"""Port of QwenLM/PolyMath's official ``math_equal`` routine.

The control flow and final LaTeX parsing fallback are fixed to
QwenLM/PolyMath@fbf4e41cae78687d6be7447dbea897357c06aaa7.
"""

import math
import multiprocessing
import re

import regex


def _parse_number(value):
    # Keep this deliberately narrow.  In particular, the official evaluator
    # does not remove LaTeX delimiters or whitespace before parsing a number.
    value = regex.sub(',', '', str(value))
    try:
        return float(value)
    except ValueError:
        if value.endswith('%'):
            try:
                return float(value[:-1].rstrip('\\')) / 100
            except ValueError:
                pass
    return None


def _choice_answer(value):
    value = str(value).strip('\n').rstrip('.').rstrip('/').strip().lstrip(':')
    matches = re.findall(r'\b([A-E])\b', value.upper())
    value = matches[-1] if matches else value.strip().strip('.')
    return value.rstrip('.').rstrip('/')


def _str_to_pmatrix(value):
    matrices = re.findall(r'\{.*,.*\}', str(value).strip())
    return ', '.join(r'\begin{pmatrix}' +
                     matrix.strip('{}').replace(',', '\\') + r'\end{pmatrix}'
                     for matrix in matrices)


def _symbolic_equal(left, right):
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    from .latex2sympy2.latex2sympy2 import latex2sympy

    def parse(value):
        for parser in (parse_latex, parse_expr, latex2sympy):
            try:
                return parser(value.replace('\\\\', '\\'))
            except Exception:
                try:
                    return parser(value)
                except Exception:
                    pass
        return value

    left, right = parse(left), parse(right)
    try:
        if str(left) == str(right) or left == right:
            return True
    except Exception:
        pass
    try:
        if left.equals(right) or simplify(left - right) == 0:
            return True
    except Exception:
        pass
    try:
        if abs(left.lhs - left.rhs).equals(abs(right.lhs - right.rhs)):
            return True
    except Exception:
        pass
    try:
        return math.isclose(float(N(left)), float(N(right)), rel_tol=1e-4)
    except Exception:
        pass
    try:
        if left.shape == right.shape:
            return left.applyfunc(lambda item: round(item, 3)).equals(
                right.applyfunc(lambda item: round(item, 3)))
    except Exception:
        return False
    return False


def _symbolic_worker(left, right, queue):
    queue.put(_symbolic_equal(left, right))


def _symbolic_with_timeout(left, right, timeout=1):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_symbolic_worker,
                                      args=(left, right, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return False
    return False if queue.empty() else bool(queue.get())


def math_equal(prediction,
               reference,
               include_percentage=True,
               is_close=True,
               timeout=True):
    """Compare boxed answers in the same order as the official checker."""
    if prediction is None or reference is None:
        return False
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True
    if reference in {'A', 'B', 'C', 'D', 'E'
                     } and _choice_answer(prediction) == reference:
        return True

    prediction_number = _parse_number(prediction)
    reference_number = _parse_number(reference)
    if prediction_number is not None and reference_number is not None:
        candidates = (
            [reference_number / 100, reference_number, reference_number *
             100] if include_percentage else [reference_number])
        for candidate in candidates:
            if (math.isclose(prediction_number, candidate, rel_tol=1e-4)
                    if is_close else prediction_number == candidate):
                return True
        return False

    prediction, reference = str(prediction).strip(), str(reference).strip()
    if not prediction:
        return False

    if 'pmatrix' in prediction and 'pmatrix' not in reference:
        reference = _str_to_pmatrix(reference)

    pred_flat, ref_flat = prediction, reference
    if ((prediction.startswith('[') and prediction.endswith(']')
         and not reference.startswith('('))
            or (prediction.startswith('(') and prediction.endswith(')')
                and not reference.startswith('['))):
        pred_flat = pred_flat.strip('[]()')
        ref_flat = ref_flat.strip('[]()')
    for character in '{}()':
        pred_flat = pred_flat.replace(character, '')
        ref_flat = ref_flat.replace(character, '')
    if pred_flat.lower() == ref_flat.lower():
        return True

    if (regex.match(r'(\(|\[).+(\)|\])', prediction) is not None
            and regex.match(r'(\(|\[).+(\)|\])', reference) is not None):
        pred_parts = prediction[1:-1].split(',')
        ref_parts = reference[1:-1].split(',')
        if len(pred_parts) == len(ref_parts) and all(
                math_equal(left, right, include_percentage, is_close)
                for left, right in zip(pred_parts, ref_parts)):
            return True

    if (((prediction.startswith(r'\begin{pmatrix}')
          or prediction.startswith(r'\begin{bmatrix}')) and
         (prediction.endswith(r'\end{pmatrix}')
          or prediction.endswith(r'\end{bmatrix}')))
            and ((reference.startswith(r'\begin{pmatrix}')
                  or reference.startswith(r'\begin{bmatrix}')) and
                 (reference.endswith(r'\end{pmatrix}')
                  or reference.endswith(r'\end{bmatrix}')))):
        pred_lines = [
            line.strip()
            for line in prediction[len(r'\begin{pmatrix}'
                                       ):-len(r'\end{pmatrix}')].split(r'\\')
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len(r'\begin{pmatrix}'
                                      ):-len(r'\end{pmatrix}')].split(r'\\')
            if line.strip()
        ]
        if len(pred_lines) == len(ref_lines):
            matched = True
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split('&')
                ref_parts = ref_line.split('&')
                if len(pred_parts) != len(ref_parts) or not all(
                        math_equal(pred_parts[index], ref_parts[index],
                                   include_percentage, is_close)
                        for index in range(len(pred_parts))):
                    matched = False
                    break
            if matched:
                return True

    if prediction.count('=') == reference.count('=') == 1:
        pred_left, pred_right = prediction.split('=')
        ref_left, ref_right = reference.split('=')
        pred = f'{pred_left.strip()} - ({pred_right.strip()})'
        ref = f'{ref_left.strip()} - ({ref_right.strip()})'
        compare = _symbolic_with_timeout if timeout else _symbolic_equal
        if compare(pred, ref) or compare(f'-({pred})', ref):
            return True
    elif (prediction.count('=') == 1
          and len(prediction.split('=')[0].strip()) <= 2
          and '=' not in reference):
        return math_equal(
            prediction.split('=')[1], reference, include_percentage, is_close)
    elif (reference.count('=') == 1
          and len(reference.split('=')[0].strip()) <= 2
          and '=' not in prediction):
        return math_equal(prediction,
                          reference.split('=')[1], include_percentage,
                          is_close)

    compare = _symbolic_with_timeout if timeout else _symbolic_equal
    return bool(compare(prediction, reference))
