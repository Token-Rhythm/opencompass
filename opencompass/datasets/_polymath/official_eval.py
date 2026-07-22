"""Dependency-light port of QwenLM/PolyMath's ``math_equal`` routine.

The upstream implementation bundles a generated LaTeX parser. OpenCompass
already requires SymPy's ANTLR parser for its math benchmarks, so this port
keeps the upstream comparison order and tolerances while using that parser.
"""

import math
import multiprocessing
import re


def _surface_form(value):
    value = re.sub(r'\s+', '', str(value).strip())
    if len(value) >= 2 and value.startswith('$') and value.endswith('$'):
        value = value[1:-1]
    return value


def _parse_number(value):
    value = re.sub(',', '', _surface_form(value))
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
    matches = re.findall(r'\b([A-E])\b', str(value).upper())
    return matches[-1] if matches else str(value).strip().rstrip('./')


def _matrix_rows(value):
    begin = None
    end = None
    for matrix_type in ('pmatrix', 'bmatrix'):
        candidate_begin = rf'\begin{{{matrix_type}}}'
        candidate_end = rf'\end{{{matrix_type}}}'
        if value.startswith(candidate_begin) and value.endswith(candidate_end):
            begin, end = candidate_begin, candidate_end
            break
    if begin is None:
        return None
    return [[cell.strip() for cell in row.split('&')]
            for row in value[len(begin):-len(end)].split(r'\\') if row.strip()]


def _symbolic_equal(left, right):
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    try:
        from latex2sympy2_extended import latex2sympy
    except ImportError:
        latex2sympy = None

    def parse(value):
        parsers = [parse_latex, parse_expr]
        if latex2sympy is not None:
            parsers.append(latex2sympy)
        for parser in parsers:
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
    # Box extraction removes spaces, while official references retain them and
    # often include outer math delimiters.
    if _surface_form(prediction).lower() == _surface_form(reference).lower():
        return True
    if str(reference).strip() in {
            'A', 'B', 'C', 'D', 'E'
    } and _choice_answer(prediction) == str(reference).strip():
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

    if (re.fullmatch(r'[\[(].+[\])]', prediction)
            and re.fullmatch(r'[\[(].+[\])]', reference)):
        pred_parts = prediction[1:-1].split(',')
        ref_parts = reference[1:-1].split(',')
        if len(pred_parts) == len(ref_parts) and all(
                math_equal(left, right, include_percentage, is_close)
                for left, right in zip(pred_parts, ref_parts)):
            return True

    pred_matrix = _matrix_rows(prediction)
    ref_matrix = _matrix_rows(reference)
    if pred_matrix is not None and ref_matrix is not None:
        if (len(pred_matrix) == len(ref_matrix) and all(
                len(left) == len(right)
                for left, right in zip(pred_matrix, ref_matrix)) and all(
                    math_equal(left_cell, right_cell, include_percentage,
                               is_close)
                    for left, right in zip(pred_matrix, ref_matrix)
                    for left_cell, right_cell in zip(left, right))):
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
