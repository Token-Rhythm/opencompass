"""Conservative final-expression extraction for AGIEval v1.1 cloze tasks.

This is format normalization plus legacy string equivalence, not symbolic or
LLM grading. Multiple blanks are ordered and require every answer to match.
"""
import re

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS

from .agieval_v1_1_postprocess import _ANSWER, _FINAL_ANSWER, _MATH_BLOCK
from .math_equivalence import is_equiv
from .post_process import parse_math_answer


def _boxes(text):
    result = []
    for match in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        start = match.end()
        depth = 1
        for i in range(start, len(text)):
            depth += (text[i] == '{') - (text[i] == '}')
            if depth == 0:
                result.append((match.start(), i + 1, text[start:i]))
                break
    return result


def _split_expressions(text):
    """Split top-level answers without splitting fractions or coordinates."""
    text = re.sub(r'\\(?:qquad|quad)\b', ' ', text)
    parts, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char in '({[':
            depth += 1
        elif char in ')}]':
            depth -= 1
        elif depth == 0 and char in ';；,，':
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def normalize_cloze_expression(text):
    text = text.strip().rstrip('。.;；').strip().strip('*').strip()
    for left, right in [('\\[', '\\]'), ('\\(', '\\)'), ('$$', '$$'), ('$', '$')]:
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
    text = text.rstrip('。.;；').strip()
    # Labels such as "直线斜率" are formatting, not part of the expression.
    text = re.sub(r'^\\text\{[^{}]*[\u4e00-\u9fff][^{}]*\}\s*', '', text)
    text = re.sub(r'\\(?:mathrm|mathbf)\{([^{}]*)\}', r'\1', text)
    text = text.replace(r'\%', '').replace('%', '')
    # Strip a variable assignment only when its RHS is a constant expression.
    # Preserve equations involving variables, e.g. y=2x and 2x+y+1=0.
    if text.count('=') == 1:
        lhs, rhs = text.split('=')
        residue = re.sub(r'\\[A-Za-z]+', '', rhs)
        if re.fullmatch(r'[A-Za-z](?:\^\{-1\})?\([^=]*\)', lhs.strip()):
            text = rhs.strip()
        elif (re.fullmatch(r'[A-Za-z](?:_\{?\d+\}?)?', lhs.strip())
                and not re.search(r'[A-Za-z]', residue)):
            text = rhs.strip()
    return text


def extract_cloze_answers(text):
    """Extract independently of reference values and reference blank count."""
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.strip()
    markers = list(_FINAL_ANSWER.finditer(text)) or list(_ANSWER.finditer(text))
    candidate = text[markers[-1].end():].strip() if markers else text
    if markers and not candidate.startswith(('\\[', '$$')):
        # Do not replace an explicit opening answer with later working formulas.
        candidate = candidate.split('\n\n', 1)[0]
    boxes = _boxes(candidate)
    if boxes:
        # Only the last adjacent group of boxes, never all intermediate boxes.
        group = [boxes[-1]]
        for box in reversed(boxes[:-1]):
            between = candidate[box[1]:group[0][0]]
            between = re.sub(r'\\(?:qquad|quad)|\\[\[\]()]', '', between)
            if not re.fullmatch(r'[\s$*,，;；。]*', between):
                break
            group.insert(0, box)
        expressions = [p for box in group for p in _split_expressions(box[2])]
    else:
        blocks = list(_MATH_BLOCK.finditer(candidate))
        if blocks:
            group = [blocks[-1]]
            for block in reversed(blocks[:-1]):
                between = candidate[block.end():group[0].start()]
                if not re.fullmatch(r'[\s,，;；。和及与]*', between):
                    break
                group.insert(0, block)
            expressions = [p for block in group
                           for p in _split_expressions(next(
                               g for g in block.groups() if g is not None))]
        else:
            line = candidate.splitlines()[0] if markers else candidate.splitlines()[-1]
            line = line.strip().strip('*').strip()
            if ('\\' not in line and '=' not in line
                    and re.search(r'[A-Za-z\u4e00-\u9fff]', line)):
                # Legacy numeric fallback for prose/units; preserve signs.
                numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', line)
                expressions = numbers[-1:] if numbers else _split_expressions(line)
            elif not re.search(r'[\u4e00-\u9fff]', line):
                expressions = _split_expressions(line)
            else:
                value = parse_math_answer('', candidate)
                expressions = [value] if value is not None else []
    if len(expressions) > 1:
        expressions = [p.split('=')[-1].strip()
                       if p.count('=') == 1 and not re.search(
                           r'[A-Za-z]', re.sub(r'\\[A-Za-z]+', '', p.split('=')[-1]))
                       else p for p in expressions]
    return [normalize_cloze_expression(p) for p in expressions]


@ICL_EVALUATORS.register_module()
class AGIEvalV11ClozeEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError('predictions and references have different lengths')
        details = []
        for text, reference in zip(predictions, references):
            pred = extract_cloze_answers(text)
            ref = [normalize_cloze_expression(p) for x in reference.split(';')
                   for p in _split_expressions(normalize_cloze_expression(x))]
            correct = (len(pred) == len(ref) and bool(pred)
                       and all(bool(p) and is_equiv(p, r)
                               for p, r in zip(pred, ref)))
            details.append(dict(pred=pred, answer=ref, correct=correct))
        return dict(score=100 * sum(d['correct'] for d in details) / len(details)
                    if details else 0, details=details)
