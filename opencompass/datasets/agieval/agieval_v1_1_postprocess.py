"""Rule-based option-set parsing for AGIEval v1.1 gaokao-mathqa."""

import re
import unicodedata

from opencompass.registry import TEXT_POSTPROCESSORS


def normalize_mathqa_label(label):
    """Canonicalize a reference without changing the downloaded source file."""
    if isinstance(label, list):
        label = ''.join(label)
    if not isinstance(label, str):
        raise ValueError(f'Invalid mathqa reference: {label!r}')
    value = re.sub(r'\s+', '', label)
    if not re.fullmatch(r'[A-D]+', value):
        raise ValueError(f'Invalid mathqa reference: {label!r}')
    return ''.join(sorted(set(value)))


def _option_expression(text):
    """Parse an entire answer expression, never collect letters from prose."""
    text = unicodedata.normalize('NFKC', text).strip().rstrip('。.！!')
    for _ in range(3):
        text = re.sub(r'\\(?:boxed|text|mathrm)\s*\{([^{}]*)\}',
                      r'\1', text)
    text = re.sub(r'\band\b', ',', text, flags=re.I)
    text = text.replace('选项', '')
    # Accept common answer wrappers: (A), ['A', 'D'], bold and math.
    text = text.translate(str.maketrans('', '', "()[]{}'\"*$" + chr(96)))
    text = text.strip().upper()
    separator = r'(?:\s*[,、&;]\s*|\s+|[和及与])'
    if not re.fullmatch(rf'[A-D](?:{separator}?[A-D])*', text):
        return ''
    return ''.join(sorted(set(re.findall(r'[A-D]', text))))


_FINAL_ANSWER = re.compile(
    r'(?:最终|最后)(?:的)?(?:答案|选项)(?:应当|应该|应)?'
    r'(?:是|为|选)?\s*[:：]?\s*'
    r'|(?:the\s+)?final\s+answers?\s*(?:is|are)?\s*[:：]?\s*',
    re.I)
_ANSWER = re.compile(
    r'(?:正确)?答案(?:选项)?(?:应当|应该|应)?(?:是|为|选)?\s*[:：]?\s*'
    r'|(?:因此|所以|故)(?:应当|应该|应)?(?:选择|选)\s*[:：]?\s*'
    r'|(?:the\s+)?(?:correct\s+)?answers?\s*(?:is|are)?\s*[:：]?\s*',
    re.I)


_MATH_BLOCK = re.compile(
    r'\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|\$([^$\n]*)\$',
    re.S)


def _math_block_options(match):
    return _option_expression(next(
        group for group in match.groups() if group is not None))


@TEXT_POSTPROCESSORS.register_module()
def agieval_mathqa_postprocess(text):
    """Extract one final option set; empty/ambiguous answers score as wrong.

    Explicit final-answer markers take priority over ordinary answer markers.
    Use the last marker at that priority, or an unmarked final answer line.
    The selected answer must consist entirely of an option expression.
    No reference answer is consulted and no second model call is made.
    """
    if not isinstance(text, str) or not text.strip():
        return ''
    text = unicodedata.normalize('NFKC', text).strip()
    matches = list(_FINAL_ANSWER.finditer(text)) or list(_ANSWER.finditer(text))
    if matches:
        candidate = text[matches[-1].end():].strip()
        # Preserve a complete math block before splitting prose into lines.
        # Its body still has to be an entire, unambiguous option expression.
        math_block = _MATH_BLOCK.match(candidate)
        if math_block:
            suffix = candidate[math_block.end():]
            if suffix.strip() and not re.match(
                    r'^[ \t]*(?:\n|[。.!！])', suffix):
                return ''
            return _math_block_options(math_block)
        # A sentence after the answer may explain it. Do not search that prose.
        candidate = re.split(r'[\n。.!！]', candidate, maxsplit=1)[0]
        return _option_expression(candidate)
    # A standalone final formula can span lines without an answer marker.
    math_blocks = list(_MATH_BLOCK.finditer(text))
    if math_blocks:
        math_block = math_blocks[-1]
        if re.fullmatch(r'\s*[。.！!]*\s*', text[math_block.end():]):
            return _math_block_options(math_block)
    # A final boxed answer is also explicit, including one following prose.
    boxed = re.search(r'\\boxed\s*\{([^{}]*)\}\s*[$*。.！!]*$', text)
    if boxed:
        return _option_expression(boxed.group(1))
    return _option_expression(text.splitlines()[-1])
