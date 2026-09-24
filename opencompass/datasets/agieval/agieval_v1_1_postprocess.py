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
    if not re.fullmatch(rf'[A-E](?:{separator}?[A-E])*', text):
        return ''
    return ''.join(sorted(set(re.findall(r'[A-E]', text))))


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




def _leading_options(candidate):
    """Read an answer at the start, allowing a following option description."""
    candidate = re.sub(r'^[\s*>:：]+', '', candidate)
    candidate = re.sub(r'^(?:应当|应该|应)(?:选择|选)\s*[:：]?\s*', '', candidate)
    block = _MATH_BLOCK.match(candidate)
    if block:
        suffix = candidate[block.end():].strip()
        if re.match(r'[,，]?\s*(?:但也可能|或\s*[*($]*[A-E]|or\s+[A-E]\b)', suffix, re.I):
            return ''
        return _math_block_options(block)
    candidate = candidate.splitlines()[0]
    # Remove formatting, but keep prose so an English word is never an option.
    for _ in range(3):
        candidate = re.sub(r'\\(?:boxed|text|mathrm)\s*\{([^{}]*)\}',
                           r'\1', candidate)
    candidate = candidate.replace('**', '').replace('选项', '').strip()
    whole = _option_expression(candidate)
    if whole:
        return whole
    labelled = re.match(r'^\(([A-E]+)\)\s+(.+)', candidate)
    if labelled and not re.match(
            r'[,，、&;和及与/(]|and\b|or\b|或|但也可能', labelled.group(2), re.I):
        return ''.join(sorted(set(labelled.group(1))))
    atom = r'(?:\([A-E]+\)|\[[A-E]+\]|[A-E]+)(?![A-Za-z])'
    match = re.match(rf'{atom}(?:(?:[ \t]*[,，、&;和及与][ \t]*|[ \t]+(?:and[ \t]+)?){atom})*',
                     candidate)
    if not match:
        return ''
    end = match.end()
    if re.match(r'\s*[,，]?\s*(?:但也可能|或|or\b|/)', candidate[end:], re.I):
        return ''
    # No prefix of an English word, invalid option, or slash alternative.
    if end < len(candidate) and (candidate[end] == '/' or (
            candidate[end].isascii() and candidate[end].isalpha()
            and not re.search(r'[\s)\]}]$', match.group()))):
        return ''
    return ''.join(sorted(set(re.findall(r'[A-E]', match.group()))))


# Standalone conclusions also occur without 因此/所以.
_ANSWER = re.compile(
    _ANSWER.pattern.replace('(?:是|为|选)', '(?:是|为|选择|选)')
    .replace('(?:选择|选)', '(?:选择|选)(?!项|择)')
    .replace(r'answers?\s*', r'answers?(?:\s+choices?)?\s*')
    + r'|(?:应当|应该|应)(?:选择|选)(?!项|择)\s*[:：]?\s*'
    r'|(?:^|[\n。])\s*选(?!项|择)\s*[:：]?\s*'
    r'|选择(?=\s*[*（(]*[A-E])\s*', re.I)


def _choice_answer(text, single):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = unicodedata.normalize('NFKC', text).strip()
    matches = list(_FINAL_ANSWER.finditer(text))
    if not matches:
        matches = [m for m in _ANSWER.finditer(text) if re.match(
            r'[\s*>:：]*(?:[A-E（(\[\x27\"]|\\|\$|不确定|无法确定)',
            text[m.end():])]
    if matches:
        value = _leading_options(text[matches[-1].end():])
    else:
        value = _option_expression(text.splitlines()[-1])
        if not value:
            boxes = list(_MATH_BLOCK.finditer(text))
            if boxes and not text[boxes[-1].end():].strip(' 。.!！'):
                value = _math_block_options(boxes[-1])
        if not value:
            boxed = re.search(r'\\boxed\s*\{([^{}]*)\}\s*[$*。.！!]*$', text)
            if boxed:
                value = _option_expression(boxed.group(1))
        if not value and re.match(r'^(?:\([A-E]+\)|[A-E]+[.:：])\s', text):
            value = _leading_options(text.splitlines()[0])
        if not value and single:
            # User-requested fallback: first capital option in textual order.
            # A standalone multi-answer must not silently become its first one.
            standalone = re.fullmatch(r'[A-E\s,、]+', text)
            if standalone:
                value = ''.join(sorted(set(re.findall(r'[A-E]', text))))
            else:
                match = re.search(r'[A-E]', text)
                value = match.group() if match else ''
    if single:
        return value if len(value) == 1 else ''
    return value if re.fullmatch(r'[A-D]+', value) else ''


@TEXT_POSTPROCESSORS.register_module()
def agieval_single_choice_postprocess(text, options='ABCDE'):
    """Explicit answer first, then first-capital fallback; reject option sets."""
    answer = _choice_answer(text, single=True)
    return answer if answer in options else ''


@TEXT_POSTPROCESSORS.register_module()
def agieval_mathqa_postprocess(text):
    return _choice_answer(text, single=False)
