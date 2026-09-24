import pytest

from opencompass.datasets.agieval.agieval_v1_1_postprocess import (
    agieval_mathqa_postprocess, agieval_single_choice_postprocess)
from opencompass.datasets.agieval.agieval_v1_1_cloze import (
    AGIEvalV11ClozeEvaluator, extract_cloze_answers)


@pytest.mark.parametrize('text,expected', [
    ('应选 **(A) 0.8**', 'A'), ('所以应选择 **(D) 18**', 'D'),
    ('应选 **C、D**。因此严格来说应选：**C 和 D**。', 'CD'),
    ('答案：**B**\n31%\nD 错误。', 'B'),
    ('答案：**C**\n11:23，长征5号B。', 'C'),
    ('The answer is **(C) F or T**', 'C'),
    ('**Answer: (B)**\n1000 is a number. A is wrong.', 'B'),
    ('答案：C\n\nA 项错误；B 项错误。', 'C'),
    ('应选择 **D**。\n\n选项 D 直接否定前提。', 'D'),
    ('答案应选择 **C**。', 'C'),
    ('应从 **A 到 D 选择 C**。', 'C'),
    ('应选择 **B**。\n这说明作者觉得选集不全面。', 'B'),
    ('应选 **C．很可能成功**。', 'C'),
    ('应选 **A（CH4）**。', 'A'),
    ('答案：\n\n> **A**', 'A'),
    ('答案：C。\n这不是选项中的答案。', 'C'),
    ('Answer Choices: (D) The government ought not to silence an opinion.', 'D'),
    ('最终答案：\n> **应选 C。**', 'C'),
    ('A. 充分不必要条件\n\n因为 a>6，所以 a²>36。', 'A'),
    ('答案是 AD，但也可能是 B', ''), ('答案是 A/B', ''),
    ('答案是 AB', 'AB'), ('答案是 Apple', ''),
])
def test_explicit_options(text, expected):
    assert agieval_mathqa_postprocess(text) == expected


@pytest.mark.parametrize('text,expected', [
    ('答案：**B**\n31%\nD 错误。', 'B'),
    ('**Answer: (B)**\n1000 is a number. A is wrong.', 'B'),
    ('AD', ''), ('答案是 AB', ''), ('答案是 A 或 B', ''),
    ('D is my choice; A is incorrect.', 'D'),
    (r'Answer: \[\boxed{E}\]', 'E'),
    ('(E) A school calendar made up of periods of study.', 'E'),
    ('(C) E and M\n\nExplanation: E and F cannot both participate.', 'C'),
])
def test_single_choice(text, expected):
    assert agieval_single_choice_postprocess(text) == expected


@pytest.mark.parametrize('text,reference', [
    (r'答案：\(\boxed{\frac{1}{2}}\)', r'$\frac{1}{2}$'),
    (r'\boxed{2x+y+1=0}', r'$2x+y+1=0$'),
    (r'\boxed{y=2x}', r'$y=2x$'),
    (r'\(a_1=5\)，\(a_2+a_3+a_4=10\)。', '$5$;$10$'),
    (r'答案：\[m-n=1,\qquad E(\xi)=\frac89\]', r'1;$\frac{8}{9}$'),
    (r'\boxed{\text{直线斜率 } \frac{2\sqrt5}{5}},\qquad'
     r'\boxed{\text{离心率 } \frac{\sqrt5}{5}}',
     r'$\frac{2\sqrt{5}}{5}$;$\frac{\sqrt{5}}{5}$'),
    (r'\boxed{(0,1)}', '$(0,1)$'),
    ('The answer is **9**.', '9'),
    ('26 feet', '26'),
    ('The answer is **7%**.', '7'),
    ('The answer is **10**.\n\nWorking: \\[19,31\\]', '10'),
    (r'\boxed{h^{-1}(x)=\frac{x+2}{7}}', r'\frac{x+2}{7}'),
    (r'\boxed{45,135}', r'45^\circ,135^\circ'),
    (r'\boxed{25}', r'25\%'),
    ('The answer is **parabola**.', 'parabola'),
    ('even', 'even'),
    ('B, C', 'B,C'),
])
def test_cloze_formats(text, reference):
    assert AGIEvalV11ClozeEvaluator().score([text], [reference])['score'] == 100


@pytest.mark.parametrize('text,reference', [
    (r'\boxed{0}', '$2x+y+1=0$'),
    (r'\boxed{10}', '$5$;$10$'),
    (r'\boxed{10};\boxed{5}', '$5$;$10$'),
    ('', '0'),
])
def test_cloze_no_partial_or_reference_guided_credit(text, reference):
    assert AGIEvalV11ClozeEvaluator().score([text], [reference])['score'] == 0


def test_only_final_adjacent_boxes():
    assert extract_cloze_answers(r'\boxed{3} intermediate. Final: \boxed{5}') == ['5']
