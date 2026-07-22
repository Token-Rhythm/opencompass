"""PolyMath adapter following the official multilingual math protocol."""

import re

from datasets import DatasetDict, load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

POLYMATH_PATH = 'Qwen/PolyMath'
POLYMATH_REVISION = '71e0db902e0ece05e208dfd8b1695bd3d95cf130'
POLYMATH_LANGUAGES = ('en', 'zh', 'ar', 'bn', 'de', 'es', 'fr', 'id', 'it',
                      'ja', 'ko', 'ms', 'pt', 'ru', 'sw', 'te', 'th', 'vi')
POLYMATH_LEVELS = ('low', 'medium', 'high', 'top')

# Verbatim from QwenLM/PolyMath@fbf4e41, instruction.py. The paper specifies
# that this instruction is appended after the input problem.
POLYMATH_ANSWER_INSTRUCTIONS = {
    'en': r'Note: Please put the final answer in the $\boxed\{\}$.',
    'zh': r'注意：请将最终答案放在 $\boxed\{\}$ 中。',
    'ar': r'ملاحظة: يُرجى وضع الإجابة النهائية في $\boxed\{\}$.',
    'bn':
    r'বিঃদ্রঃ: অনুগ্রহ করে চূড়ান্ত উত্তরটি $\boxed\{\}$ এর মধ্যে রাখুন।',
    'de': r'Hinweis: Bitte setzen Sie die endgültige Antwort in $\boxed\{\}$.',
    'es': r'Nota: Por favor, coloque la respuesta final en el $\boxed\{\}$.',
    'fr':
    r'Remarque : Veuillez mettre la réponse finale dans le $\boxed\{\}$.',
    'id': r'Catatan: Silakan letakkan jawaban akhir di dalam $\boxed\{\}$.',
    'it': r'Nota: Per favore, metti la risposta finale nel $\boxed\{\}$.',
    'ja': r'注意：最終的な答えを $\boxed\{\}$ に入れてください。',
    'ko': r'참고: 최종 답안을 $\boxed\{\}$ 안에 넣어 주세요.',
    'ms': r'Nota: Sila letakkan jawapan akhir dalam $\boxed\{\}$.',
    'pt': r'Nota: Por favor, coloque a resposta final no $\boxed\{\}$.',
    'ru':
    r'Примечание: Пожалуйста, поместите окончательный ответ в $\boxed\{\}$.',
    'sw': r'Kumbuka: Tafadhali weka jibu la mwisho katika $\boxed\{\}$.',
    'te': r'గమనిక: దయచేసి తుది జవాబును $\boxed\{\}$ లో ఉంచండి.',
    'th': r'หมายเหตุ: กรุณาใส่คำตอบสุดท้ายใน $\boxed\{\}$.',
    'vi': r'Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong $\boxed\{\}$.',
}


def _format_polymath(row, lang, level):
    return {
        'id': row['id'],
        'prompt': f'{row["question"]}\n\n{POLYMATH_ANSWER_INSTRUCTIONS[lang]}',
        'answer': str(row['answer']),
        'language': lang,
        'level': level,
    }


@LOAD_DATASET.register_module()
class PolyMathDataset(BaseDataset):
    """Load one of the official 18 language x 4 difficulty subsets."""

    @staticmethod
    def load(path=POLYMATH_PATH,
             lang=None,
             level=None,
             revision=POLYMATH_REVISION):
        if lang not in POLYMATH_LANGUAGES:
            raise ValueError(f'Unsupported PolyMath language: {lang!r}')
        if level not in POLYMATH_LEVELS:
            raise ValueError(f'Unsupported PolyMath level: {level!r}')
        split = load_dataset(path,
                             data_files=f'{lang}/{level}.parquet',
                             revision=revision,
                             split='train')
        split = split.map(
            lambda row: _format_polymath(row, lang=lang, level=level))
        return DatasetDict({'test': split})


def extract_first_boxed_content(text):
    """Match the official evaluator: remove spaces and use the first box."""
    text = str(text).replace(' ', '')
    results = []
    for match in re.finditer(r'boxed\{', text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            if text[index] == '{':
                depth += 1
            elif text[index] == '}':
                depth -= 1
            index += 1
        if depth == 0:
            results.append(text[start:index - 1])
    return results[0] if results else None


class PolyMathEvaluator(BaseEvaluator):
    """Official boxed-answer extraction and PolyMath mathematical equality."""

    def score(self, predictions, references, test_set=None):
        try:
            from ._polymath.official_eval import math_equal
        except ImportError as error:
            raise ImportError(
                'PolyMath evaluation requires sympy and '
                'antlr4-python3-runtime==4.11. Install the OpenCompass extra '
                'requirements.') from error

        correct = 0
        details = []
        for prediction, reference in zip(predictions, references):
            parsed = extract_first_boxed_content(prediction)
            is_correct = bool(math_equal(parsed, reference))
            correct += is_correct
            details.append({
                'pred': prediction,
                'parsed': parsed,
                'answer': reference,
                'correct': is_correct,
            })
        return {
            'accuracy': 100 * correct / len(references) if references else 0,
            'details': details,
        }
