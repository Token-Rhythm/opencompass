"""Official MMLU-ProX Full 5-shot CoT evaluation."""

from opencompass.datasets import MMLUProXDataset, MMLUProXEvaluator
from opencompass.openicl.icl_inferencer import ParallelGenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever

question_stops = {
    'en': 'Question:',
    'ja': '質問：',
    'zh': '问题：',
    'ko': '질문：',
    'fr': 'Question :',
    'de': 'Frage:',
    'es': 'Pregunta:',
    'pt': 'Pergunta:',
    'zu': 'Umbuzo:',
    'sw': 'Swali:',
    'wo': 'Laaj:',
    'yo': 'Ìbéèrè:',
    'th': 'คำถาม:',
    'ar': 'سؤال:',
    'hi': 'प्रश्न:',
    'bn': 'প্রশ্ন:',
    'mr': 'प्रश्न:',
    'ne': 'प्रश्न:',
    'af': 'Vraag:',
    'te': 'ప్రశ్న:',
    'ur': 'سوال:',
    'ru': 'Вопрос:',
    'id': 'Pertanyaan:',
    'vi': 'Câu hỏi:',
    'cs': 'Otázka:',
    'hu': 'Kérdés:',
    'it': 'Domanda:',
    'sr': 'Pitanje:',
    'uk': 'Питання:',
}
categories = ('biology', 'business', 'chemistry', 'computer science',
              'economics', 'engineering', 'health', 'history', 'law', 'math',
              'other', 'philosophy', 'physics', 'psychology')

mmlu_prox_5shot_datasets = []
for lang, question_stop in question_stops.items():
    for category in categories:
        mmlu_prox_5shot_datasets.append(
            dict(abbr=f'mmlu_prox_5shot_{lang}_{category.replace(" ", "_")}',
                 type=MMLUProXDataset,
                 path='li-lab/MMLU-ProX',
                 revision='8e6106a6c6ce1c5027e66cc338143cf997b2aa09',
                 lang=lang,
                 category=category,
                 reader_cfg=dict(input_columns=[
                     'description', 'question_prompt',
                     'fewshot_question_prompt', 'fewshot_answer_prompt'
                 ],
                                 output_column='answer_letter',
                                 train_split='validation',
                                 test_split='test'),
                 infer_cfg=dict(
                     ice_template=dict(
                         type=PromptTemplate,
                         template='{fewshot_prompt}',
                     ),
                     # lm-eval-harness concatenates the task description,
                     # five demonstrations, and the test question before
                     # applying any model chat template. Keep that complete
                     # task context in one user message at the API boundary.
                     prompt_template=dict(
                         type=PromptTemplate,
                         template='{description}</E>{question_prompt}',
                         ice_token='</E>',
                     ),
                     retriever=dict(type=FixKRetriever,
                                    fix_id_list=[0, 1, 2, 3, 4],
                                    ice_separator='',
                                    ice_eos_token=''),
                     inferencer=dict(type=ParallelGenInferencer,
                                     max_out_len=2048,
                                     save_every=1)),
                 eval_cfg=dict(evaluator=dict(type=MMLUProXEvaluator))))
