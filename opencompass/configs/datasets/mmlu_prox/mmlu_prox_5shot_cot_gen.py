from opencompass.datasets import MMLUProXDataset, MMLUProXEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
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
    'uk': 'Питання:'
}
categories = ('biology', 'business', 'chemistry', 'computer science',
              'economics', 'engineering', 'health', 'history', 'law', 'math',
              'other', 'philosophy', 'physics', 'psychology')

mmlu_prox_datasets = []
for lang, question_stop in question_stops.items():
    for category in categories:
        reader_cfg = dict(
            input_columns=['description', 'question_prompt', 'fewshot_prompt'],
            output_column='answer_letter',
            train_split='validation',
            test_split='test')
        infer_cfg = dict(
            ice_template=dict(type=PromptTemplate,
                              template='{fewshot_prompt}'),
            prompt_template=dict(type=PromptTemplate,
                                 template='{description}</E>{question_prompt}',
                                 ice_token='</E>'),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            inferencer=dict(
                type=GenInferencer,
                max_out_len=2048,
                stopping_criteria=['</s>', 'Q:', question_stop, '<|im_end|>'],
                generation_kwargs=dict(do_sample=False, temperature=0.0)))
        eval_cfg = dict(evaluator=dict(type=MMLUProXEvaluator))
        mmlu_prox_datasets.append(
            dict(abbr=f'mmlu_prox_{lang}_{category.replace(" ", "_")}',
                 type=MMLUProXDataset,
                 lang=lang,
                 category=category,
                 reader_cfg=reader_cfg,
                 infer_cfg=infer_cfg,
                 eval_cfg=eval_cfg))
