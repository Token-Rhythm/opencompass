from opencompass.datasets.ceval import (build_ceval_evalscope_prompt,
                                        ceval_answer_postprocess,
                                        ceval_evalscope_answer_postprocess)


def test_ceval_answer_postprocess_prefers_explicit_answer_over_acronyms():
    assert ceval_answer_postprocess('根据 TCP/IP 协议分析，答案是 B。') == 'B'


def test_ceval_answer_postprocess_supports_common_final_formats():
    assert ceval_answer_postprocess('{"answer": "C"}') == 'C'
    assert ceval_answer_postprocess('The final answer is (D).') == 'D'
    assert ceval_answer_postprocess(r'因此最终结果为 \\boxed{A}') == 'A'


def test_ceval_answer_postprocess_uses_final_standalone_choice():
    assert ceval_answer_postprocess('A 与 B 均需比较。\n最终选择：C') == 'C'
    assert ceval_answer_postprocess('分析结束。\n**d.**') == 'D'


def test_ceval_answer_postprocess_rejects_missing_choice():
    assert ceval_answer_postprocess('无法确定。') == ''
    assert ceval_answer_postprocess('') == ''


def test_evalscope_answer_postprocess_matches_official_strict_format():
    assert ceval_evalscope_answer_postprocess('分析。\n答案：B') == 'B'
    assert ceval_evalscope_answer_postprocess('答案：A\n修正。\n答案：D') == 'D'
    assert ceval_evalscope_answer_postprocess('答案是 B') == ''
    assert ceval_evalscope_answer_postprocess('The final answer is C.') == ''
    assert ceval_evalscope_answer_postprocess('答案： B') == ''


def test_evalscope_prompt_keeps_five_shots_and_target_in_one_context():
    dev_rows = [
        dict(question=f'dev-{index}', A='a', B='b', C='c', D='d',
             explanation=f'exp-{index}', answer='ABCD'[index % 4])
        for index in range(6)
    ]
    target = dict(question='target', A='ta', B='tb', C='tc', D='td')

    prompt = build_ceval_evalscope_prompt('测试科目', dev_rows, target)

    assert prompt.startswith('以下是一些示例问题：\n\n')
    assert prompt.count('解析：') == 5
    assert 'dev-4' in prompt
    assert 'dev-5' not in prompt
    assert '以下是中国关于测试科目的单项选择题' in prompt
    assert prompt.endswith('问题：target\n选项：\nA. ta\nB. tb\nC. tc\nD. td\n')


def test_evalscope_config_maps_the_available_val_split_for_reader():
    from opencompass.configs.datasets.ceval.ceval_evalscope_gen import (
        ceval_evalscope_datasets,
    )

    reader_cfg = ceval_evalscope_datasets[0]['reader_cfg']
    assert reader_cfg['train_split'] == 'val'
    assert reader_cfg['test_split'] == 'val'
    inferencer = ceval_evalscope_datasets[0]['infer_cfg']['inferencer']
    from opencompass.openicl.icl_inferencer import ParallelGenInferencer
    assert inferencer == {
        'type': ParallelGenInferencer,
        'save_every': 1,
    }
