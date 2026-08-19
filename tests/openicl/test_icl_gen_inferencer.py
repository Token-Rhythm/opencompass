import json
from unittest.mock import MagicMock

from opencompass.openicl.icl_inferencer.icl_gen_inferencer import GenInferencer


def test_gen_inferencer_resumes_sparse_checkpoint(tmp_path):
    model = MagicMock()
    model.is_api = True
    model.generation_kwargs = {}
    model.generate_from_template.side_effect = [['new-one'], ['new-three']]
    model.parse_template.side_effect = lambda entries, mode: entries

    inferencer = GenInferencer(model=model,
                               max_out_len=32,
                               batch_size=1,
                               output_json_filepath=str(tmp_path),
                               output_json_filename='predictions.json')
    inferencer.get_generation_prompt_list_from_retriever_indices = MagicMock(
        return_value=['zero', 'one', 'two', 'three'])

    checkpoint = tmp_path / 'tmp_predictions.jsonl'
    checkpoint.write_text(
        '\n'.join([
            json.dumps({
                'idx': '0',
                'origin_prompt': 'zero',
                'prediction': 'old-zero'
            }),
            json.dumps({
                'idx': '2',
                'origin_prompt': 'two',
                'prediction': 'old-two'
            }),
        ]) + '\n')

    retriever = MagicMock()
    retriever.retrieve.return_value = [[], [], [], []]
    retriever.dataset_reader.output_column = None

    result = inferencer.inference(retriever)

    assert result == ['old-zero', 'new-one', 'old-two', 'new-three']
    calls = model.generate_from_template.call_args_list
    assert [call.args[0] for call in calls] == [['one'], ['three']]
    persisted = json.loads((tmp_path / 'predictions.json').read_text())
    assert persisted['0']['prediction'] == 'old-zero'
    assert persisted['1']['prediction'] == 'new-one'
    assert persisted['2']['prediction'] == 'old-two'
    assert persisted['3']['prediction'] == 'new-three'
