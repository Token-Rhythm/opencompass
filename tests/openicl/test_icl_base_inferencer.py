import json
import tempfile
import unittest
from pathlib import Path

from opencompass.openicl.icl_inferencer import BaseInferencer
from opencompass.openicl.icl_inferencer.icl_base_inferencer import (
    ChatOutputHandler, GenInferencerOutputHandler, prediction_content)


class TestBaseInferencer(unittest.TestCase):

    def test_get_dataloader_with_none_batch_size_keeps_batch(self):
        prompt = [{'role': 'user', 'content': 'q1'}]
        gold = {'capability': 'writing'}
        sample = (prompt, gold)

        dataloader = BaseInferencer.get_dataloader([sample], batch_size=None)

        self.assertEqual(next(iter(dataloader)), [sample])

    def test_structured_reasoning_output_is_saved_separately(self):
        handler = GenInferencerOutputHandler()

        handler.save_results(
            origin_prompt='question',
            prediction={
                'reasoning_content': 'private reasoning',
                'content': 'B',
            },
            idx=0,
            gold='B',
        )

        self.assertEqual(
            handler.results_dict['0'], {
                'origin_prompt': 'question',
                'prediction': 'B',
                'content': 'B',
                'reasoning_content': 'private reasoning',
                'gold': 'B',
            })

    def test_prediction_content_supports_multiple_sequences(self):
        output = [{
            'reasoning_content': 'r1',
            'content': 'A'
        }, {
            'reasoning_content': 'r2',
            'content': 'C'
        }]

        self.assertEqual(prediction_content(output), ['A', 'C'])

    def test_structured_generation_error_is_persisted(self):
        handler = GenInferencerOutputHandler()

        handler.save_results(
            origin_prompt='question',
            prediction={
                'reasoning_content': '',
                'content': '',
                'inference_error': 'TimeoutError: stream timed out',
            },
            idx=0,
            gold='B',
        )

        self.assertEqual(handler.results_dict['0']['prediction'], '')
        self.assertEqual(handler.results_dict['0']['inference_error'],
                         'TimeoutError: stream timed out')

    def test_jsonl_restore_keeps_valid_rows_and_quarantines_damage(self):
        for handler_type in (GenInferencerOutputHandler, ChatOutputHandler):
            with self.subTest(handler_type=handler_type.__name__), \
                    tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'tmp_predictions.jsonl'
                records = [
                    {'idx': '0', 'prediction': 'A'},
                    {'idx': '1', 'prediction': 'B'},
                ]
                path.write_text(
                    ''.join(json.dumps(record) + '\n'
                            for record in records) + '{"idx": "2",\n',
                    encoding='utf-8')

                handler = handler_type()
                restored = handler.restore_from_jsonl(
                    directory, path.name)

                self.assertEqual(set(restored), {'0', '1'})
                self.assertEqual(handler.dumped_indices, {'0', '1'})
                self.assertTrue(Path(str(path) + '.bak').is_file())
                clean_lines = path.read_text(encoding='utf-8').splitlines()
                self.assertEqual(len(clean_lines), 2)
                self.assertEqual(
                    [json.loads(line)['idx'] for line in clean_lines],
                    ['0', '1'])

    def test_jsonl_restore_does_not_overwrite_existing_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tmp_predictions.jsonl'
            Path(str(path) + '.bak').write_text('older backup',
                                                encoding='utf-8')
            path.write_text('{broken\n', encoding='utf-8')

            restored = GenInferencerOutputHandler().restore_from_jsonl(
                directory, path.name)

            self.assertEqual(restored, {})
            self.assertEqual(
                Path(str(path) + '.bak').read_text(encoding='utf-8'),
                'older backup')
            self.assertTrue(Path(str(path) + '.bak.1').is_file())

    def test_jsonl_restore_preserves_unicode_line_separators(self):
        for handler_type in (GenInferencerOutputHandler, ChatOutputHandler):
            with self.subTest(handler_type=handler_type.__name__), \
                    tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'tmp_predictions.jsonl'
                records = [
                    {
                        'idx': '0',
                        'prediction': 'before\u2028middle\u0085after',
                    },
                    {
                        'idx': '1',
                        'prediction': 'B',
                    },
                ]
                path.write_text(
                    ''.join(
                        json.dumps(record, ensure_ascii=False) + '\n'
                        for record in records),
                    encoding='utf-8')

                handler = handler_type()
                restored = handler.restore_from_jsonl(directory, path.name)

                self.assertEqual(set(restored), {'0', '1'})
                self.assertEqual(restored['0']['prediction'],
                                 'before\u2028middle\u0085after')
                self.assertFalse(Path(str(path) + '.bak').exists())
