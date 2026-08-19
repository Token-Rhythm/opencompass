"""Unit tests for ParallelGenInferencer."""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel import \
    ParallelGenInferencer


class TestParallelGenInferencer(unittest.TestCase):
    """Test cases for ParallelGenInferencer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_model = MagicMock()
        self.mock_model.is_api = True
        self.mock_model.generate = MagicMock()
        self.mock_model.generate_from_template = MagicMock(
            return_value=['test_output'])
        self.mock_model.parse_template = MagicMock(
            return_value='parsed_prompt')
        self.mock_model.get_token_len = MagicMock(return_value=10)
        self.mock_model.generation_kwargs = {}

    def test_initialization(self):
        """Test ParallelGenInferencer initialization."""
        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512,
                                           max_infer_workers=4)
        self.assertEqual(inferencer.max_infer_workers, 4)
        self.assertIsNone(inferencer.progress_tracker)
        self.assertEqual(inferencer.max_out_len, 512)

    def test_initialization_ignores_injected_batch_size(self):
        """Standard OpenICL tasks inject a model batch size."""
        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512,
                                           batch_size=128,
                                           max_infer_workers=4)

        self.assertEqual(inferencer.batch_size, 1)
        self.assertEqual(inferencer.max_infer_workers, 4)

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512)
        self.assertIsNone(inferencer.max_infer_workers)
        self.assertIsNone(inferencer.progress_tracker)

    def test_resolve_max_workers_from_config(self):
        """Test _resolve_max_workers uses max_infer_workers."""
        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512,
                                           max_infer_workers=8)
        result = inferencer._resolve_max_workers()
        self.assertEqual(result, 8)

    def test_resolve_max_workers_from_model(self):
        """Test _resolve_max_workers uses model.max_workers."""
        self.mock_model.max_workers = 6
        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512)
        result = inferencer._resolve_max_workers()
        self.assertEqual(result, 6)

    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.os.cpu_count'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.getattr'  # noqa
    )
    def test_resolve_max_workers_default(self, mock_getattr, mock_cpu_count):
        """Test _resolve_max_workers uses default calculation."""
        mock_cpu_count.return_value = 8

        # Make getattr return None for max_workers
        def getattr_side_effect(obj, attr, default=None):
            if attr == 'max_workers':
                return None
            return getattr(obj, attr, default)

        mock_getattr.side_effect = getattr_side_effect

        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512)
        inferencer.max_infer_workers = None
        result = inferencer._resolve_max_workers()
        # Should be min(32, cpu_count + 4) = min(32, 12) = 12
        self.assertEqual(result, 12)

    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.os.cpu_count'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.getattr'  # noqa
    )
    def test_resolve_max_workers_max_limit(self, mock_getattr, mock_cpu_count):
        """Test _resolve_max_workers respects max limit."""
        mock_cpu_count.return_value = 100

        # Make getattr return None for max_workers
        def getattr_side_effect(obj, attr, default=None):
            if attr == 'max_workers':
                return None
            return getattr(obj, attr, default)

        mock_getattr.side_effect = getattr_side_effect

        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512)
        inferencer.max_infer_workers = None
        result = inferencer._resolve_max_workers()
        # Should be min(32, 100 + 4) = 32
        self.assertEqual(result, 32)

    def test_progress_update(self):
        """Test _progress_update method."""
        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512)
        mock_tracker = MagicMock()
        inferencer.progress_tracker = mock_tracker

        inferencer._progress_update(5)
        mock_tracker.incr.assert_called_once_with(5)

    def test_progress_update_no_tracker(self):
        """Test _progress_update when tracker is None."""
        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512)
        # Should not raise error
        inferencer._progress_update(5)

    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.os.path.exists'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.os.makedirs'  # noqa
    )
    @patch.object(ParallelGenInferencer,
                  '_resolve_max_workers',
                  return_value=2)
    def test_inference_basic(self, mock_resolve, mock_makedirs, mock_exists):
        """Test inference method basic functionality."""
        mock_exists.return_value = False

        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512,
                                           output_json_filepath=self.temp_dir,
                                           output_json_filename='test.json')

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [[0], [1]]
        mock_retriever.dataset_reader = MagicMock()
        mock_retriever.dataset_reader.output_column = None

        # Mock the get_generation_prompt_list_from_retriever_indices method
        inferencer.get_generation_prompt_list_from_retriever_indices = MagicMock(  # noqa
            return_value=['prompt1', 'prompt2'])

        result = inferencer.inference(mock_retriever)

        self.assertIsInstance(result, list)
        mock_retriever.retrieve.assert_called_once()

    def test_inference_returns_dataset_order_after_out_of_order_completion(self):
        def generate(entries, **kwargs):
            prompt = entries[0]
            if prompt == 'slow-first':
                time.sleep(0.05)
            return [prompt]

        self.mock_model.generate_from_template.side_effect = generate
        inferencer = ParallelGenInferencer(
            model=self.mock_model,
            max_out_len=512,
            max_infer_workers=2,
            output_json_filepath=self.temp_dir,
            output_json_filename='ordered.json',
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [[], []]
        mock_retriever.dataset_reader.output_column = None
        inferencer.get_generation_prompt_list_from_retriever_indices = (
            MagicMock(return_value=['slow-first', 'fast-second']))

        result = inferencer.inference(mock_retriever)

        self.assertEqual(result, ['slow-first', 'fast-second'])

    def test_inference_persists_failed_sample_and_continues(self):
        def generate(entries, **kwargs):
            if entries[0] == 'bad-first':
                raise TimeoutError('stream timed out')
            return ['good-second']

        self.mock_model.generate_from_template.side_effect = generate
        inferencer = ParallelGenInferencer(
            model=self.mock_model,
            max_out_len=512,
            max_infer_workers=2,
            output_json_filepath=self.temp_dir,
            output_json_filename='with_error.json',
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [[], []]
        mock_retriever.dataset_reader.output_column = None
        inferencer.get_generation_prompt_list_from_retriever_indices = (
            MagicMock(return_value=['bad-first', 'good-second']))

        result = inferencer.inference(mock_retriever)

        with open(os.path.join(self.temp_dir, 'with_error.json'),
                  encoding='utf-8') as file:
            persisted = json.load(file)
        self.assertEqual(result, ['', 'good-second'])
        self.assertEqual(persisted['0']['prediction'], '')
        self.assertEqual(persisted['0']['content'], '')
        self.assertIn('TimeoutError: stream timed out',
                      persisted['0']['inference_error'])
        self.assertEqual(persisted['1']['prediction'], 'good-second')

    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.os.path.exists'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.os.makedirs'  # noqa
    )
    @patch.object(ParallelGenInferencer,
                  '_resolve_max_workers',
                  return_value=2)
    def test_inference_with_progress_tracker(self, mock_resolve, mock_makedirs,
                                             mock_exists):
        """Test inference with progress tracker."""
        mock_exists.return_value = False

        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512,
                                           output_json_filepath=self.temp_dir,
                                           output_json_filename='test.json')

        mock_tracker = MagicMock()
        inferencer.progress_tracker = mock_tracker

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [[0], [1]]
        mock_retriever.dataset_reader = MagicMock()
        mock_retriever.dataset_reader.output_column = None

        inferencer.get_generation_prompt_list_from_retriever_indices = MagicMock(  # noqa
            return_value=['prompt1', 'prompt2'])

        inferencer.inference(mock_retriever)

        # Verify tracker was used
        mock_tracker.set_total.assert_called_once_with(2)

    def test_inference_persists_reasoning_and_returns_only_content(self):
        self.mock_model.generate_from_template.return_value = [{
            'reasoning_content':
            'reasoning says A before correcting itself',
            'content':
            'B',
        }]
        inferencer = ParallelGenInferencer(
            model=self.mock_model,
            max_out_len=512,
            max_infer_workers=1,
            output_json_filepath=self.temp_dir,
            output_json_filename='structured.json',
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [[]]
        mock_retriever.dataset_reader.output_column = None
        inferencer.get_generation_prompt_list_from_retriever_indices = (
            MagicMock(return_value=['prompt']))

        result = inferencer.inference(mock_retriever)

        with open(os.path.join(self.temp_dir, 'structured.json'),
                  encoding='utf-8') as file:
            persisted = json.load(file)['0']
        self.assertEqual(result, ['B'])
        self.assertEqual(persisted['prediction'], 'B')
        self.assertEqual(persisted['content'], 'B')
        self.assertEqual(persisted['reasoning_content'],
                         'reasoning says A before correcting itself')

    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.GenInferencerOutputHandler'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.osp.exists'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.os.path.exists'  # noqa
    )
    @patch(
        'opencompass.openicl.icl_inferencer.icl_gen_inferencer_parallel.os.makedirs'  # noqa
    )
    @patch.object(ParallelGenInferencer,
                  '_resolve_max_workers',
                  return_value=2)
    def test_inference_with_resume(self, mock_resolve, mock_makedirs,
                                   mock_exists, mock_osp_exists,
                                   mock_handler_class):
        """Test inference with resume from tmp file."""
        mock_exists.return_value = True
        mock_osp_exists.return_value = False  # File doesn't exist for removal

        inferencer = ParallelGenInferencer(model=self.mock_model,
                                           max_out_len=512,
                                           output_json_filepath=self.temp_dir,
                                           output_json_filename='test.json')

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [[0], [1]]
        mock_retriever.dataset_reader = MagicMock()
        mock_retriever.dataset_reader.output_column = None

        inferencer.get_generation_prompt_list_from_retriever_indices = MagicMock(  # noqa
            return_value=['prompt1', 'prompt2'])

        # Mock the output handler to simulate resume from existing results
        mock_output_handler = MagicMock()
        mock_output_handler.results_dict = {'0': {'prediction': 'existing'}}
        mock_output_handler.restore_from_jsonl = MagicMock(
            return_value={'0': {
                'prediction': 'existing'
            }})
        mock_output_handler.save_results = MagicMock()
        mock_output_handler.write_to_jsonl = MagicMock()
        mock_output_handler.write_to_json = MagicMock()
        mock_handler_class.return_value = mock_output_handler

        inferencer.inference(mock_retriever)

        # Should only process index 1 (0 is already done)
        mock_retriever.retrieve.assert_called_once()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
