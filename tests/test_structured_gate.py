import importlib.util
import logging
import os
import os.path as osp
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock


FAIL_MSG = 'Failed to obtain answer via API.'


def _dump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_inference():
    """Load vlmeval/inference.py with stubbed heavy dependencies.

    Follows the same pattern as tests/test_inference_api.py so the module can
    be exercised without installing torch and the rest of requirements.txt.
    """
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']

    smp = types.ModuleType('vlmeval.smp')
    smp.dump = _dump
    smp.load = _load
    smp.get_logger = lambda name: logging.getLogger(name)
    smp.get_pred_file_format = lambda: 'pkl'
    smp.get_pred_file_path = (
        lambda work_dir, model_name, dataset_name, **kwargs:
        osp.join(work_dir, f'{model_name}_{dataset_name}.pkl')
    )
    smp.get_rank_and_world_size = lambda: (0, 1)

    smp_log = types.ModuleType('vlmeval.smp.log')
    smp_log.setup_subprocess_logger = lambda *args, **kwargs: None

    utils = types.ModuleType('vlmeval.utils')
    utils.__path__ = ['vlmeval/utils']
    utils.track_progress_rich = lambda func, tasks, **kwargs: [func(t) for t in tasks]

    config = types.ModuleType('vlmeval.config')
    config.supported_VLM = {}

    torch = types.ModuleType('torch')
    torch_dist = types.ModuleType('torch.distributed')
    torch_dist.barrier = lambda *args, **kwargs: None
    torch.distributed = torch_dist

    tqdm_mod = types.ModuleType('tqdm')
    tqdm_mod.tqdm = lambda x, **kwargs: x

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.smp': smp,
        'vlmeval.smp.log': smp_log,
        'vlmeval.utils': utils,
        'vlmeval.config': config,
        'torch': torch,
        'torch.distributed': torch_dist,
        'tqdm': tqdm_mod,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.inference',
            'vlmeval/inference.py',
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules['vlmeval.inference'] = module
        spec.loader.exec_module(module)
        sys.modules.pop('vlmeval.inference', None)
        return module


class FakeDataset:
    dataset_name = 'FakeGate'

    def __init__(self, index):
        # infer_data_job only needs dict-like column access on .data
        self.data = {'index': list(index)}


def _run_infer_data_job(index, data_all, env=None):
    inference = _load_inference()
    dataset = FakeDataset(index)

    def fake_infer_data(**kwargs):
        # mimic infer_data: write the per-rank results pickle, return the model
        _dump(data_all, kwargs['out_file'])
        return kwargs['model']

    inference.infer_data = fake_infer_data

    old_env = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            inference.infer_data_job(
                model=object(),  # no split_thinking attr -> default split func
                work_dir=tmpdir,
                model_name='mock',
                dataset=dataset,
            )
            result_file = osp.join(tmpdir, 'mock_FakeGate.pkl')
            return _load(result_file)
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestStructuredGateMainBranch(unittest.TestCase):
    """The non-SPLIT_THINK assembly branch in vlmeval/inference.py."""

    def test_all_structured_unchanged(self):
        data_all = {
            '0': {'prediction': 'A', 'extra_records': {'tool_call_count': 1}},
            '1': {'prediction': 'B', 'extra_records': {'tool_call_count': 2}},
        }
        result = _run_infer_data_job(['0', '1'], data_all)
        self.assertEqual(result['prediction'], ['A', 'B'])
        self.assertEqual(
            result['extra_records'],
            [{'tool_call_count': 1}, {'tool_call_count': 2}],
        )

    def test_none_structured_unchanged(self):
        data_all = {'0': 'plain answer', '1': FAIL_MSG}
        result = _run_infer_data_job(['0', '1'], data_all)
        self.assertEqual(result['prediction'], ['plain answer', FAIL_MSG])
        self.assertNotIn('extra_records', result)

    def test_mixed_failure_keeps_structured_records(self):
        # the bug in issue #1665: one failed sample used to destroy the
        # structured output of every other sample in the dataset
        data_all = {
            '0': {'prediction': 'A', 'extra_records': {'tool_call_count': 1}},
            '1': FAIL_MSG,
            '2': {'prediction': 'B', 'extra_records': {'tool_call_count': 2}},
        }
        result = _run_infer_data_job(['0', '1', '2'], data_all)
        self.assertEqual(result['prediction'], ['A', FAIL_MSG, 'B'])
        self.assertEqual(
            result['extra_records'],
            [{'tool_call_count': 1}, {}, {'tool_call_count': 2}],
        )


class TestStructuredGateSplitThinkBranch(unittest.TestCase):
    """The SPLIT_THINK assembly branch shares the same gate."""

    ENV = {'SPLIT_THINK': '1'}

    def test_all_structured_unchanged(self):
        data_all = {
            '0': {'prediction': 'A', 'extra_records': {'tool_call_count': 1}},
            '1': {'prediction': 'B</think>real', 'extra_records': {}},
        }
        result = _run_infer_data_job(['0', '1'], data_all, env=self.ENV)
        self.assertEqual(result['prediction'], ['A', 'real'])
        # default split_thinking: '<think>' missing -> thinking keeps '</think>'
        self.assertEqual(result['thinking'], ['', 'B</think>'])
        self.assertEqual(result['extra_records'], [{'tool_call_count': 1}, {}])

    def test_none_structured_unchanged(self):
        data_all = {'0': 'plain answer', '1': FAIL_MSG}
        result = _run_infer_data_job(['0', '1'], data_all, env=self.ENV)
        self.assertEqual(result['prediction'], ['plain answer', FAIL_MSG])
        self.assertEqual(result['thinking'], ['', ''])
        self.assertNotIn('extra_records', result)

    def test_mixed_failure_keeps_structured_records(self):
        data_all = {
            '0': {'prediction': 'A</think>ans', 'extra_records': {'tool_call_count': 1}},
            '1': FAIL_MSG,
            '2': {'prediction': 'B', 'extra_records': {'tool_call_count': 2}},
        }
        result = _run_infer_data_job(['0', '1', '2'], data_all, env=self.ENV)
        # failed row keeps the fail_msg as its prediction and gets empty thinking
        self.assertEqual(result['prediction'], ['ans', FAIL_MSG, 'B'])
        self.assertEqual(result['thinking'], ['A</think>', '', ''])
        self.assertEqual(
            result['extra_records'],
            [{'tool_call_count': 1}, {}, {'tool_call_count': 2}],
        )


if __name__ == '__main__':
    unittest.main()
