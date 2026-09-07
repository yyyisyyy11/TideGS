import ast
import time
import unittest
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "storage" / "async_pipeline.py"
ADAPTER_PATH = ROOT / "storage" / "tide_storage_adapter.py"


def _method_node(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _load_methods(path, class_name, method_names, namespace):
    nodes = [_method_node(path, class_name, name) for name in method_names]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return [namespace[name] for name in method_names]


class _Cache:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.calls = []

    def prefetch_future(self, block_ids, *, target_iteration=None):
        self.calls.append((list(block_ids), target_iteration))
        self.pipeline.running = False
        return len(block_ids)


class AsyncPrefetchTargetTest(unittest.TestCase):
    def test_future_queue_preserves_target_iteration_for_worker(self):
        request_prefetch, future_worker = _load_methods(
            PIPELINE_PATH,
            "AsyncPipeline",
            ["request_prefetch", "_future_prefetch_worker"],
            {
                "Empty": Empty,
                "List": List,
                "Optional": Optional,
                "time": time,
            },
        )
        pipeline = SimpleNamespace(
            urgent_prefetch_queue=Queue(),
            future_prefetch_queue=Queue(),
            running=True,
            stats={
                "future_prefetch_time": 0.0,
                "future_prefetch_jobs": 0,
                "future_prefetch_blocks": 0,
            },
        )
        pipeline.cache = _Cache(pipeline)

        request_prefetch(
            pipeline,
            iteration=7,
            needed_blocks=[1],
            future_blocks=[2, 3],
            future_target_iteration=23,
        )
        self.assertEqual(pipeline.urgent_prefetch_queue.get_nowait(), (7, [1]))
        future_worker(pipeline)

        self.assertEqual(pipeline.cache.calls, [([2, 3], 23)])
        self.assertEqual(pipeline.stats["future_prefetch_jobs"], 1)

    def test_adapter_uses_next_batch_iteration_as_future_target(self):
        (prefetch_for_next_iteration,) = _load_methods(
            ADAPTER_PATH,
            "TideStorageAdapter",
            ["prefetch_for_next_iteration"],
            {
                "List": List,
                "get_current_and_next_camera_batches": lambda **kwargs: (
                    SimpleNamespace(batch_indices=[10]),
                    SimpleNamespace(batch_indices=[20]),
                ),
            },
        )
        calls = []
        adapter = SimpleNamespace(
            execution_metrics={"prefetch_requests": 0},
            get_visible_blocks=lambda camera_id: {
                10: [1, 2],
                20: [2, 3],
            }[camera_id],
            pipeline=SimpleNamespace(
                request_prefetch=lambda **kwargs: calls.append(kwargs)
            ),
        )

        prefetch_for_next_iteration(
            adapter,
            iteration=7,
            batch_size=16,
            training_schedule=[10, 20],
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["iteration"], 7)
        self.assertEqual(calls[0]["future_target_iteration"], 23)
        self.assertEqual(calls[0]["needed_blocks"], [1, 2])
        self.assertEqual(calls[0]["future_blocks"], [2, 3])


if __name__ == "__main__":
    unittest.main()
