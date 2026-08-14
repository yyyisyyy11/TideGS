import ast
import operator
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import unittest
from unittest import mock


ENGINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "strategies"
    / "tide_engine"
    / "engine.py"
)
RUNTIME_PATH = ENGINE_PATH.with_name("runtime.py")


def _source_tree(path=ENGINE_PATH):
    return ast.parse(path.read_text(encoding="utf-8"))


def _function_node(name, path=ENGINE_PATH):
    for node in _source_tree(path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Missing function {name}")


def _load_batch_resolver():
    function = _function_node("_resolve_single_rank_sophia_batch")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Optional": Optional,
        "resolve_curvature_schedule": lambda *, iteration, batch_size, interval,
        optimizer_step=None: (
            (iteration - 1) // batch_size + 1,
            ((iteration - 1) // batch_size) % interval == 0,
        )
        if optimizer_step is None
        or optimizer_step == (iteration - 1) // batch_size + 1
        else (_ for _ in ()).throw(
            ValueError("optimizer_step disagrees with the iteration-derived step")
        ),
    }
    exec(compile(module, str(ENGINE_PATH), "exec"), namespace)
    return namespace["_resolve_single_rank_sophia_batch"]


def _load_next_resident_runtime(*, curvature_due, sampled_s2):
    functions = [
        _function_node("resolve_next_resident_camera_ids", RUNTIME_PATH),
        _function_node("plan_and_start_resident_prefetch", RUNTIME_PATH),
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    sample_calls = []

    def _sample_s2_camera_ids(**kwargs):
        sample_calls.append(dict(kwargs))
        return list(sampled_s2)

    namespace = {
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "operator": operator,
        "get_current_and_next_camera_batches": lambda **kwargs: (
            None,
            SimpleNamespace(batch_indices=[10, 11]),
        ),
        "should_update_curvature": lambda step, interval: bool(
            curvature_due and step == 11 and interval == 10
        ),
        "sample_s2_camera_ids": _sample_s2_camera_ids,
    }
    exec(compile(module, str(RUNTIME_PATH), "exec"), namespace)
    return namespace, sample_calls


class _Range:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class SingleRankSophiaBatchContractTest(unittest.TestCase):
    def test_public_batch_entrypoint_exposes_s2_and_optimizer_clock(self):
        function = _function_node("clm_offload_train_one_batch")
        positional = [argument.arg for argument in function.args.args]
        self.assertEqual(positional[-2:], ["curvature_cameras", "optimizer_step"])
        self.assertEqual(len(function.args.defaults), 4)
        self.assertTrue(all(isinstance(value, ast.Constant) and value.value is None
                            for value in function.args.defaults[-2:]))

    def test_adam_ignores_sophia_only_inputs(self):
        resolve = _load_batch_resolver()
        s1 = [object(), object()]
        result = resolve(
            args=SimpleNamespace(paper_optimizer_algorithm="adam"),
            iteration=1,
            gradient_cameras=s1,
            curvature_cameras=[object()],
            optimizer_step=999,
        )
        self.assertEqual(result[:4], (False, None, False, []))
        self.assertEqual(result[4], s1)

    def test_curvature_step_requires_equal_s1_and_s2_batches(self):
        resolve = _load_batch_resolver()
        args = SimpleNamespace(
            paper_optimizer_algorithm="3dgs2_tr",
            paper_sophia_curvature_interval=10,
            bsz=2,
        )
        s1 = [object(), object()]
        s2 = [object(), object()]
        result = resolve(
            args=args,
            iteration=1,
            gradient_cameras=s1,
            curvature_cameras=s2,
            optimizer_step=1,
        )
        self.assertEqual(result[:3], (True, 1, True))
        self.assertEqual(result[3], s2)
        self.assertEqual(result[4], s1 + s2)

        with self.assertRaisesRegex(ValueError, "same global size"):
            resolve(
                args=args,
                iteration=1,
                gradient_cameras=s1,
                curvature_cameras=s2[:1],
                optimizer_step=1,
            )

    def test_explicit_step_must_match_iteration_derived_step(self):
        resolve = _load_batch_resolver()
        args = SimpleNamespace(
            paper_optimizer_algorithm="3dgs2_tr",
            paper_sophia_curvature_interval=10,
            bsz=2,
        )
        with self.assertRaisesRegex(ValueError, "iteration-derived"):
            resolve(
                args=args,
                iteration=3,
                gradient_cameras=[object(), object()],
                curvature_cameras=None,
                optimizer_step=1,
            )

    def test_s2_uses_shared_ranked_probe_and_separate_optimizer_rows(self):
        curvature_function = _function_node(
            "_run_single_rank_sophia_curvature_batch"
        )
        generator_calls = [
            node
            for node in ast.walk(curvature_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "estimate_seeded_curvature_sample"
        ]
        self.assertEqual(len(generator_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in generator_calls[0].keywords}
        self.assertEqual(
            set(keywords),
            {
                "base_seed",
                "optimizer_step",
                "microbatch_index",
                "sample_index",
                "sample_count",
                "rank",
            },
        )
        self.assertIsInstance(keywords["rank"], ast.Constant)
        self.assertEqual(keywords["rank"].value, 0)

        train_function = _function_node("clm_offload_train_one_batch")
        optimizer_calls = [
            node
            for node in ast.walk(train_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_gpu_resident_adam_step"
        ]
        self.assertEqual(len(optimizer_calls), 1)
        optimizer_keywords = {
            keyword.arg: keyword.value for keyword in optimizer_calls[0].keywords
        }
        self.assertIn("sparse_grad_local_ids", optimizer_keywords)
        self.assertIn("sparse_curvature_local_ids", optimizer_keywords)
        self.assertNotEqual(
            ast.dump(optimizer_keywords["sparse_grad_local_ids"]),
            ast.dump(optimizer_keywords["sparse_curvature_local_ids"]),
        )

    def test_runtime_wrapper_forwards_separate_curvature_rows(self):
        function = _function_node("run_gpu_resident_adam_step", RUNTIME_PATH)
        keyword_only = [argument.arg for argument in function.args.kwonlyargs]
        self.assertIn("sparse_curvature_local_ids", keyword_only)
        step_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ]
        self.assertEqual(len(step_calls), 1)
        dictionary_nodes = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Dict)
            and any(
                isinstance(key, ast.Constant)
                and key.value == "sparse_curvature_local_ids"
                for key in node.keys
            )
        ]
        self.assertEqual(len(dictionary_nodes), 1)

    def test_next_resident_batch_keeps_adam_and_non_curvature_s1_only(self):
        namespace, sample_calls = _load_next_resident_runtime(
            curvature_due=False,
            sampled_s2=[11, 12],
        )
        resolve = namespace["resolve_next_resident_camera_ids"]

        self.assertEqual(
            resolve(
                training_schedule=list(range(20)),
                iteration=1,
                batch_size=2,
                schedule_ordering="shuffle",
            ),
            ([10, 11], [], [10, 11]),
        )
        self.assertEqual(
            resolve(
                training_schedule=list(range(20)),
                iteration=19,
                batch_size=2,
                schedule_ordering="shuffle",
                optimizer_algorithm="3dgs2_tr",
                current_optimizer_step=10,
                curvature_interval=10,
                curvature_seed=37,
            ),
            ([10, 11], [], [10, 11]),
        )
        self.assertEqual(sample_calls, [])

    def test_next_curvature_batch_deduplicates_s1_s2_for_residency(self):
        namespace, sample_calls = _load_next_resident_runtime(
            curvature_due=True,
            sampled_s2=[11, 12],
        )
        resolve = namespace["resolve_next_resident_camera_ids"]

        result = resolve(
            training_schedule=list(range(20)),
            iteration=19,
            batch_size=2,
            schedule_ordering="shuffle",
            optimizer_algorithm="3dgs2_tr",
            current_optimizer_step=10,
            curvature_interval=10,
            curvature_seed=37,
        )

        self.assertEqual(result, ([10, 11], [11, 12], [10, 11, 12]))
        self.assertEqual(
            sample_calls,
            [
                {
                    "population_camera_ids": list(range(20)),
                    "s1_camera_ids": [10, 11],
                    "seed": 37,
                    "optimizer_step": 11,
                }
            ],
        )

    def test_next_curvature_union_uses_one_cull_and_existing_handoff(self):
        namespace, _ = _load_next_resident_runtime(
            curvature_due=True,
            sampled_s2=[11, 12],
        )
        clock = iter(range(100, 1000))
        namespace["time"] = SimpleNamespace(perf_counter_ns=lambda: next(clock))
        namespace["torch"] = SimpleNamespace(
            cuda=SimpleNamespace(nvtx=SimpleNamespace(range=lambda _: _Range()))
        )
        compute_block_sets = mock.Mock(
            return_value={
                "stream_in_blocks": [3],
                "next_resident_blocks": [1, 3],
                "keep_resident_blocks": [1],
                "evict_blocks": [2],
            }
        )
        namespace["compute_paper_block_sets"] = compute_block_sets
        plan = namespace["plan_and_start_resident_prefetch"]

        pending_copy_barrier = mock.Mock()
        storage = SimpleNamespace(
            get_visible_blocks_batch=mock.Mock(
                return_value=(7, {10: [1], 11: [2], 12: [3]})
            ),
            wait_for_pending_gpu_copies=pending_copy_barrier,
        )
        reader = SimpleNamespace(hint_future=mock.Mock(return_value=1))
        double_buffer = SimpleNamespace(start_prefetch=mock.Mock())

        result = plan(
            storage_adapter=storage,
            training_schedule=list(range(20)),
            iteration=19,
            batch_size=2,
            current_block_ids=[1],
            schedule_ordering="shuffle",
            current_resident_blocks=[1, 2],
            current_resident_recency_scores={},
            resident_selection_policy="topc_balanced",
            resident_lambda=0.3,
            resident_recency_decay=0.95,
            resident_capacity_blocks=8,
            balanced_seed_fraction=0.25,
            active_block_reader=reader,
            double_buffer=double_buffer,
            optimizer_algorithm="3dgs2_tr",
            current_optimizer_step=10,
            curvature_interval=10,
            curvature_seed=37,
        )

        storage.get_visible_blocks_batch.assert_called_once_with([10, 11, 12])
        selection_kwargs = compute_block_sets.call_args.kwargs
        self.assertEqual(
            selection_kwargs["next_camera_ids_override"], [10, 11, 12]
        )
        self.assertEqual(
            selection_kwargs["next_camera_blocks_override"],
            {10: [1], 11: [2], 12: [3]},
        )
        reader.hint_future.assert_called_once_with([3])
        prefetch_kwargs = double_buffer.start_prefetch.call_args.kwargs
        self.assertEqual(prefetch_kwargs["iteration"], 21)
        self.assertEqual(prefetch_kwargs["visible_block_ids"], [1, 3])
        self.assertIs(
            prefetch_kwargs["before_target_reuse"], pending_copy_barrier
        )
        self.assertTrue(prefetch_kwargs["defer_resident_copy"])
        self.assertEqual(result["next_s1_camera_ids"], [10, 11])
        self.assertEqual(result["next_s2_camera_ids"], [11, 12])
        self.assertEqual(result["next_resident_camera_ids"], [10, 11, 12])

    def test_single_rank_engine_passes_optimizer_clock_to_next_plan(self):
        function = _function_node("clm_offload_train_one_batch")
        plan_submissions = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "submit_resident_plan"
            and any(
                isinstance(argument, ast.Name)
                and argument.id == "_plan_and_start_resident_prefetch"
                for argument in node.args
            )
        ]
        self.assertEqual(len(plan_submissions), 1)
        keyword_values = {
            keyword.arg: keyword.value for keyword in plan_submissions[0].keywords
        }
        self.assertEqual(
            {
                "optimizer_algorithm",
                "current_optimizer_step",
                "curvature_interval",
                "curvature_seed",
            }.difference(keyword_values),
            set(),
        )
        self.assertIsInstance(keyword_values["current_optimizer_step"], ast.Name)
        self.assertEqual(
            keyword_values["current_optimizer_step"].id, "optimizer_step"
        )


if __name__ == "__main__":
    unittest.main()
