import ast
from pathlib import Path
import unittest


TRAIN_PATH = Path(__file__).resolve().parents[1] / "train_tidegs.py"


def _function_node(name):
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_synchronized_phase():
    function = _function_node("_run_synchronized_training_phase")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return namespace["_run_synchronized_training_phase"]


class _Context:
    def __init__(self, *, enabled, rank=0, gathered_statuses=None):
        self.enabled = enabled
        self.rank = rank
        self.gathered_statuses = gathered_statuses
        self.payloads = []

    def all_gather_object(self, payload):
        self.payloads.append(payload)
        if self.gathered_statuses is None:
            return [payload]
        return self.gathered_statuses(payload)


class SynchronizedTrainingPhaseTest(unittest.TestCase):
    def setUp(self):
        self.run_phase = _load_synchronized_phase()

    def test_non_distributed_result_and_original_error_are_preserved(self):
        context = _Context(enabled=False)
        marker = object()
        self.assertIs(
            self.run_phase(
                context,
                phase="local phase",
                operation=lambda: marker,
            ),
            marker,
        )
        self.assertEqual(context.payloads, [])

        original = OSError("local disk failure")
        with self.assertRaises(OSError) as raised:
            self.run_phase(
                context,
                phase="local phase",
                operation=lambda: (_ for _ in ()).throw(original),
            )
        self.assertIs(raised.exception, original)

    def test_local_failure_is_published_and_chained_before_return(self):
        original = OSError("rank 0 disk failure")

        def gathered(payload):
            return [payload, {"rank": 1, "error": None}]

        context = _Context(enabled=True, rank=0, gathered_statuses=gathered)
        with self.assertRaisesRegex(
            RuntimeError,
            "Distributed training dirty flush failed: "
            "rank=0 OSError: rank 0 disk failure",
        ) as raised:
            self.run_phase(
                context,
                phase="dirty flush",
                operation=lambda: (_ for _ in ()).throw(original),
            )

        self.assertIs(raised.exception.__cause__, original)
        self.assertEqual(
            context.payloads,
            [{"rank": 0, "error": "OSError: rank 0 disk failure"}],
        )

    def test_peer_failure_stops_successful_rank_without_hiding_failure(self):
        marker = object()

        def gathered(payload):
            return [payload, {"rank": 1, "error": "OSError: peer disk failure"}]

        context = _Context(enabled=True, rank=0, gathered_statuses=gathered)
        with self.assertRaisesRegex(
            RuntimeError,
            "Distributed training checkpoint drain failed: "
            "rank=1 OSError: peer disk failure",
        ) as raised:
            self.run_phase(
                context,
                phase="checkpoint drain",
                operation=lambda: marker,
            )

        self.assertIsNone(raised.exception.__cause__)
        self.assertEqual(context.payloads, [{"rank": 0, "error": None}])

    def test_rank_without_operation_still_joins_prune_failure_sync(self):
        def gathered(payload):
            return [
                {"rank": 0, "error": "OSError: rank 0 prune failure"},
                payload,
            ]

        context = _Context(enabled=True, rank=1, gathered_statuses=gathered)
        with self.assertRaisesRegex(
            RuntimeError,
            "Distributed training checkpoint history prune failed: "
            "rank=0 OSError: rank 0 prune failure",
        ):
            self.run_phase(
                context,
                phase="checkpoint history prune",
                operation=None,
            )

        self.assertEqual(context.payloads, [{"rank": 1, "error": None}])

    def test_successful_distributed_phase_returns_after_one_collective(self):
        marker = object()
        context = _Context(enabled=True, rank=1)
        result = self.run_phase(
            context,
            phase="successful phase",
            operation=lambda: marker,
        )

        self.assertIs(result, marker)
        self.assertEqual(context.payloads, [{"rank": 1, "error": None}])


class SynchronizedTrainingCallSiteTest(unittest.TestCase):
    def test_storage_phases_use_synchronized_helper(self):
        training = _function_node("training")
        synchronized_calls = []
        for call in (
            node for node in ast.walk(training) if isinstance(node, ast.Call)
        ):
            if not (
                isinstance(call.func, ast.Name)
                and call.func.id == "_run_synchronized_training_phase"
            ):
                continue
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            phase = keywords.get("phase")
            if isinstance(phase, ast.Constant) and isinstance(phase.value, str):
                synchronized_calls.append((phase.value, keywords["operation"], call))

        calls_by_phase = {
            phase: (operation, call)
            for phase, operation, call in synchronized_calls
        }
        expected_attributes = {
            "resident dirty flush": "flush_resident_dirty",
            "checkpoint writeback drain": "drain_cache_writebacks",
            "shutdown resident dirty flush": "flush_resident_dirty",
        }
        for phase, attribute in expected_attributes.items():
            with self.subTest(phase=phase):
                operation, _ = calls_by_phase[phase]
                self.assertIsInstance(operation, ast.Attribute)
                self.assertEqual(operation.attr, attribute)

        for phase in (
            "output initialization",
            "prebuilt manifest validation",
            "distributed checkpoint resume validation",
            "scene initialization",
            "rank 0 storage and schedule initialization",
            "worker storage initialization",
            "legacy metrics finalize",
            "compaction metrics write",
            "storage shutdown",
            "distributed metrics finalize",
            "double buffer shutdown",
        ):
            with self.subTest(phase=phase):
                self.assertIn(phase, calls_by_phase)

        storage_shutdown, _ = calls_by_phase["storage shutdown"]
        self.assertIsInstance(storage_shutdown, ast.Lambda)
        shutdown_call = next(
            node
            for node in ast.walk(storage_shutdown)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "shutdown"
        )
        compact_storage = {
            keyword.arg: keyword.value for keyword in shutdown_call.keywords
        }["compact_storage"]
        self.assertIsInstance(compact_storage, ast.Constant)
        self.assertIs(compact_storage.value, False)

        double_buffer_shutdown, _ = calls_by_phase["double buffer shutdown"]
        self.assertIsInstance(double_buffer_shutdown, ast.Name)
        self.assertEqual(double_buffer_shutdown.id, "shutdown_double_buffer_gpu")

        schedule_broadcast = next(
            call
            for call in ast.walk(training)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "broadcast_object"
            and any(
                isinstance(argument, ast.Name)
                and argument.id == "ssd_training_schedule"
                for argument in call.args
            )
        )
        self.assertLess(
            calls_by_phase["rank 0 storage and schedule initialization"][1].lineno,
            schedule_broadcast.lineno,
        )
        self.assertLess(
            schedule_broadcast.lineno,
            calls_by_phase["worker storage initialization"][1].lineno,
        )

        prune_operation, prune_call = calls_by_phase[
            "checkpoint history prune"
        ]
        self.assertIsInstance(prune_operation, ast.Name)
        self.assertEqual(prune_operation.id, "prune_operation")

        rank0_prune_if = next(
            node
            for node in ast.walk(training)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Attribute) and child.attr == "is_rank0"
                for child in ast.walk(node.test)
            )
            and any(
                isinstance(child, ast.Name) and child.id == "prune_operation"
                for statement in node.body
                for child in ast.walk(statement)
            )
        )
        self.assertGreater(prune_call.lineno, rank0_prune_if.end_lineno)

    def test_rank0_planning_errors_are_synchronized_before_broadcast(self):
        cases = {
            "_prepare_distributed_block_owner": "block owner preparation",
            "_build_distributed_plan": "initial distributed planning",
            "_preview_distributed_plan": "predictive distributed planning",
            "_finalize_distributed_plan": "final distributed planning",
        }
        for function_name, expected_phase in cases.items():
            with self.subTest(function=function_name):
                function = _function_node(function_name)
                synchronized_call = next(
                    call
                    for call in ast.walk(function)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_run_synchronized_training_phase"
                    and any(
                        keyword.arg == "phase"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value == expected_phase
                        for keyword in call.keywords
                    )
                )
                broadcast_call = next(
                    call
                    for call in ast.walk(function)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "broadcast_object"
                )
                self.assertLess(synchronized_call.lineno, broadcast_call.lineno)

        owner_function = _function_node("_prepare_distributed_block_owner")
        owner_broadcast = next(
            call
            for call in ast.walk(owner_function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "broadcast_object"
        )
        owner_configuration = next(
            call
            for call in ast.walk(owner_function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_run_synchronized_training_phase"
            and any(
                keyword.arg == "phase"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "block owner configuration"
                for keyword in call.keywords
            )
        )
        self.assertLess(owner_broadcast.lineno, owner_configuration.lineno)


if __name__ == "__main__":
    unittest.main()
