"""CLI/config canonicalization and legacy timeout compatibility without torch."""

import argparse
import ast
from contextlib import redirect_stderr
from datetime import timedelta
import io
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_micro_pipeline_context_lifecycle import context_class
from utils.pipeline_mode import normalize_pipeline_mode


ROOT = Path(__file__).resolve().parents[1]


def arguments_parser():
    path = ROOT / "arguments/__init__.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name in {"GroupParams", "ParamGroup", "AuxiliaryParams"}]
    namespace = {"ArgumentParser": argparse.ArgumentParser,
                 "normalize_pipeline_mode": normalize_pipeline_mode}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    parser = argparse.ArgumentParser()
    params = namespace["AuxiliaryParams"](parser)
    return parser, params


def timeout_reader():
    path = ROOT / "utils/distributed.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "_positive_timeout")
    namespace = {"os": os, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_positive_timeout"]


class MicroPipelineModeTest(unittest.TestCase):
    def test_cli_defaults_off_and_canonicalizes_old_and_new_spellings(self):
        parser, _ = arguments_parser()
        self.assertEqual(parser.parse_args([]).tide_forward_pipeline, "off")
        for value, expected in (("off", "off"), ("micro", "micro"),
                                ("pc1", "micro"), (" PC1 ", "micro")):
            with self.subTest(value=value):
                args = parser.parse_args(["--tide_forward_pipeline", value])
                self.assertEqual(args.tide_forward_pipeline, expected)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--tide_forward_pipeline", "unsupported"])

    def test_extracted_checkpoint_args_are_canonicalized(self):
        _, params = arguments_parser()
        extracted = params.extract(SimpleNamespace(tide_forward_pipeline="pc1"))
        self.assertEqual(extracted.tide_forward_pipeline, "micro")
        for invalid in (None, "", "pipeline", "micro_pipeline"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                normalize_pipeline_mode(invalid)

    def test_runtime_normalizes_before_nondistributed_early_return(self):
        args = SimpleNamespace(tide_distributed_mode="off", tide_forward_pipeline="pc1")
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
            context = context_class(None).initialize(args)
        self.assertFalse(context.enabled)
        self.assertEqual(args.tide_forward_pipeline, "micro")

    def test_legacy_mode_starts_micro_controller_with_old_timeout_environment(self):
        store_timeouts = []
        store = SimpleNamespace(port=4321, set_timeout=store_timeouts.append,
                                check=lambda keys: False, set=lambda *args: None)
        dist = SimpleNamespace(
            is_initialized=lambda: True, get_rank=lambda: 0, get_world_size=lambda: 2,
            new_group=lambda **kwargs: "gloo", TCPStore=lambda **kwargs: store,
            broadcast_object_list=lambda *args, **kwargs: None,
            barrier=lambda **kwargs: None, destroy_process_group=lambda *args: None,
        )
        args = SimpleNamespace(tide_distributed_mode="gaussian_sharded", tide_forward_pipeline="pc1")
        environment = {
            "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2", "MASTER_ADDR": "127.0.0.1",
            "TIDE_PC1_ERROR_TIMEOUT_SEC": "3", "TIDE_PC1_CONTROL_IO_TIMEOUT_SEC": "0.25",
            "TIDE_PC1_STAGE_TIMEOUT_SEC": "7",
        }
        with patch.dict(os.environ, environment, clear=True):
            context = context_class(dist).initialize(args)
            try:
                self.assertEqual(args.tide_forward_pipeline, "micro")
                self.assertEqual(context.micro_pipeline_error_controller.stage_timeout, 7.0)
                self.assertEqual(store_timeouts, [timedelta(seconds=0.25)])
            finally:
                context.close()

    def test_new_timeouts_override_old_values_and_invalid_new_values_fail(self):
        read = timeout_reader()
        for suffix, default in (("ERROR_TIMEOUT_SEC", "30"),
                                ("CONTROL_IO_TIMEOUT_SEC", "1"),
                                ("STAGE_TIMEOUT_SEC", "60")):
            current, legacy = "TIDE_MICRO_PIPELINE_" + suffix, "TIDE_PC1_" + suffix
            with self.subTest(suffix=suffix):
                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(read(current, default), float(default))
                with patch.dict(os.environ, {legacy: "4"}, clear=True):
                    self.assertEqual(read(current, default), 4.0)
                with patch.dict(os.environ, {legacy: "4", current: "2"}, clear=True):
                    self.assertEqual(read(current, default), 2.0)
                for invalid in ("0", "-1", "nan", "inf", ""):
                    with patch.dict(os.environ, {legacy: "4", current: invalid}, clear=True):
                        with self.assertRaises(ValueError):
                            read(current, default)


if __name__ == "__main__":
    unittest.main()
