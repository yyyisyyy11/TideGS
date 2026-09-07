import ast
import glob
import os
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_READERS = REPO_ROOT / "scene" / "dataset_readers.py"


def _load_resolver():
    tree = ast.parse(DATASET_READERS.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_city_ply_path"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"glob": glob, "os": os}
    exec(compile(module, str(DATASET_READERS), "exec"), namespace)
    return namespace["_resolve_city_ply_path"]


class CityPrebuiltManifestTest(unittest.TestCase):
    def test_prebuilt_manifest_does_not_require_ply(self):
        resolve = _load_resolver()
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(resolve(root, "", "", allow_missing=True), "")

    def test_regular_run_still_requires_ply(self):
        resolve = _load_resolver()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                resolve(root, "", "")

    def test_explicit_ply_takes_precedence(self):
        resolve = _load_resolver()
        self.assertEqual(
            resolve("/unused", "/data/explicit.ply", "/data/resume.ply"),
            "/data/explicit.ply",
        )


if __name__ == "__main__":
    unittest.main()
