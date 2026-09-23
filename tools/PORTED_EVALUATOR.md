> 注：本文写于 2026-09-21 首次部署时，当时这四个文件**未**被 git 跟踪。
> 在 `handoff/knn3-4gpu` 分支（以及 `feat/tidegs-four-gpu`）上它们**已经被跟踪**，
> 所以下文"untracked / worktree remove 会删掉"的提醒只对旧部署有效。

# Ported upstream evaluator (untracked files)

Deployed 2026-09-21 from the fork branch `feat/tidegs-four-gpu`, commit `ad6bc18`
("Port the upstream evaluation pipeline"), which ports upstream
sponge-lab/TideGS `e4f44be`'s evaluation path.  These four files are NOT tracked
by git in this worktree, and `git worktree remove` would delete them.

    78ab0aa88a307241326a2fd48a0bdc4d59807ea3b25823d5631bf661febef627  tools/eval_protocol.py
    341c25d1217cb09399060ebc3a4fb22bdb686756c821f6491fd99fa08e3a1b02  tools/evaluate_pure_ssd_native.py
    66bbfa26846f3e895b57988dc2726c2bbca30ede59f371b7cd8ce37893c8bb4d  scripts/evaluate_missing_checkpoints.py
    a7aa67600bc231d34b076190bf061ce83aab3b0e875293e7945b311e1208813a  tests/test_eval_protocol.py

`tools/eval_protocol.py`, `scripts/evaluate_missing_checkpoints.py` and
`tests/test_eval_protocol.py` are byte-identical to upstream.

`tools/evaluate_pure_ssd_native.py` is upstream's logic adapted to this fork's
pre-refactor runtime, at four call sites: OffloadSceneDataset takes one argument
here, TideGaussianModel's base __init__ has no `args`, TideStorageAdapter takes
keyword arguments rather than a StorageConfig and rejects any execution_mode other
than "paper", and the camera schedule cache is disabled so evaluation cannot write
into a training run's cache directory.  It therefore runs against this checkout
(e327e1f) but NOT against an upstream e4f44be tree, whose runtime has the
refactored signatures.

The execution path has not been run end to end.  It was checked statically --
imports resolve, signatures match, the args attribute surface is present -- and the
protocol layer passes its own 18 tests, but no evaluation has been executed here.

The 200-view protocol is the point of the port: a fixed linspace subset of
transforms_test.json, identical for every checkpoint, with a sha256 over the
selected frames driving skip and resume.  It is also roughly 45x cheaper than the
fork's all-9066 sweep.

To use it, point the driver at a run directory; it evaluates every complete
checkpoint that has no finished evaluation yet:

    python scripts/evaluate_missing_checkpoints.py --run_dir <run_dir>
