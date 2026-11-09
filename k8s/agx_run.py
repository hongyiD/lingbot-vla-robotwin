#!/usr/bin/env python3
"""Run a LingBot script with a faster LeRobot chunk query (returned values unchanged).

Our LeRobot datasets keep frames as images inside the data files (image mode). LingBot's
LeRobotDataset._query_hf_dataset reads the 50 rows of an action chunk row-first, which with datasets 3.6 decodes all
three cameras of those rows (about 150 images per sample) just to keep the action column. Here non-image keys are
read row-first from a cached view of the same rows without the image columns; image keys use the original dataset.

  torchrun ... agx_run.py <script.py> [args...]   run the script as __main__ with the patch applied
  python agx_run.py --check <dataset dir>         compare items with and without the patch, and time both
"""
import json
import os
import random
import runpy
import sys
import time

import torch
import lingbotvla.data.vla_data.base_dataset as bd

_original_query = bd.LeRobotDataset._query_hf_dataset


def _non_image_view(ds):
    cached = ds.__dict__.get("_agx_view")
    if cached is None or cached[0] is not ds.hf_dataset:
        image_cols = [c for c in ds.meta.camera_keys if c in ds.hf_dataset.column_names]
        view = ds.hf_dataset.remove_columns(image_cols)
        view.set_transform(bd.hf_transform_to_torch)
        cached = ds._agx_view = (ds.hf_dataset, view)
    return cached[1]


def _fast_query(self, query_indices):
    result = {}
    for key, q_idx in query_indices.items():
        if key in self.meta.video_keys:
            continue
        rows = bd._to_relative_indices(self, q_idx)
        source = self.hf_dataset if key in self.meta.camera_keys else _non_image_view(self)
        result[key] = torch.stack(source[rows][key])
    return result


bd.LeRobotDataset._query_hf_dataset = _fast_query


def _difference(a, b):
    if a.keys() != b.keys():
        return f"keys differ: {sorted(set(a) ^ set(b))}"
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor):
            same = isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
        else:
            same = x == y
        if not same:
            return f"{k} differs"
    return None


def check(path):
    with open(os.path.join(path, "meta", "info.json")) as f:
        fps = json.load(f)["fps"]
    ds = bd.LeRobotDataset(repo_id=path, delta_timestamps={"action": [t / fps for t in range(50)]}, load_image=True)
    idx = [0, len(ds) - 1] + random.Random(0).sample(range(len(ds)), 6)

    def timed(query):
        bd.LeRobotDataset._query_hf_dataset = query
        start = time.perf_counter()
        items = [ds[i] for i in idx]
        return items, (time.perf_counter() - start) / len(idx)

    bd.LeRobotDataset._query_hf_dataset = _fast_query
    ds[idx[0]]  # builds the cached view outside the timing
    fast, t_fast = timed(_fast_query)
    orig, t_orig = timed(_original_query)
    bd.LeRobotDataset._query_hf_dataset = _fast_query
    diffs = [(i, d) for i, a, b in zip(idx, orig, fast) if (d := _difference(a, b))]
    print(f"{os.path.basename(path)}: {len(ds)} frames, image keys {ds.meta.camera_keys}, video keys {ds.meta.video_keys}")
    print(f"per sample: original {t_orig:.3f}s, patched {t_fast:.3f}s")
    print(f"global batch 256 over 4 ranks x 8 workers: >= {8 * t_orig:.1f}s/step original, >= {8 * t_fast:.1f}s/step patched")
    print("items identical" if not diffs else f"DIFFERENT: {diffs[:3]}")
    return 1 if diffs else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        sys.exit(check(sys.argv[2]))
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path[0] = os.path.dirname(os.path.abspath(script))
    runpy.run_path(script, run_name="__main__")
