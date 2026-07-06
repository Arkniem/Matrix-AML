#!/usr/bin/env python3
"""Probe the GPU node env for the NN experiment: torch + CUDA + GPU name, and that the data pipeline
loads (holdout active, composition/RNA shapes). Writes runs/gpu_probe.json (direct, pollable)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "gpu_probe.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
res = {}
try:
    import torch
    res["torch"] = torch.__version__
    res["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        res["gpu"] = torch.cuda.get_device_name(0)
        res["n_gpu"] = int(torch.cuda.device_count())
        res["cuda"] = torch.version.cuda
except Exception as e:
    res["torch_error"] = "%s: %s" % (type(e).__name__, e)
try:
    from amlmm.context import build_context, Config
    from amlmm import discovery as D
    ctx = build_context(Config(run_id="gpu_probe"))
    res["holdout"] = len(ctx.holdout)
    res["n_samples"] = int(ctx.tables["samples"].shape[0])
    comp = D._sample_level_matrix(ctx, "composition", set(ctx.tables["samples"].index))
    res["composition_shape"] = list(comp.shape)
except Exception as e:
    res["data_error"] = "%s: %s" % (type(e).__name__, e)
with open(out, "w") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps(res, indent=2, default=str), flush=True)
