# LingBot-VLA 2.0 on RoboTwin 2.0: full-parameter fine-tuning on 4× NVIDIA B300

Data, training and evaluation tooling for fully fine-tuning the 6B [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) on the [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) `aloha-agilex_clean_50` demonstrations, built for the simulation track (preliminary round) of the 2026 Ant Lingbo embodied foundation model challenge.

## Results

| | |
|---|---|
| Data | 50 tasks, 2,500 episodes; 552,287 raw frames, 549,787 frames after conversion to LeRobot v3.0; 0 structural errors |
| Training | `lingbot-vla-v2-6b`, full-parameter, 4× B300 (SM 10.3), global batch 256, 10,735 steps (5 epochs), cosine LR 1e-4 → 5e-5; no non-finite step |
| Loss | VLA loss 0.139 → 0.013 (mean of first vs. last 100 steps, −90.5%) |
| Evaluation | 85% success in RoboTwin 2.0 simulation |
| Data loading | per-sample read time 2.53 s → 0.048 s; the checked samples are identical tensor for tensor |

## Pipeline

1. **Download and checks** (`tools/`). Fixed-revision manifests with sizes and SHA-256 (`hf_manifest.py`, `verify_downloads.py`), path-safe extraction, structural checks of every HDF5 episode (`check_raw_data.py`) and action-quality statistics that flag episodes for review without dropping any (`data_stats.py`).
2. **Deterministic conversion** (`patches/`). RoboTwin's pi0 converter picks each episode's language instruction with an unseeded random choice. The patch makes the pick a hash of (seed, dataset, episode file), so converting the two pilot tasks on two different machines produced identical instruction CSVs. The LeRobot v2.1 output is then converted to v3.0 and read back.
3. **Normalization and compliance.** Normalization statistics are recomputed on the clean data only, and the robot config points to that one file, so training and inference use the same statistics. `check_compliance.py` checks data scope, seen-only instructions, normalization consistency, base model and evaluation script.
4. **Training image** (`docker/`). CUDA 12.9 / PyTorch 2.8 / Triton 3.4 / flash-attn 2.8.3 for SM 10.3. The build ends with a self-check that includes cross-compiling Triton kernels for SM 10.3 on a machine without that GPU.
5. **Bring-up on B300** (`tools/gpu_env_probe.py`). Compares every GPU code path LingBot uses (matmul, attention, fused MoE, compile, video decode) against fp32/fp64 references and tests NCCL collectives (all-reduce bus bandwidth 642 GB/s at 1 GiB across 4 GPUs).
6. **Training** (`k8s/`). `make_train_configs.py` writes smoke and formal configs from the official `robotwin.yaml`, changing only paths, batch split and step counts. Smoke runs covered the compiled vs. reference GPU path, checkpoint resume and two batch splits. GPUs were available only overnight, so each run stopped automatically right after a completed checkpoint and was later resumed from it; `train_metrics.py` merges the logs of resumed runs.
7. **Data-loading fix** (`k8s/agx_run.py`). With frames stored as images inside the LeRobot data files, reading one 50-step action chunk decoded all three cameras of those rows, about 150 images per sample. Reading the non-image columns from a cached view without the image columns removed the data wait (4 GPUs × 8 workers: lower bound 20.2 s → 0.4 s per step).
8. **Simulator on Blackwell** (`k8s/rt_probe.py`, `k8s/oidn_swap.sh`). Headless Vulkan in the container needed the EGL ICD instead of the GLX one; SAPIEN's bundled OIDN 2.0.1 has no Blackwell kernels, so ray-traced frames came out undenoised while the render probe still passed. `oidn_swap.sh` swaps in OIDN 2.3.3, which adds them.
9. **Evaluation and decisions.** `parse_eval_run.py` turns evaluation logs into per-episode CSVs; `ab_test.py` compares two policies with a paired McNemar exact test, a task-level Wilcoxon gate and bootstrap CIs.
10. **Ledger** (`ledger/`). Append-only experiment log (28 entries) with configs, hashes and measurements for every step above.

## Layout

| Path | Contents |
|---|---|
| `tools/` | Data, compliance, evaluation, statistics and ledger scripts, with 53 tests on synthetic data |
| `patches/` | Deterministic-instruction patch for RoboTwin 13c3c47's pi0 converter |
| `docker/lingbotvla/` | Training image for B300 and its build-time self-check |
| `k8s/` | Kubernetes Deployments, Pod environment (`k8s.env`), config generator, data-loading patch, metrics, render probe |
| `data/manifests/` | Fixed-revision file manifests for the dataset and the three model weights |
| `templates/` | Result JSON template of the challenge |
| `ledger/` | Experiment ledger (`ledger.jsonl` is the source, `ledger.md` is rendered from it) |

## Tests

```bash
cd tools
pip install -r requirements.txt
python -m pytest -q
```

## Notes

- Upstream code (LingBot-VLA 2.0 at `be969b8`, RoboTwin at `13c3c47`, LeRobot), datasets, weights, checkpoints and logs are not included.
- Cluster-specific values are placeholders: `<REGISTRY>`, `<PROJECT>`, `<NAMESPACE>`, `<PROXY_HOST>`, `<PROXY_PORT>`, `<CLUSTER_CIDRS>`. Some comments refer to sections of internal runbooks that are not published, and hashes in the ledger refer to the files before these values were replaced.
- Code comments, the ledger and `tools/README.md` are partly in Chinese.
