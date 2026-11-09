# 灵波初赛工具集（embodiedx/tools）

这些是独立 Python 脚本，不依赖 torch 和 AgenticX，在 Mac 和 GPU 机上都能跑（例外：`gpu_env_probe.py` 的全量模式要在 LingBot 训练环境里运行）。只做校验、统计、解析、判定和记账，**不训练、不改数据、不自动提交**。

## 环境

```bash
cd tools
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q        # 53 个合成数据用例
```

GPU 机上同理（只需 numpy/scipy/h5py/pyyaml/pyarrow/pillow）；也可以直接用训练环境的 Python 跑这些脚本，它们没有额外依赖。

## 工具一览

| 阶段 | 脚本 | 作用 |
|---|---|---|
| 上机 | `gpu_env_probe.py` | 租来的 GPU 机（B300 优先）的环境验证，单文件可直接拷过去用。`--platform_only` 只用标准库，检查系统、驱动、NVSwitch、拓扑、资源、编译器和网络；全量模式在训练环境里运行，把 LingBot 依赖的每条 GPU 路径与 fp32/fp64 参考结果比对；`--nccl` 用 torchrun 测集合通信；`--triton_target X.Y` 在没有该型号 GPU 的机器上为它交叉编译 Triton 内核（B300 填 10.3），训练镜像构建时用它自检。 |
| 下载 | `hf_manifest.py` | 从 HF API 生成固定 revision 的文件清单（大小 + sha256），只读元数据；`--repo_type model` 加 `--exclude` 可为模型权重生成清单 |
| 下载 | `verify_downloads.py` | 按清单校验本地文件（数据 zip 或模型权重目录）的大小和 sha256，失败时打印重下命令 |
| 下载 | `extract_archives.py` | 安全解压到 `<raw_root>/<task>/aloha-agilex_clean_50/`（防路径穿越、校验顶层目录、幂等） |
| 清洗 | `check_raw_data.py` | 结构完整性检查：episode 数、HDF5 键/长度/NaN/范围/静止、JPEG 解码、指令池、seed。`--print_tree` 可查看陌生 HDF5 的结构 |
| 清洗 | `data_stats.py` | 动作质量统计加任务内 Z-score，只标记不删除，输出带视频路径的复核清单与 14 维关节分位数 |
| 训练前 | `make_train_list.py` | 生成 `robotwin_c50 <abs_path>` 训练清单，校验 episode 数和禁用词，输出总帧数和每 epoch 步数 |
| 训练前 | `check_compliance.py` | 合规检查 C1–C5：数据范围、指令只用 seen、clean-only 归一化且训练与推理一致、底模、评测脚本 |
| 评测后 | `parse_eval_run.py` | 把 LingBot 启动器或 AMD benchmark 的评测日志解析成逐 episode CSV（只取最后一次重试），并核对指令池 |
| 评测后 | `make_results_json.py` | 按官方模板生成提交 JSON（键序一致；指令池不符或缺任务时拒绝生成） |
| 决策 | `ab_test.py` | 按 seed 配对做 McNemar 精确检验，加任务层 Wilcoxon 门控（与 AgenticX `rl/acceptance.py` 语义一致）和 bootstrap CI，按采纳规则输出 ADOPT / REJECT / INCONCLUSIVE |
| 记账 | `ledger.py` | 追加式实验台账 `ledger/ledger.jsonl`，自动渲染 `ledger.md`，可记录文件 sha256 |
| 汇报 | `daily_report.py` | 从台账生成日报，可选推送到钉钉或飞书自定义机器人（支持加签） |

## 典型流程

1. 数据：`hf_manifest` → `hf download` → `verify_downloads` → `extract_archives` → `check_raw_data` → `data_stats` → `ledger add --kind data`
2. 训练前：pi0 链路产出的 LeRobot v2.1 先转 v3.0（`python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id <name> --root <dir> --push-to-hub false`；训练环境的 lerobot 0.4.2 只认 v3.0）→ `make_train_list` → 重算 clean-only 归一化统计（LingBot `scripts/compute_norm_stats.py`）→ `check_compliance`
3. 评测后：`parse_eval_run` → `ab_test`（需要比较时）→ `make_results_json` → 人工上传 → `ledger add --kind submit`


## 训练镜像与 Kubernetes 部署（同级目录）

- `../docker/lingbotvla/`：训练镜像的 `Dockerfile`、`build_image.sh`（组装只含 tools、清单、k8s 的构建上下文）、`check_image.sh`（构建最后一步的自检）。本目录会打进镜像的 `/opt/embodiedx/tools/`。
- `../k8s/`：Kubernetes Deployment 模板、Pod 环境文件 `k8s.env`、配置生成脚本 `make_train_configs.py`、数据读取补丁 `agx_run.py`、训练指标汇总 `train_metrics.py`、渲染探针 `rt_probe.py` 和 OIDN 替换脚本 `oidn_swap.sh`。

## 约定

- 50 个任务的顺序与官方 JSON 模板（`../templates/prelim_results_template.json`）一致（`lingbot_tools/tasks.py`）。
- 指令池：clean 对应 seen，randomized 对应 unseen。clean 用了 unseen 评测时，`parse_eval_run` 记为 ERROR，`make_results_json` 拒绝生成。
- 所有脚本出错时退出码非 0，可以直接串进 shell、cron 或 `agx loop`。
