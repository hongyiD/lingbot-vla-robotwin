import json
import re
from pathlib import Path

import pytest
import yaml

import check_compliance
import make_train_list
from helpers import make_lerobot, make_task

TASKS2 = ["click_bell", "lift_pot"]


def test_train_list_ok_and_summary(tmp_path):
    lr = tmp_path / "lerobot"
    for t in TASKS2:
        make_lerobot(lr, t)
    out = tmp_path / "lists" / "train.txt"
    assert make_train_list.main(["--lerobot_root", str(lr), "--tasks", ",".join(TASKS2), "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines == [f"robotwin_c50 {(lr / f'{t}-aloha-agilex_clean_50-50').resolve()}" for t in TASKS2]
    summary = json.loads((tmp_path / "lists" / "train.txt.summary.json").read_text())
    assert summary["total_frames"] == 2000


@pytest.mark.parametrize("kwargs,pattern", [
    ({"episodes": 49}, "{task}-aloha-agilex_clean_50-50"),
    ({}, "{task}-aloha-agilex_randomized_500-500"),
])
def test_train_list_rejects(tmp_path, kwargs, pattern):
    lr = tmp_path / "lerobot"
    for t in TASKS2:
        make_lerobot(lr, t, name_pattern=pattern, **kwargs)
    assert make_train_list.main(["--lerobot_root", str(lr), "--tasks", ",".join(TASKS2), "--name_pattern", pattern,
                                 "--out", str(tmp_path / "t.txt")]) == 1


def _setup(tmp_path, *, robot_norm="assets/norm_stats/robotwin_c50.json",
           train_norm="assets/norm_stats/robotwin_c50.json", model="/models/robbyant_lingbot-vla-v2-6b",
           task_strings=None, version="v2.1"):
    lingbot = tmp_path / "lingbot-vla-v2"
    (lingbot / "configs" / "robot_configs").mkdir(parents=True)
    (lingbot / "assets" / "norm_stats").mkdir(parents=True)
    (lingbot / "assets" / "norm_stats" / "robotwin.json").write_text('{"norm_stats": {"official": 1}}')
    (lingbot / "assets" / "norm_stats" / "robotwin_c50.json").write_text('{"norm_stats": {"clean": 1}}')
    (lingbot / "configs" / "robot_configs" / "robotwin_c50.yaml").write_text(yaml.safe_dump({"norm_stats": robot_norm}))
    raw, lr = tmp_path / "raw", tmp_path / "lerobot"
    for t in TASKS2:
        make_task(raw, t, 2)
        strings = (task_strings or {}).get(t, [f"{t} seen instruction 0", f"{t} seen instruction 2"])
        make_lerobot(lr, t, task_strings=strings, version=version,
                     name_pattern="{task}_joint_v30" if version == "v3.0" else "{task}-aloha-agilex_clean_50-50")
    pattern = "{task}_joint_v30" if version == "v3.0" else "{task}-aloha-agilex_clean_50-50"
    train_list = tmp_path / "train.txt"
    assert make_train_list.main(["--lerobot_root", str(lr), "--tasks", ",".join(TASKS2), "--name_pattern", pattern,
                                 "--out", str(train_list)]) == 0
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(yaml.safe_dump({
        "data": {"data_name": "multi", "train_path": str(train_list), "norm_stats_file": train_norm},
        "model": {"model_path": model, "config_path": model},
        "train": {"load_checkpoint_path": None}}))
    return ["--train_list", str(train_list), "--robot_config_root", str(lingbot / "configs" / "robot_configs"),
            "--train_config", str(train_cfg), "--raw_root", str(raw)]


def _statuses(tmp_path, argv):
    report = tmp_path / "compliance.md"
    rc = check_compliance.main(argv + ["--report", str(report)])
    return rc, dict(re.findall(r"^## (C\d) .*?: \*\*(\w+)\*\*$", report.read_text(), flags=re.M))


def test_compliance_pass(tmp_path):
    rc, st = _statuses(tmp_path, _setup(tmp_path))
    assert rc == 0, st
    assert st == {"C1": "WARN", "C2": "PASS", "C3": "PASS", "C4": "PASS", "C5": "SKIP"}


def test_compliance_pass_v30_parquet(tmp_path):
    rc, st = _statuses(tmp_path, _setup(tmp_path, version="v3.0"))
    assert rc == 0, st


def test_unseen_instruction_leak_fails_c2(tmp_path):
    argv = _setup(tmp_path, task_strings={"click_bell": ["click_bell unseen instruction 1"]})
    rc, st = _statuses(tmp_path, argv)
    assert rc == 1 and st["C2"] == "FAIL"


def test_generated_instruction_fails_c2(tmp_path):
    argv = _setup(tmp_path, task_strings={"lift_pot": ["Please lift the pot carefully (rephrased)"]})
    rc, st = _statuses(tmp_path, argv)
    assert rc == 1 and st["C2"] == "FAIL"


def test_official_norm_stats_fails_c3(tmp_path):
    argv = _setup(tmp_path, robot_norm="assets/norm_stats/robotwin.json", train_norm="assets/norm_stats/robotwin.json")
    rc, st = _statuses(tmp_path, argv)
    assert rc == 1 and st["C3"] == "FAIL"
    assert check_compliance.main(argv + ["--allow_official_norm"]) == 0


def test_train_inference_norm_mismatch_fails_c3(tmp_path):
    argv = _setup(tmp_path, train_norm="assets/norm_stats/robotwin.json")
    rc, st = _statuses(tmp_path, argv)
    assert rc == 1 and st["C3"] == "FAIL"


def test_unset_config_path_defaults_to_model_path_c4(tmp_path):
    argv = _setup(tmp_path)
    cfg_path = Path(argv[argv.index("--train_config") + 1])
    cfg = yaml.safe_load(cfg_path.read_text())
    del cfg["model"]["config_path"]
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc, st = _statuses(tmp_path, argv)
    assert rc == 0 and st["C4"] == "PASS"


def test_post_trained_model_fails_c4(tmp_path):
    argv = _setup(tmp_path, model="/models/robbyant_lingbot-vla-v2-6b-robotwin")
    rc, st = _statuses(tmp_path, argv)
    assert rc == 1 and st["C4"] == "FAIL"


def test_eval_script_checks_c5(tmp_path):
    argv = _setup(tmp_path)
    two_models = tmp_path / "eval_two.sh"
    two_models.write_text("python eval.py --model_path /ckpt/a --task_config demo_clean --instruction_type seen\n"
                          "python eval.py --model_path /ckpt/b --task_config demo_randomized --instruction_type unseen\n")
    rc, st = _statuses(tmp_path, argv + ["--eval_script", str(two_models)])
    assert rc == 1 and st["C5"] == "FAIL"
    wrong_pool = tmp_path / "eval_wrong.sh"
    wrong_pool.write_text("python eval.py --model_path /ckpt/a \\\n  --task_config demo_clean --instruction_type unseen\n")
    rc, st = _statuses(tmp_path, argv + ["--eval_script", str(wrong_pool)])
    assert rc == 1 and st["C5"] == "FAIL"
    good = tmp_path / "eval_good.sh"
    good.write_text("M=/ckpt/a\npython eval.py --model_path /ckpt/a --task_config demo_clean --instruction_type seen\n"
                    "python eval.py --model_path /ckpt/a --task_config demo_randomized --instruction_type unseen\n")
    rc, st = _statuses(tmp_path, argv + ["--eval_script", str(good)])
    assert rc == 0 and st["C5"] == "PASS", st
