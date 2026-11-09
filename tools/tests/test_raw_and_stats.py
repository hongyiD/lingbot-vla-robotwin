import json

import check_raw_data
import data_stats
from helpers import SETTING, make_task
from lingbot_tools.io_utils import read_csv


def test_clean_data_passes(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    make_task(raw, "click_bell", 5)
    make_task(raw, "lift_pot", 5)
    rc = check_raw_data.main(["--raw_root", str(raw), "--tasks", "click_bell,lift_pot", "--expect_episodes", "5",
                              "--jobs", "1", "--out_dir", str(out)])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["errors"] == 0
    assert summary["episodes"] == 10
    rows = {r["task"]: r for r in read_csv(out / "summary.csv")}
    assert rows["click_bell"]["image_hw"] == "240x320"
    assert rows["click_bell"]["seeds"] == "5"
    assert rows["lift_pot"]["T_min"] == "60" and rows["lift_pot"]["T_max"] == "64"


def test_faults_are_reported(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    sd = make_task(raw, "click_bell", 6, overrides={
        1: {"nan": True}, 2: {"cam_len_delta": -1}, 3: {"bad_jpeg": True}, 4: {"static": True},
        5: {"drop_key": "joint_action/right_gripper"}})
    (sd / "instructions" / "episode0.json").unlink()
    rc = check_raw_data.main(["--raw_root", str(raw), "--tasks", "click_bell", "--expect_episodes", "7",
                              "--jobs", "1", "--out_dir", str(out)])
    assert rc == 1
    codes = {(i["episode"], i["code"]) for i in read_csv(out / "issues.csv")}
    for expected in [("1", "non_finite_joint"), ("2", "camera_length_mismatch"), ("3", "bad_jpeg"),
                     ("4", "static_episode"), ("5", "missing_keys"), ("0", "missing_instruction_file"),
                     ("", "episode_count"), ("", "missing_episode_indices")]:
        assert expected in codes, expected


def test_print_tree(tmp_path, capsys):
    sd = make_task(tmp_path / "raw", "click_bell", 1)
    assert check_raw_data.main(["--print_tree", str(sd / "data" / "episode0.hdf5")]) == 0
    assert "joint_action/left_arm  shape=(60, 6)" in capsys.readouterr().out


def test_data_stats_flags_only_the_jerky_episode(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    make_task(raw, "click_bell", 15, overrides={7: {"noise": 0.02}})
    rc = data_stats.main(["--raw_root", str(raw), "--tasks", "click_bell", "--jobs", "1", "--out_dir", str(out)])
    assert rc == 0
    flagged = read_csv(out / "flagged.csv")
    assert [r["episode"] for r in flagged] == ["7"]
    assert "jerk_rms" in flagged[0]["flag_reasons"]
    stats = read_csv(out / "episode_stats.csv")
    assert len(stats) == 15
    assert all(r["grip_toggles_l"] == "2" for r in stats)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["lerobot_frames_est"] == sum(60 + i - 1 for i in range(15))
    quant = json.loads((out / "joint_quantiles.json").read_text())
    assert len(quant["q99"]) == 14
    assert (raw / "click_bell" / SETTING / "data" / "episode7.hdf5").exists()  # never deletes
