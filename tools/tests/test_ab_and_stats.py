import math

import pytest

import ab_test
from lingbot_tools.io_utils import write_csv
from lingbot_tools.stats import bootstrap_mean_ci, mcnemar_exact, wilcoxon_gate
from lingbot_tools.tasks import TASKS


def test_mcnemar_exact_values():
    assert math.isclose(mcnemar_exact(10, 2), 2 * 79 / 4096, rel_tol=1e-12)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(2, 10) == mcnemar_exact(10, 2)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_wilcoxon_gate_semantics():
    assert wilcoxon_gate([0.5] * 5, [0.5] * 5).reason == "insufficient_nonzero_pairs"
    assert wilcoxon_gate([0.5] * 5, [0.5] * 4 + [0.9]).reason == "insufficient_nonzero_pairs"
    base = [0.1 * i for i in range(10)]
    assert wilcoxon_gate(base, [b - 0.05 for b in base]).reason == "no_mean_gain"
    d = wilcoxon_gate(base, [b + 0.2 + 0.01 * i for i, b in enumerate(base)])
    assert d.accepted and d.reason == "significant_gain" and d.n_pairs == 10
    with pytest.raises(ValueError):
        wilcoxon_gate([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        wilcoxon_gate([], [])


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=2000)
    assert lo < 3.0 < hi


def _rows(run_id, flips_by_setting, n_tasks=10, n_eps=20):
    rows = []
    for setting in ("clean", "randomized"):
        flip = flips_by_setting.get(setting, 0)
        for task in TASKS[:n_tasks]:
            for i in range(n_eps):
                seed = 100000 + i
                ok = int(i % 2 == 0)
                if flip > 0 and ok == 0 and i < 2 * flip:
                    ok = 1
                if flip < 0 and ok == 1 and i < -2 * flip:
                    ok = 0
                rows.append({"run_id": run_id, "setting": setting, "task": task, "episode_idx": i, "seed": seed,
                             "success": ok})
    return rows


def _run_ab(tmp_path, base_rows, cand_rows):
    b, c = tmp_path / "b" / "episodes.csv", tmp_path / "c" / "episodes.csv"
    fields = ["run_id", "setting", "task", "episode_idx", "seed", "success"]
    write_csv(b, base_rows, fields)
    write_csv(c, cand_rows, fields)
    pairs, info = ab_test.pair_episodes(ab_test.read_csv(b), ab_test.read_csv(c), "seed")
    assert info["paired"] == len(base_rows)
    res = ab_test.analyse(pairs, alpha_mcnemar=0.05, alpha_wilcoxon=0.10, max_setting_drop=1.0, n_boot=500)
    assert ab_test.main(["--baseline", str(b), "--candidate", str(c), "--n_boot", "200",
                         "--out_md", str(tmp_path / "ab.md"), "--out_json", str(tmp_path / "ab.json")]) == 0
    return res


def test_ab_adopt(tmp_path):
    res = _run_ab(tmp_path, _rows("base", {}), _rows("cand", {"clean": 2, "randomized": 2}))
    assert res["decision"] == "ADOPT"
    assert res["pooled"]["b_fail_to_success"] == 40 and res["pooled"]["c_success_to_fail"] == 0
    assert res["pooled"]["diff_pp"] == 10.0


def test_ab_reject_on_setting_drop(tmp_path):
    res = _run_ab(tmp_path, _rows("base", {}), _rows("cand", {"clean": 3, "randomized": -1}))
    assert res["pooled"]["diff_pp"] > 0
    assert res["decision"] == "REJECT"
    assert any("setting drop" in r for r in res["reasons"])


def test_ab_inconclusive_small_gain(tmp_path):
    base = _rows("base", {})
    cand = [dict(r) for r in base]
    flipped = 0
    for r in cand:
        if r["success"] == 0 and flipped < 2:
            r["success"] = 1
            flipped += 1
    res = _run_ab(tmp_path, base, cand)
    assert res["decision"] == "INCONCLUSIVE"
