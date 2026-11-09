"""Task list, settings and naming conventions for the RoboTwin 2.0 preliminary round."""
from __future__ import annotations

# Order matches templates/prelim_results_template.json and task_list_all in
# lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh.
TASKS: tuple[str, ...] = (
    "lift_pot", "hanging_mug", "stack_bowls_three", "scan_object", "handover_block",
    "click_bell", "put_object_cabinet", "open_microwave", "stack_blocks_three", "place_shoe",
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "dump_bin_bigbin", "grab_roller", "handover_mic", "move_can_pot",
    "move_pillbottle_pad", "move_playingcard_away", "place_cans_plasticbox",
    "place_container_plate", "place_dual_shoes", "place_empty_cup", "place_fan",
    "place_mouse_pad", "place_object_basket", "place_object_scale", "place_object_stand",
    "place_phone_stand", "move_stapler_pad", "open_laptop", "pick_diverse_bottles",
    "pick_dual_bottles", "place_a2b_left", "place_a2b_right", "place_bread_basket",
    "place_bread_skillet", "place_burger_fries", "place_can_basket", "press_stapler",
    "rotate_qrcode", "shake_bottle_horizontally", "shake_bottle", "stack_blocks_two",
    "stack_bowls_two", "stamp_seal", "turn_switch", "put_bottles_dustbin",
)

SETTINGS: tuple[str, ...] = ("clean", "randomized")

TASK_CONFIG_TO_SETTING = {
    "demo_clean": "clean",
    "demo_randomized": "randomized",
}

# Expected instruction pool per setting (competition definition: clean2clean uses seen,
# clean2random uses unseen).
EXPECTED_INSTRUCTION_TYPE = {"clean": "seen", "randomized": "unseen"}

CLEAN_ARCHIVE = "aloha-agilex_clean_50.zip"
CLEAN_SETTING_DIR = "aloha-agilex_clean_50"
EPISODES_PER_TASK = 50

# Substrings that must never appear in a training dataset path.
FORBIDDEN_DATA_TOKENS: tuple[str, ...] = ("randomized", "piper", "arx", "franka", "ur5")

OFFICIAL_MIXED_NORM_STATS = "assets/norm_stats/robotwin.json"


def resolve_tasks(spec: str | None) -> list[str]:
    """Parse a task spec: None/"all" -> all 50 tasks, else a comma-separated subset."""
    if spec is None or spec.strip().lower() in ("", "all"):
        return list(TASKS)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(unknown)}")
    return names


def setting_from_task_config(task_config: str) -> str | None:
    return TASK_CONFIG_TO_SETTING.get(task_config)
