PRE_ROUND_FEATURE_GROUPS = {
    "context": [
        "map_name",
        "round_num",
    ],
    "score": [
        "ct_score",
        "t_score",
        "score_diff_ct",
    ],
    "economy": [
        "ct_eq_value",
        "t_eq_value",
        "eq_value_diff_ct",
        "ct_cash",
        "t_cash",
        "cash_diff_ct",
    ],
    "armor_utility": [
        "ct_armor",
        "t_armor",
        "armor_diff_ct",
        "ct_helmets",
        "t_helmets",
        "helmet_diff_ct",
        "ct_defuse_kits",
        "ct_grenades",
        "t_grenades",
        "grenade_diff_ct",
    ],
    "weapons": [
        "ct_ak47",
        "t_ak47",
        "ct_m4a4",
        "t_m4a4",
        "ct_m4a1_s",
        "t_m4a1_s",
        "ct_awp",
        "t_awp",
        "awp_diff_ct",
        "ct_rifles",
        "t_rifles",
        "rifle_diff_ct",
        "ct_smgs",
        "t_smgs",
        "smg_diff_ct",
    ],
}


PRE_ROUND_FEATURES = [
    feature
    for group_features in PRE_ROUND_FEATURE_GROUPS.values()
    for feature in group_features
]


# These are valid extraction/quality-control fields, but the M4.2 contract requires
# every accepted pre-combat snapshot to be 5v5, so they are constant model inputs.
PRE_ROUND_EXCLUDED_CONSTANTS = [
    "ct_alive",
    "t_alive",
    "alive_diff_ct",
]


LEGACY_PRE_ROUND_FEATURES = [
    "map_name",
    "round_num",
    "ct_score",
    "t_score",
    "score_diff_ct",
    "ct_alive",
    "t_alive",
    "alive_diff_ct",
    "ct_eq_value",
    "t_eq_value",
    "eq_value_diff_ct",
    "ct_cash",
    "t_cash",
    "cash_diff_ct",
    "ct_armor",
    "t_armor",
    "armor_diff_ct",
    "ct_helmets",
    "t_helmets",
    "helmet_diff_ct",
    "ct_defuse_kits",
    "ct_ak47",
    "t_ak47",
    "ct_m4a4",
    "t_m4a4",
    "ct_m4a1_s",
    "t_m4a1_s",
    "ct_awp",
    "t_awp",
    "awp_diff_ct",
    "ct_rifles",
    "t_rifles",
    "rifle_diff_ct",
    "ct_smgs",
    "t_smgs",
    "smg_diff_ct",
    "ct_grenades",
    "t_grenades",
    "grenade_diff_ct",
]

FIRST_KILL_EXTRA_FEATURES = [
    "first_kill_time",
    "first_kill_is_ct",
    "first_death_is_ct",
    "first_kill_headshot",
    "first_kill_weapon",
    "ct_alive_after_fk",
    "t_alive_after_fk",
    "alive_diff_ct_after_fk",
    "first_kill_advantage_ct",
]

FIRST_KILL_FEATURES = PRE_ROUND_FEATURES + FIRST_KILL_EXTRA_FEATURES

ID_COLUMNS = [
    "series_id",
    "game_id",
    "round_id",
]
