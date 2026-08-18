import unittest

import pandas as pd

from src.csdemo.features import make_first_kill_samples


IDS = {
    "series_id": "series-1",
    "game_id": "lan:game-1",
    "round_id": "lan:game-1_1",
}


def round_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **IDS,
                "ct_win": 1,
                "map_name": "de_mirage",
                "ct_alive": 5,
                "t_alive": 5,
                "ct_eq_value": 20_000,
                "t_eq_value": 18_000,
            }
        ]
    )


class FirstKillFeatureTests(unittest.TestCase):
    def test_selects_earliest_tick_when_seconds_reset(self) -> None:
        kills = pd.DataFrame(
            [
                {
                    **IDS,
                    "time": 20.0,
                    "tick": 1_000,
                    "killer_side": "CT",
                    "victim_side": "T",
                    "weapon": "M4A4",
                    "headshot": 1,
                    "is_first_kill": 1,
                },
                {
                    **IDS,
                    "time": 5.0,
                    "tick": 2_000,
                    "killer_side": "T",
                    "victim_side": "CT",
                    "weapon": "AK-47",
                    "headshot": 0,
                    "is_first_kill": 0,
                },
            ]
        )

        sample = make_first_kill_samples(round_frame(), kills).iloc[0]

        self.assertEqual(sample["first_kill_weapon"], "M4A4")
        self.assertEqual(sample["first_kill_time"], 20.0)
        self.assertEqual(sample["first_kill_is_ct"], 1)

    def test_builds_post_kill_alive_state_from_the_5v5_snapshot(self) -> None:
        kills = pd.DataFrame(
            [
                {
                    **IDS,
                    "time": 15.0,
                    "tick": 1_500,
                    "killer_side": "T",
                    "victim_side": "CT",
                    "weapon": "AK-47",
                    "headshot": 0,
                    "is_first_kill": 1,
                }
            ]
        )

        sample = make_first_kill_samples(round_frame(), kills).iloc[0]

        self.assertEqual(sample["ct_alive_after_fk"], 4)
        self.assertEqual(sample["t_alive_after_fk"], 5)
        self.assertEqual(sample["alive_diff_ct_after_fk"], -1)
        self.assertEqual(sample["first_kill_advantage_ct"], -1)

    def test_ignores_non_enemy_kills_before_selecting_the_event(self) -> None:
        kills = pd.DataFrame(
            [
                {
                    **IDS,
                    "time": 10.0,
                    "tick": 1_000,
                    "killer_side": "CT",
                    "victim_side": "CT",
                    "weapon": "HE Grenade",
                    "headshot": 0,
                    "is_first_kill": 0,
                },
                {
                    **IDS,
                    "time": 11.0,
                    "tick": 1_100,
                    "killer_side": "CT",
                    "victim_side": "T",
                    "weapon": "M4A4",
                    "headshot": 0,
                    "is_first_kill": 0,
                },
            ]
        )

        sample = make_first_kill_samples(round_frame(), kills).iloc[0]

        self.assertEqual(sample["first_kill_weapon"], "M4A4")

    def test_requires_tick_for_reliable_event_ordering(self) -> None:
        kills = pd.DataFrame(
            [
                {
                    **IDS,
                    "time": 10.0,
                    "killer_side": "CT",
                    "victim_side": "T",
                }
            ]
        )

        with self.assertRaisesRegex(KeyError, "tick"):
            make_first_kill_samples(round_frame(), kills)


if __name__ == "__main__":
    unittest.main()
