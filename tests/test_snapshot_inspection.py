import unittest

from src.csdemo.inspect_snapshot import inspect_round_data


class SnapshotInspectionTests(unittest.TestCase):
    def test_inspection_uses_nearest_freeze_end_frame_and_lists_inventory(self) -> None:
        round_data = {
            "roundNum": 20,
            "freezeTimeEndTick": 100,
            "frames": [
                {
                    "tick": 95,
                    "t": {
                        "alivePlayers": 5,
                        "totalUtility": 21,
                        "players": [
                            {
                                "steamID": 1,
                                "name": "player-one",
                                "isAlive": True,
                                "inventory": [
                                    {"weaponName": "AK-47", "weaponClass": "Rifle"},
                                    {"weaponName": "Galil AR", "weaponClass": "Rifle"},
                                    {"weaponName": "Flashbang", "weaponClass": "Grenade"},
                                ],
                            }
                        ],
                    },
                    "ct": {"alivePlayers": 5, "totalUtility": 12, "players": []},
                },
                {"tick": 140, "t": {}, "ct": {}},
            ],
        }

        report = inspect_round_data(round_data)

        self.assertEqual(report["frame_tick"], 95)
        self.assertEqual(report["freeze_time_end_tick"], 100)
        self.assertEqual(report["frame_tick_offset"], -5)
        self.assertEqual(report["t"]["reported_total_utility"], 21)
        self.assertEqual(report["t"]["derived_rifle_count"], 2)
        self.assertEqual(report["t"]["derived_utility_count"], 1)
        self.assertEqual(report["t"]["players"][0]["name"], "player-one")
        self.assertEqual(report["t"]["players"][0]["rifle_count"], 2)
        self.assertEqual(
            report["t"]["players"][0]["inventory"],
            [
                {"weapon_name": "AK-47", "weapon_class": "Rifle"},
                {"weapon_name": "Galil AR", "weapon_class": "Rifle"},
                {"weapon_name": "Flashbang", "weapon_class": "Grenade"},
            ],
        )

    def test_missing_round_number_raises_clear_error(self) -> None:
        demo = {"gameRounds": [{"roundNum": 1}]}

        with self.assertRaisesRegex(ValueError, "Round 2 was not found"):
            inspect_round_data(demo, round_num=2)


if __name__ == "__main__":
    unittest.main()
