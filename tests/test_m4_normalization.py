from pathlib import Path
import unittest

from src.csdemo.esta_to_tables import demo_rows, side_features, weapon_bucket


def player(inventory: list[dict]) -> dict:
    return {"steamID": 1, "isAlive": True, "inventory": inventory}


class M4NormalizationTests(unittest.TestCase):
    def test_esta_m4a1_name_is_counted_as_m4a1_s_and_rifle(self) -> None:
        inventory = [{"weaponName": "M4A1", "weaponClass": "Rifle"}]

        features = weapon_bucket(inventory)

        self.assertEqual(features["m4a1_s"], 1)
        self.assertEqual(features["rifles"], 1)

    def test_one_player_contributes_at_most_one_rifle(self) -> None:
        inventory = [
            {"weaponName": "AK-47", "weaponClass": "Rifle"},
            {"weaponName": "AK-47", "weaponClass": "Rifle"},
            {"weaponName": "Galil AR", "weaponClass": "Rifle"},
        ]

        features = weapon_bucket(inventory)

        self.assertEqual(features["ak47"], 1)
        self.assertEqual(features["rifles"], 1)

    def test_player_utility_is_recomputed_and_capped_at_four(self) -> None:
        five_grenades = [
            {"weaponName": "Smoke Grenade", "weaponClass": "Grenade"},
            {"weaponName": "Smoke Grenade", "weaponClass": "Grenade"},
            {"weaponName": "Flashbang", "weaponClass": "Grenade"},
            {"weaponName": "HE Grenade", "weaponClass": "Grenade"},
            {"weaponName": "Molotov", "weaponClass": "Grenade"},
        ]
        frame = {
            "t": {
                "alivePlayers": 5,
                "totalUtility": 21,
                "players": [player(five_grenades)],
            }
        }

        features = side_features(frame, "t")

        self.assertEqual(features["t_grenades"], 4)

    def test_non_five_v_five_snapshot_and_its_kills_are_excluded(self) -> None:
        demo = {
            "matchId": "series-1",
            "mapName": "de_vertigo",
            "gameRounds": [
                {
                    "roundNum": 5,
                    "freezeTimeEndTick": 100,
                    "winningSide": "CT",
                    "frames": [
                        {
                            "tick": 100,
                            "ct": {"alivePlayers": 1, "players": []},
                            "t": {"alivePlayers": 0, "players": []},
                        }
                    ],
                    "kills": [
                        {
                            "seconds": 20.0,
                            "attackerSide": "CT",
                            "victimSide": "T",
                        }
                    ],
                }
            ],
        }

        rounds, kills = demo_rows(demo, Path("lan/game-a.json.xz"))

        self.assertEqual(rounds, [])
        self.assertEqual(kills, [])


if __name__ == "__main__":
    unittest.main()
