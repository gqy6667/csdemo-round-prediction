import unittest

from src.csdemo.train_xgb import make_model


class M8TuningTests(unittest.TestCase):
    def test_pre_round_model_uses_controlled_tuning_winner(self) -> None:
        params = make_model(task="pre_round").get_params()

        self.assertEqual(params["n_estimators"], 3000)
        self.assertEqual(params["max_depth"], 2)
        self.assertEqual(params["min_child_weight"], 3)
        self.assertEqual(params["learning_rate"], 0.03)
        self.assertEqual(params["subsample"], 0.85)
        self.assertEqual(params["colsample_bytree"], 0.85)
        self.assertEqual(params["early_stopping_rounds"], 100)

    def test_first_kill_model_keeps_untuned_baseline(self) -> None:
        params = make_model(task="first_kill").get_params()

        self.assertEqual(params["n_estimators"], 500)
        self.assertEqual(params["max_depth"], 4)
        self.assertIsNone(params.get("early_stopping_rounds"))


if __name__ == "__main__":
    unittest.main()
