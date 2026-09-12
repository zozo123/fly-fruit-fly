import copy
import unittest

from fly_fruit_fly.assessment import assess


def report(n=10):
    return {"episodes": [{"seed": i, "duration_s": 0.599,
                          "return": 200., "mean_tracking_error_cm": 0.05,
                          "completed_reference": True, "failed": False}
                         for i in range(n)]}


class AssessmentTests(unittest.TestCase):
    def test_pass(self):
        self.assertTrue(assess(report())["task_gate_passed"])

    def test_small_sample_does_not_pass(self):
        self.assertFalse(assess(report(3))["task_gate_passed"])

    def test_nine_of_ten_boundary(self):
        r = report()
        r["episodes"][0]["mean_tracking_error_cm"] = 0.11
        self.assertTrue(assess(r)["task_gate_passed"])
        r["episodes"][1]["mean_tracking_error_cm"] = 0.11
        self.assertFalse(assess(r)["task_gate_passed"])

    def test_high_reward_crashes_do_not_pass(self):
        r = report()
        for e in r["episodes"]:
            e.update(failed=True, completed_reference=False, return_value=1e9)
            e["return"] = 1e9
        self.assertFalse(assess(r)["task_gate_passed"])

    def test_short_flight_does_not_pass(self):
        r = report()
        for e in r["episodes"]:
            e["duration_s"] = 0.06
        self.assertFalse(assess(r)["task_gate_passed"])

    def test_nonfinite_and_empty_rejected(self):
        for value in (float("nan"), float("inf"), "nan"):
            r = report()
            r["episodes"][0]["return"] = value
            with self.assertRaises(ValueError):
                assess(r)
        with self.assertRaises(ValueError):
            assess({"episodes": []})

    def test_duplicate_seeds_rejected(self):
        r = report()
        r["episodes"][1]["seed"] = 0
        with self.assertRaises(ValueError):
            assess(r)

    def test_paired_comparison(self):
        r = report()
        baseline = copy.deepcopy(r)
        baseline["episodes"].reverse()
        self.assertEqual(assess(r, baseline)["paired_mean_delta"]["return"], 0)

    def test_mismatched_seeds_rejected(self):
        with self.assertRaises(ValueError):
            assess(report(), report(9))

    def test_mismatched_tasks_rejected(self):
        r = report()
        r["task"] = {"speed": 99}
        with self.assertRaises(ValueError):
            assess(r, report())

    def test_mismatched_controller_config_rejected(self):
        candidate = report()
        baseline = report()
        candidate.update(control_mode="wings", action_repeat=10)
        baseline.update(control_mode="wings", action_repeat=1)
        with self.assertRaisesRegex(ValueError, "controller configurations"):
            assess(candidate, baseline)

    def test_legacy_controller_config_is_full_rate(self):
        result = assess(report())
        self.assertEqual(result["controller_config"],
                         {"control_mode": "full", "action_repeat": 1})

    def test_inconsistent_flags_rejected(self):
        r = report()
        r["episodes"][0]["failed"] = True
        with self.assertRaises(ValueError):
            assess(r)


if __name__ == "__main__":
    unittest.main()
