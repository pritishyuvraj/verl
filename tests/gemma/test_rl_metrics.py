import unittest
from verl.extensions.gemma.rl_metrics import group_metrics, require_complete_validation


class EpisodeMetricTest(unittest.TestCase):
    def test_missing_or_duplicate_validation_refused(self):
        expected = {("dev", "a"):1,("dev", "b"):1}
        require_complete_validation(["dev","dev"],["a","b"],expected)
        for sources, uids in ((["dev"],["a"]),(["dev","dev"],["a","a"]),([],[])):
            with self.assertRaises(RuntimeError):
                require_complete_validation(sources,uids,expected)

    def test_variable_turns_do_not_reweight_rewards(self):
        keys, scores = [], []
        for group in range(3):
            for episode in range(8):
                reward = int(group == 0 or group == 2 and episode < 4)
                for turn in range(episode+1):
                    keys.append(f"issue_{group}_{episode}_{turn}")
                    scores.append(reward)
        m = group_metrics(keys, scores, n=8)
        self.assertEqual(m["reward/mean"], .5)
        self.assertEqual(m["groups/all_pass_count"], 1)
        self.assertEqual(m["groups/all_zero_count"], 1)
        self.assertEqual(m["groups/mixed_count"], 1)

    def test_incomplete_group_refused(self):
        with self.assertRaises(ValueError):
            group_metrics(["u_0_0"], [1], n=8)

    def test_nonbinary_reward_refused(self):
        with self.assertRaises(ValueError):
            group_metrics(["u_0_0"], [.5], n=1)
