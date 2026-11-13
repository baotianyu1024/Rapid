import unittest

from export_hugectr_acp import make_plan


class ACPPlanTest(unittest.TestCase):
    def test_paper_and_remote_thresholds_and_tie(self):
        profile = dict(ranks=8, batch_size=10, batches=4, per_table=[
            dict(table=0, pooling=100, unique_receive_mean=69),
            dict(table=1, pooling=100, unique_receive_mean=70),
            dict(table=2, pooling=100, unique_receive_mean=80),
        ])
        self.assertEqual([x['strategy'] for x in make_plan(profile, 'paper')['tables']],
                         ['Unique', 'Unique', 'Reduction'])
        self.assertEqual([x['strategy'] for x in make_plan(profile)['tables']],
                         ['Unique', 'Reduction', 'Reduction'])

    def test_rejects_duplicate_and_nonfinite_profile(self):
        profile = dict(ranks=8, batch_size=10, batches=1, per_table=[
            dict(table=0, pooling=1, unique_receive_mean=1),
            dict(table=0, pooling=1, unique_receive_mean=2),
        ])
        with self.assertRaises(ValueError):
            make_plan(profile)
        profile['per_table'] = [dict(table=0, pooling=1, unique_receive_mean=float('nan'))]
        with self.assertRaises(ValueError):
            make_plan(profile)


if __name__ == '__main__':
    unittest.main()
