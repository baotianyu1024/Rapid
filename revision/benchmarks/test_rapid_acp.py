import json
import tempfile
import unittest
from pathlib import Path

from rapid_acp import read_acp_plan


class ACPReaderTest(unittest.TestCase):
    def test_mixed_plan_and_layout_validation(self):
        plan = dict(schema='rapid-hugectr-acp-v1', ranks=2, local_batch=4,
                    table_lengths=[1, 3], tables=[
                        dict(table=0, pooling=1, strategy='Unique'),
                        dict(table=1, pooling=3, strategy='Reduction')])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'acp.json'
            path.write_text(json.dumps(plan))
            self.assertEqual(read_acp_plan(path, 2, 8, [1, 3]), [0])
            with self.assertRaises(ValueError):
                read_acp_plan(path, 2, 16, [1, 3])
            plan['tables'][1]['pooling'] = 1
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                read_acp_plan(path, 2, 8, [1, 3])


if __name__ == '__main__':
    unittest.main()
