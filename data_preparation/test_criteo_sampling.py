"""Verify max_rows samples the whole source before the default shuffle."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from criteo_singlehot_to_multihot import (MULTI_HOT_SIZES, RECORD_DTYPE, convert,
                                         select_source_rows)


class CriteoSamplingTest(unittest.TestCase):
    def test_full_source_sampling_and_reproducible_shuffle(self):
        with tempfile.TemporaryDirectory(prefix='rapid_criteo_sampling_', dir='/tmp') as tmp:
            root = Path(tmp)
            source = root / 'source.tsv'
            output = root / 'sample.bin'
            if shutil.disk_usage(root).free < 1 << 20:
                self.skipTest('Insufficient space for synthetic input')
            with source.open('w') as stream:
                for row in range(100):
                    stream.write('\t'.join([str(row), *(['1'] * 13), *(['0'] * 26)]) + '\n')
            tables = [np.zeros((1, size), dtype=np.int32) for size in MULTI_HOT_SIZES]
            with patch('criteo_singlehot_to_multihot.build_multi_hot_tables', return_value=tables):
                convert(str(source), str(output), max_rows=10, chunk_size=9)
            selected = select_source_rows(100, 10, 42)
            expected = selected[np.random.RandomState(42).permutation(10)]
            actual = np.fromfile(output, dtype=RECORD_DTYPE)['label']
            np.testing.assert_array_equal(actual, expected)
            self.assertGreater(int(actual.max()), 50)


if __name__ == '__main__':
    unittest.main()
