import unittest
import numpy as np
from kill_detector.src.utils import filter_timestamps, MIN_GAP_SECONDS

class TestPostProcessing(unittest.TestCase):
    """
    Tests for post-processing functions like timestamp gap filtering.
    """

    def test_gap_filtering_collapse(self):
        """
        Tests that timestamps with gaps smaller than MIN_GAP_SECONDS 
        are correctly collapsed while preserving ordering.
        """
        # Test case 1: Multiple small gaps
        timestamps1 = [1.0, 1.1, 1.2, 2.5, 2.6, 3.0]
        expected1 = [1.0, 2.5, 3.0]
        result1 = filter_timestamps(timestamps1, min_gap_seconds=0.5)
        self.assertEqual(result1, expected1)

        # Test case 2: No gaps filtered (all gaps large enough)
        timestamps2 = [1.0, 2.0, 3.0, 4.0]
        expected2 = [1.0, 2.0, 3.0, 4.0]
        result2 = filter_timestamps(timestamps2, min_gap_seconds=0.5)
        self.assertEqual(result2, expected2)

        # Test case 3: All gaps filtered (too small)
        timestamps3 = [1.0, 1.1, 1.2]
        expected3 = [1.0] # Only the first element remains
        result3 = filter_timestamps(timestamps3, min_gap_seconds=0.5)
        self.assertEqual(result3, expected3)

        # Test case 4: Empty list
        timestamps4 = []
        expected4 = []
        result4 = filter_timestamps(timestamps4, min_gap_seconds=0.5)
        self.assertEqual(result4, expected4)

        # Test case 5: Single element list
        timestamps5 = [5.0]
        expected5 = [5.0]
        result5 = filter_timestamps(timestamps5, min_gap_seconds=0.5)
        self.assertEqual(result5, expected5)

if __name__ == '__main__':
    unittest.main()
