import unittest

import numpy as np

from fdth import fdt

class Test(unittest.TestCase):
    """Statistical function tests"""

    def assertClose(self, value: float, expected: float, atol: float = 1e-08) -> None:
        assert np.isclose(value, expected, atol=atol), f"expected around {expected}, got {value}"

    def test_mean_fdt(self):
        data = np.array([1, 5, 3, 2, 1, 8])
        fd = fdt(data)
        self.assertClose(data.mean(), fd.mean(), atol=0.5)

    def test_var_fdt(self):
        data = np.array([1, 5, 3, 2, 1, 8])
        fd = fdt(data)
        self.assertClose(data.var(), fd.var(), atol=1.5)

    def test_sd_fdt(self):
        data = np.array([1, 5, 3, 2, 1, 8])
        fd = fdt(data)
        self.assertClose(data.std(), fd.sd(), atol=1.5)

    def test_median(self):
        data = np.array([1, 5, 3, 2, 1, 8])
        fd = fdt(data)
        self.assertClose(np.median(data), fd.median(), atol=0.4)

    def test_mfv(self):
        def mfv_and_compare(data, expected_mfv) -> None:
            calculated_mfv = fdt(data).mfv()
            assert all(calculated_mfv == expected_mfv)

        mfv_and_compare([1, 2, 2, 3, 4], [2])
        mfv_and_compare([1, 1, 2, 2, 3], [1, 2])
        mfv_and_compare([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        mfv_and_compare([2, 2, 2, 2, 2], [2])
        mfv_and_compare(["a", "b", "b", "c"], ["b"])
