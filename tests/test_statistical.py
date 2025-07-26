import unittest

import numpy as np

from fdth import fdt, Binning


class Test(unittest.TestCase):
    """Statistical function tests"""

    def assertClose(self, value: float, expected: float, atol: float = 1e-08) -> None:
        assert np.isclose(
            value, expected, atol=atol
        ), f"expected around {expected}, got {value}"

    def test_mean_fdt(self):
        data = np.array([1, 5, 3, 2, 1, 8])
        fd = fdt(data)
        self.assertClose(fd.mean(), data.mean(), atol=0.5)

    def test_var_fdt(self):
        data = np.array([1, 5, 3, 2, 1, 8])
        fd = fdt(data)
        self.assertClose(fd.var(), data.var(), atol=1.5)

    def test_sd_fdt(self):
        data = np.array([1, 5, 3, 2, 1, 8])
        fd = fdt(data)
        self.assertClose(fd.sd(), data.std(), atol=1.5)

    def test_median(self):
        data = np.array([1, 5, 3, 2, 1, 8])
        fd = fdt(data)
        self.assertClose(fd.median(), np.median(data), atol=0.4)

    def test_quantiles(self):
        data = np.array([1, 5, 3, 2, 1, 8])
        fd = fdt(data)

        # using the median as a reference point here.
        median = fd.median()

        # basic usage - 0 <= `pos` <= 1, `by` unspecified
        self.assertEqual(median, fd.quantile(0.5))

        # 5th decile
        self.assertEqual(median, fd.quantile(5, by=10))

        # 50th percentile
        self.assertEqual(median, fd.quantile(50, by=100))

        # second quartile, using the index `2` on the specified binning.
        # equivalent to fd.quantile(np.arange(0, 1, 0.25)[2])
        self.assertEqual(median, fd.quantile(2, by=np.arange(0, 1, 0.25)))

    def test_mfv(self):
        def mfv_and_compare(data, expected_mfv) -> None:
            calculated_mfv = fdt(data).mfv()
            assert all(calculated_mfv == expected_mfv)

        mfv_and_compare(["a", "b", "b", "c"], ["b"])
        mfv_and_compare(["a", "b", "b", "c", "c"], ["b", "c"])
        self.assertClose(fdt([0, 1, 1, 2, 2, 3]).mfv()[0], 1.515, atol=0.4)

    def test_at(self):
        data = np.array([1, 5, 8, 12, 30])
        fd = fdt(data, binning=Binning.auto(start=data.min(), end=data.max()))
        self.assertEqual(fd.at(), data.max() - data.min())
