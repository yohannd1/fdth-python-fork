from fdth import fdt, NumericalFrequencyDistribution, CategoricalFrequencyDistribution

import unittest
import pandas as pd

class Test(unittest.TestCase):
    def test_numerical_fdt(self):
        data = [1, 2, 6, 8, 10]
        fd = fdt(data)
        assert isinstance(fd, NumericalFrequencyDistribution)
        _ = fd.get_table()
        _ = fd.mean()
        _ = fd.median()
        _ = fd.var()

    def test_categorical_fdt(self):
        data = ["a", "b", "a", "c", "c"]
        fd = fdt(data)
        assert isinstance(fd, CategoricalFrequencyDistribution)
        _ = fd.get_table()
