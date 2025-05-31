import unittest
import pandas as pd

from fdth import fdt, NumericalFDT, CategoricalFDT

class Test(unittest.TestCase):
    def test_numerical_fdt(self):
        data = [1, 2, 6, 8, 10]
        fd = fdt(data)
        assert isinstance(fd, NumericalFDT)
        _ = fd.get_table()
        _ = fd.mean()
        _ = fd.median()
        _ = fd.var()
        _ = fd.plot_histogram()

    def test_categorical_fdt(self):
        data = ["a", "b", "a", "c", "c"]
        fd = fdt(data)
        assert isinstance(fd, CategoricalFDT)
        _ = fd.get_table()
        _ = fd.plot_histogram()

    def test_categorical_fdt_mixed(self):
        data = [1, 5, "b"]
        fd = fdt(data)
        assert isinstance(fd, CategoricalFDT)
        
