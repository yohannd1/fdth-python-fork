import unittest

import pandas as pd
import numpy as np

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
        fd.plot_histogram()

    def test_categorical_fdt(self):
        data = ["a", "b", "a", "c", "c"]
        fd = fdt(data)
        assert isinstance(fd, CategoricalFDT)
        _ = fd.get_table()
        fd.plot_histogram()

    def test_categorical_fdt_mixed(self):
        data = [1, 5, "b"]
        fd = fdt(data)
        assert isinstance(fd, CategoricalFDT)

    def test_dataframe_fdt(self):
        df = pd.DataFrame({
            "A": [1, "j", 2],
            "B": [10, 20, 30],
        }) # fmt: skip
        fd = fdt(df)
        assert isinstance(fd.fdts["A"], CategoricalFDT)
        assert isinstance(fd.fdts["B"], NumericalFDT)

    def test_ndarray_fdt(self):
        x = np.array([
            ["Dog", "Cat", "Fish"],
            ["Dog", "Dog", "Bird"],
            ["Cat", "Cat", "Fish"],
            ["Fish", "Dog", "Bird"],
        ]) # fmt: skip
        fd = fdt(x)

        assert len(fd.fdts) == 3
        assert isinstance(fd.fdts[0], CategoricalFDT)
        assert isinstance(fd.fdts[1], CategoricalFDT)
        assert isinstance(fd.fdts[2], CategoricalFDT)
