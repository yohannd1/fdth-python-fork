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

    def test_categorical_fdt(self):
        data = ["a", "b", "a", "c", "c"]
        fd = fdt(data)
        assert isinstance(fd, CategoricalFDT)

        _ = fd.get_table()

    def test_categorical_fdt_mixed(self):
        data = [1, 5, "b"]
        fd = fdt(data)
        assert isinstance(fd, CategoricalFDT)

    def test_categorical_fdt_numerical(self):
        data = [1, 5, 2, 3, 8, 10]
        fd = fdt(data, kind="categorical")
        assert isinstance(fd, CategoricalFDT)

        _ = fd.get_table()

    def test_dataframe_fdt(self):
        df = pd.DataFrame({
            "A": [1, "j", 2],
            "B": [10, 20, 30],
        }) # fmt: skip
        fd = fdt(df)

        # expected types for each column
        assert isinstance(fd.get_fdt("A"), CategoricalFDT)
        assert isinstance(fd.get_fdt("B"), NumericalFDT)

        # different ways to get the same FDT
        assert fd.get_fdt("A") is fd.get_fdt(index=0)
        assert fd.get_fdt("B") is fd.get_fdt(index=1)

    def test_ndarray_fdt(self):
        x = np.array([
            ["Dog", "Cat", "Fish"],
            ["Dog", "Dog", "Bird"],
            ["Cat", "Cat", "Fish"],
            ["Fish", "Dog", "Bird"],
        ]) # fmt: skip
        fd = fdt(x)

        assert len(fd.fdts_by_index) == 3
        assert isinstance(fd.get_fdt(0), CategoricalFDT)
        assert isinstance(fd.get_fdt(1), CategoricalFDT)
        assert isinstance(fd.get_fdt(2), CategoricalFDT)

    def test_dataframe_fdt_mixed(self):
        df = pd.DataFrame({
            "foo": ["bar", "baz", "abc", "jj", "Bin"],
            "bar": [1, 5, 3, 8, 10],
        })
        fd = fdt(df, sort=False)
        assert isinstance(fd.get_fdt("foo"), CategoricalFDT)
        assert isinstance(fd.get_fdt("bar"), NumericalFDT)
