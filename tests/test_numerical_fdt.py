import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fdth import fdt, NumericalFDT, Binning

class Test(unittest.TestCase):
    def test_numerical_fdt(self):
        data = [1, 2, 6, 8, 10]
        fd = fdt(data)
        assert isinstance(fd, NumericalFDT)

        # tabela
        table = fd.get_table()
        assert isinstance(table, pd.DataFrame)
        assert not table.empty

        # média
        mean_value = fd.mean()
        assert isinstance(mean_value, float)

        # mediana
        median_value = fd.median()
        assert isinstance(median_value, float)

        # variância
        var_value = fd.var()
        assert isinstance(var_value, float)

        # testa o __repr__
        repr_output = repr(fd)
        assert isinstance(repr_output, str)
        assert "NumericalFDT" in repr_output

    def test_binnings(self):
        data = [1, 2, 6, 8, 10]
        fd = fdt(data, binning=Binning.from_sturges)
        fd = fdt(data, binning=Binning.from_scott)
        fd = fdt(data, binning=Binning.from_fd)

    def test_from_freqs(self):
        freqs = [5, 2, 1, 3]
        fd = fdt(freqs=pd.Series(freqs), binning=Binning.linspace(start=1, end=10, k=4))
        fd = NumericalFDT(freqs=freqs, binning=Binning.linspace(start=1, end=10, k=4))

    def shorthand_notation(self):
        data = [1, 5, 3, 8, 10]
        fd1 = fdt(data, start=1, end=10, k=3)
        fd2 = fdt(data, binning=Binning.auto(start=1, end=10, k=3))
        self.assertEqual(fd1.to_string(), fd2.to_string())
