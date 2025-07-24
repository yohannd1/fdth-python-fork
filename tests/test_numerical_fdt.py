import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fdth import fdt, NumericalFDT

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
