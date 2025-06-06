import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fdth import CategoricalFDT

class Test(unittest.TestCase):
    def test_categorical_fdt(self):
        data = ["Azul", "Vermelho", "Azul", "Verde", "Azul", "Vermelho"]
        fd = CategoricalFDT(data)
        assert isinstance(fd, CategoricalFDT)

        # tabela
        table = fd.get_table()
        assert isinstance(table, pd.DataFrame)
        assert not table.empty

        # plotar histograma
        fd.plot_histogram()

        # testa o __repr__
        repr_output = repr(fd)
        assert isinstance(repr_output, str)
        assert "CategoricalFDT" in repr_output
