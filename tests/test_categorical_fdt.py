import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fdth import fdt, CategoricalFDT

class Test(unittest.TestCase):
    def test_categorical_fdt(self):
        data = ["Blue", "Red", "Blue", "Green", "Blue", "Red"]
        fd = CategoricalFDT(data)
        assert isinstance(fd, CategoricalFDT)

        # tabela
        assert isinstance(fd.table, pd.DataFrame)
        assert not fd.table.empty

        # plotar histograma
        fd.plot_histogram()

        # testa o __repr__
        repr_output = repr(fd)
        assert isinstance(repr_output, str)
        assert "CategoricalFDT" in repr_output

    def test_data_and_freqs(self):
        data = ["Blue", "Red", "Blue", "Green", "Blue", "Red"]
        fd_1 = CategoricalFDT(data)
        fd_2 = CategoricalFDT(freqs={"Blue": 3, "Red": 2, "Green": 1})
        fd_3 = fdt(freqs={"Blue": 3, "Red": 2, "Green": 1})
        self.assertEqual(fd_1.to_string(), fd_2.to_string())
        self.assertEqual(fd_1.to_string(), fd_3.to_string())
