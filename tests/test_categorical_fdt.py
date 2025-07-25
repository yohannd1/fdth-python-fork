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

    def test_plot(self):
        data = ["Blue", "Red", "Blue", "Green", "Blue", "Red"]
        fd = fdt(data)
        types = ["fb", "fp", "fd", "pa", "rfb", "rfp", "rfd", "rfpb", "rfpp", "rfpd", "cfb", "cfp", "cfd", "cfpb", "cfpp", "cfpd"] # fmt: skip
        for type_ in types:
            fd.plot(type_=type_, show=False)
