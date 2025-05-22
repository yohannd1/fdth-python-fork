import unittest

import numpy as np

from fdth.mean import mean_fdt
from fdth.var_fdt_default import var_fdt
from fdth.sd_fdt_default import sd_fdt


class Test(unittest.TestCase):
    def testar_mean_fdt(self):
        x_test = {
            "breaks": {"start": 0, "end": 40, "h": 10},
            "table": np.array([[1, 5], [2, 10], [3, 15], [4, 10]]),
        }
        resultado = mean_fdt(x_test)
        esperado = 22.5
        assert np.isclose(
            resultado, esperado
        ), f"expected around {esperado}, got {resultado}"

    def testar_var_fdt(self):
        x_test = {
            "breaks": {"start": 0, "end": 40, "h": 10},
            "table": np.array([[1, 5], [2, 10], [3, 15], [4, 10]]),
        }
        resultado = var_fdt(x_test)
        esperado = 96.15
        assert np.isclose(
            resultado, esperado, atol=0.01
        ), f"expected around {esperado}, got {resultado}"

    def testar_sd_fdt(self):
        x_test = {
            "breaks": {"start": 0, "end": 40, "n": 4},
            "table": np.array([[1, 5], [2, 10], [3, 15], [4, 10]]),
        }
        resultado = sd_fdt(x_test)
        esperado = 9.8
        assert np.isclose(
            resultado, esperado, atol=0.01
        ), f"expected around {esperado}, got {resultado}"
