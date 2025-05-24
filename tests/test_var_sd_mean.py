import unittest

import numpy as np

from fdth.var_fdt_default import var_fdt
from fdth.sd_fdt_default import sd_fdt


class Test(unittest.TestCase):
    def assertClose(self, value, expected, atol=1e-08) -> None:
        assert np.isclose(value, expected, atol=atol), f"expected around {expected}, got {value}"

    def testar_mean_fdt(self):
        x_test = {
            "breaks": {"start": 0, "end": 40, "h": 10},
            "table": np.array([[1, 5], [2, 10], [3, 15], [4, 10]]),
        }
        resultado = x_test["table"].mean()
        esperado = 22.5
        self.assertClose(resultado, esperado)

    def testar_var_fdt(self):
        x_test = {
            "breaks": {"start": 0, "end": 40, "h": 10},
            "table": np.array([[1, 5], [2, 10], [3, 15], [4, 10]]),
        }
        resultado = var_fdt(x_test)
        esperado = 96.15
        self.assertClose(resultado, esperado, atol=0.01)

    def testar_sd_fdt(self):
        x_test = {
            "breaks": {"start": 0, "end": 40, "h": 4},
            "table": np.array([[1, 5], [2, 10], [3, 15], [4, 10]]),
        }
        resultado = sd_fdt(x_test)
        esperado = 9.8
        self.assertClose(resultado, esperado, atol=0.01)
