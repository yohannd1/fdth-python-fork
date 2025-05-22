import unittest

from fdth.median_fdt import median_fdt
from fdth import fdt

class Test(unittest.TestCase):
    def test_median(self):
        data = [10, 12, 15, 20, 22, 25, 25, 30, 35, 40]
        fd = fdt(data)
        mediana = median_fdt(tabela_fdt)
        print("A mediana é:", mediana)
