import unittest

from fdth.summary_fdt_cat_default import summary_fdt_cat_default
from fdth import fdt


class Test(unittest.TestCase):
    def test_purely_categorical(self):
        data = ["Masculino", "Feminino", "Feminino", "Masculino", "Masculino", "Feminino", "Outro", "Masculino", "Feminino", "Outro", "Feminino", "Masculino", "Outro", "Masculino", "Feminino", "Masculino", "Outro", "Feminino", "Outro", "Feminino", "Masculino", "Outro", "Feminino", "Outro", "Masculino", "Feminino", "Masculino", "Outro", "Outro", "Feminino"] # fmt: skip

        fd = fdt(data)
        summary_fdt_cat_default(
            object=fd.get_table(),
            columns=[0, 1, 3, 4],
            round=2,
            row_names=False,
            right=False,
        )

    def test_categorical_mixed(self):
        data = ["A", "B", "C", 1, 2, 3, "A", 1, "C", "B", 2, "A", "C", 3, "B", 1, "C", "A", 2, 3, "B", 1, "C", 2, "A", "B", 3, "C", 1, 2] # fmt: skip

        fd = fdt(data)
        summary_fdt_cat_default(object=fd.get_table(), round=2)
