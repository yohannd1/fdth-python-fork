from fdth.summary_fdt_default import print_fdt_default
from fdth import fdt

# Vetor inicial
data = [6.34, 4.57, 7.89, 5.12, 4.26, 5.77, 2.95, 8.13, 3.48, 6.05, 4.93, 6.88, 7.21, 3.69, 5.55, 2.87, 5.02, 4.31, 6.79, 3.98, 7.44, 5.36, 6.12, 4.59, 8.27, 3.65, 5.48, 7.81, 3.93, 5.67] # fmt: skip

# Gera uma tdf
fd = fdt(data, breaks="Sturges")
tbl = fd.get_table()

# Faz a sumarização e formatação de duas maneiras diferentes
print("Exemplo com seleção de colunas e nome de colunas")
print_fdt_default(
    tbl, columns=[0, 1, 2], round=2, format_classes=False, row_names=True
)
print()

print("Exemplo com seleção de colunas e formatação de classes")
col = [0, 3, 5]
print_fdt_default(tbl, columns=col, format_classes=True, pattern="{:.5e}")

"""
A formatação das classe é feita através da função "make_fdt_format_classes", chamada durante a 
execução da função summary_fdt_default.
"""
