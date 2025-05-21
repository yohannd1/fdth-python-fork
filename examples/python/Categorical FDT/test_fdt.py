from fdth import fdt

"""
# Test 1 em Python: Verifica a criação de uma tabela de frequência categórica,
# ordenando os resultados em ordem decrescente com base na frequência.

x = ["A", "B", "A", "C", "B", "A", "C", "C", "A", "B"]
res_py = fdt(x, sort=True, decreasing=True)
print(res_py.get_table())

# Test 2 em Python: Verifica a criação de uma tabela de frequência categórica
# para um conjunto homogêneo (todos os valores iguais), sem ordenar os resultados.

x = ["X", "X", "X", "X", "X", "X"]
res_py = fdt(x, sort=False, decreasing=False)
print(res_py.get_table())

# Test 3 em Python: Verifica a criação de uma tabela de frequência categórica,
# ordenando os resultados em ordem crescente com base na frequência.

x = ["Dog", "Cat", "Cat", "Dog", "Fish", "Fish", "Bird", "Dog"]
res_py = fdt(x, sort=True, decreasing=False)
print(res_py.get_table())
"""

# Test 3 em Python: Verifica a criação de uma tabela de frequência categórica,
# ordenando os resultados em ordem crescente com base na frequência.

x = ["Dog", "Cat", "Cat", "Dog", "Fish", "Fish", "Bird", "Dog"] # fmt: skip
res_py = fdt(x, sort=True, decreasing=False)
print(res_py.get_table())
