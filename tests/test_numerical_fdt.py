import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fdth import fdt, NumericalFDT


def test_numerical_fdt():
    data = [1, 2, 6, 8, 10]
    fd = fdt(data)
    assert isinstance(fd, NumericalFDT)

    # Tabela
    table = fd.get_table()
    assert isinstance(table, pd.DataFrame)
    assert not table.empty

    # Média
    # mean_value = fd.mean()
    # assert isinstance(mean_value, float)

    # Mediana
    # median_value = fd.median()
    # assert isinstance(median_value, float)

    # Variância
    # var_value = fd.var()
    # assert isinstance(var_value, float)

    # Histograma
    _ = fd.plot_histogram()
    plt.close()  # Fecha o gráfico pra não travar

    # Testa o __repr__
    repr_output = repr(fd)
    assert isinstance(repr_output, str)
    assert "NumericalFDT" in repr_output
    assert "Número de dados:" in repr_output
    assert "Tabela de Frequência" in repr_output

    print(repr_output)


if __name__ == "__main__":
    test_numerical_fdt()
    print("Numerical FDT tests completed.")
