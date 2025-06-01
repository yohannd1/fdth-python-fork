import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fdth import CategoricalFDT

def test_categorical_fdt():
    data = ["Azul", "Vermelho", "Azul", "Verde", "Azul", "Vermelho"]
    fd = CategoricalFDT(data)
    assert isinstance(fd, CategoricalFDT)

    # Tabela
    table = fd.get_table()
    assert isinstance(table, pd.DataFrame)
    assert not table.empty

    # Plotar histograma
    _ = fd.plot_histogram()
    plt.close()

    # Testa o __repr__
    repr_output = repr(fd)
    assert isinstance(repr_output, str)
    assert "CategoricalFDT" in repr_output
    assert "Número de dados:" in repr_output
    assert "Tabela de Frequência" in repr_output

    print(repr_output)

if __name__ == "__main__":
    test_categorical_fdt()
    print("Categorical FDT tests completed.")
