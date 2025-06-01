import numpy as np
from typing import Any


def mean_fdt(x: Any) -> float:
    # Definir intervalos de classe com base nos valores 'start', 'end' e 'h'
    breaks = np.arange(
        x["breaks"]["start"], x["breaks"]["end"] + x["breaks"]["h"], x["breaks"]["h"]
    )

    # Calcular pontos médios dos intervalos de classe
    mids = 0.5 * (breaks[:-1] + breaks[1:])

    # Frequências das classes
    y = x["table"][:, 1]

    # Calcular a média ponderada dos pontos médios
    res = np.sum(y * mids) / np.sum(y)

    # Retornar a média
    return res

# Teste de exemplo para a função mean_fdt

"""# Exemplo de entrada de teste para mean_fdt
x_test = {
    'breaks': {
        'start': 0,
        'end': 40,
        'h': 10
    },
    'table': np.array([
        [1, 5],
        [2, 10],
        [3, 15],
        [4, 10]
    ])
}

# Executar a função e imprimir o resultado
resultado = mean_fdt(x_test)
print("Média:", resultado)
"""
