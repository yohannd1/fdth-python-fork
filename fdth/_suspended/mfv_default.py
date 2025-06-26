from collections import Counter
from typing import TypeVar, Optional, Sequence, Any

T = TypeVar("T")


def mfv_default(x: Sequence[T], **kwargs: Any) -> Optional[T]:
    # Se o conjunto de dados estiver vazio, retorne None
    if len(x) == 0:
        return None

    # Conta a frequência de cada elemento no conjunto de dados
    frequency = Counter(x)

    # Encontra a frequência máxima
    max_freq = max(frequency.values())

    # Seleciona o primeiro elemento que tem a frequência máxima
    mode = next(key for key, value in frequency.items() if value == max_freq)

    return mode
