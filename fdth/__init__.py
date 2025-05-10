import pandas as pd

from .frequency_distribution import FrequencyDistribution
from .numerical_fdt import NumericalFrequencyDistribution
from .categorical_fdt import CategoricalFrequencyDistribution

def fdt(data: pd.Series | list) -> FrequencyDistribution:
    """
    Cria uma distribuição de frequência a depender do tipo de dado fornecido.
    """
    if isinstance(data, list):
        data = pd.Series(data)
    elif not isinstance(data, pd.Series):
        raise TypeError("Data must be a list or pandas.Series")

    if data.dtype == "object" or isinstance(data.iloc[0], str):
        return CategoricalFrequencyDistribution(data)
    else:
        return NumericalFrequencyDistribution(data)
