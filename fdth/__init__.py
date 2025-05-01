import pandas as pd

from .frequency_distribution import FrequencyDistribution
from .fdt_data_frame import NumericalFrequencyDistribution
from .fdt_cat_data_frame import CategoricalFrequencyDistribution

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
