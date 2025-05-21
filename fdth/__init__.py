import pandas as pd

from .frequency_distribution import FrequencyDistribution
from .numerical_fdt import NumericalFDT
from .categorical_fdt import CategoricalFDT

def fdt(data: pd.Series | list, sort: bool = True, decreasing: bool = True) -> FrequencyDistribution:
    """
    Create a frequency distribution table for the specified data.

    TODO: talk about the deductions used to determine whether it will use a NumericalFDT or a CategoricalFDT

    :param data: the input data set
    :param sort: if True, sorts the table by frequency. Only works on categorical data.
    :param decreasing: if sort is True, sorts in the descending if it is True, otherwise in ascending order. Only works on categorical data.
    :return: a DataFrame containing the frequency distribution table.
    """
    if isinstance(data, list):
        data = pd.Series(data)
    elif not isinstance(data, pd.Series):
        raise TypeError("Data must be a list or pandas.Series")

    if data.dtype == "object" or isinstance(data.iloc[0], str):
        return CategoricalFDT(data, sort=sort, decreasing=decreasing)
    else:
        # FIXME: ignora sort e decreasing mesmo??
        return NumericalFDT(data)
