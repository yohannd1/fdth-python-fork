from typing import Literal

import pandas as pd
import numpy as np

from .utils import deduce_fdt_kind
from .frequency_distribution import FrequencyDistribution
from .numerical_fdt import NumericalFDT, NumericalBin
from .categorical_fdt import CategoricalFDT
from .multiple_fdt import MultipleFDT


def fdt(
    data: pd.Series | list | pd.DataFrame | np.ndarray,
    kind: Literal["numerical", "categorical", None] = None,
    **kwargs,
    # sort: bool = True,
    # decreasing: bool = True,
    # k: int | None = None,
    # start: float | None = None,
    # end: float | None = None,
    # h: float | None = None,
    # breaks: NumericalBin = "Sturges",
    # right: bool = False,
    # na_rm: bool = False,
) -> FrequencyDistribution | MultipleFDT:
    """
    Create a frequency distribution table for the specified data.

    TODO: talk about the deductions used to determine whether it will use a NumericalFDT or a CategoricalFDT

    :param data: the input data set, or collection of data sets (with pandas.DataFrame or numpy.ndarray)

    :param sort: (for categorical FDTs) if True, sorts the table by frequency.
    :param decreasing: (for categorical FDTs) if sort is True, sorts in the descending if it is True, otherwise in ascending order.

    :param k: (for numerical FDTs) the number of bins/classes. If None, calculates the number based on the method specified on the `breaks` argument.
    :param start: (for numerical FDTs) the start of the interval range.
    :param end: (for numerical FDTs) the end of the interval range.
    :param h: (for numerical FDTs) the class interval width.
    :param breaks: (for numerical FDTs) method for determining bins ('Sturges', 'Scott', 'FD').
    :param right: (for numerical FDTs) whether to include the right endpoint in each interval.
    :param na_rm: (for numerical FDTs) remove missing values if True.

    :return: a DataFrame containing the frequency distribution table.
    """

    data_series: pd.Series

    if isinstance(data, list):
        data_series = pd.Series(data)
    elif isinstance(data, pd.Series):
        data_series = data
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data_series = pd.Series(data)
        else:
            return MultipleFDT(data)
    elif isinstance(data, pd.DataFrame):
        return MultipleFDT(data)
    else:
        raise TypeError(
            "data must be list | pandas.Series | pandas.DataFrame | numpy.ndarray"
        )

    kind = kind or deduce_fdt_kind(data_series)
    if kind == "categorical":
        return CategoricalFDT(data_series, **kwargs)
    elif kind == "numerical":
        return NumericalFDT(data_series, **kwargs)
    else:
        raise TypeError(f"unexpected kind: {repr(kind)}")
