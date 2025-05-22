from typing import Literal

import pandas as pd

from .frequency_distribution import FrequencyDistribution
from .numerical_fdt import NumericalFDT, NumericalBin
from .categorical_fdt import CategoricalFDT

FdtKind = Literal["numerical", "categorical", "auto"]


def fdt(
    data: pd.Series | list,
    *,
    kind: FdtKind = "auto",
    sort: bool = True,
    decreasing: bool = True,
    k: int | None = None,
    start: float | None = None,
    end: float | None = None,
    h: float | None = None,
    breaks: NumericalBin = "Sturges",
    right: bool = False,
    na_rm: bool = False,
) -> FrequencyDistribution:
    """
    Create a frequency distribution table for the specified data.

    TODO: talk about the deductions used to determine whether it will use a NumericalFDT or a CategoricalFDT

    :param data: the input data set

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
    if isinstance(data, list):
        data = pd.Series(data)
    elif not isinstance(data, pd.Series):
        raise TypeError("Data must be a list or pandas.Series")

    if kind == "auto":
        kind = _deduce_kind(data)

    if kind == "categorical":
        return CategoricalFDT(data, sort=sort, decreasing=decreasing)
    elif kind == "numerical":
        return NumericalFDT(
            data,
            k=k,
            start=start,
            end=end,
            h=h,
            breaks=breaks,
            right=right,
            na_rm=na_rm,
        )
    else:
        raise TypeError(f"unexpected kind: {repr(kind)}")


def _deduce_kind(data: pd.Series) -> Literal["categorical", "numerical"]:
    if data.dtype == "object" or isinstance(data.iloc[0], str):
        return "categorical"
    return "numerical"
