from typing import Literal, Optional, Any

import pandas as pd
import numpy as np

from .utils import deduce_fdt_kind
from .numerical_fdt import NumericalFDT
from .categorical_fdt import CategoricalFDT
from .multiple_fdt import MultipleFDT


def fdt(
    data: Optional[pd.Series | list | pd.DataFrame | np.ndarray] = None,
    *,
    freqs: Optional[pd.Series | list | dict[Any, int]] = None,
    kind: Literal["numerical", "categorical", None] = None,
    **kwargs,
) -> NumericalFDT | CategoricalFDT | MultipleFDT:
    """
    Create a frequency distribution table for a given data set or frequency set, automatically detecting whether it is supposed to refer to a numerical one, a categorical one or a combination 2 or more of them.

    Trailing parameters are forwarded to the constructors of `fdth.numerical_fdt.NumericalFDT`, `fdth.categorical_fdt.CategoricalFDT` and `fdth.multiple_fdt.MultipleFDT`.

    :param data: the input data set, or collection of data sets (with pandas.DataFrame or numpy.ndarray)
    :param freqs: frequencies, as an alternative to inputting the data itself. If it is a series, it will be interpreted as numerical, and if it is a dictionary, it will be interpreted as categorical.
    """

    data_: pd.Series

    if isinstance(data, list):
        data_ = pd.Series(data)
    elif isinstance(data, pd.Series):
        data_ = data
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data_ = pd.Series(data)
        else:
            return MultipleFDT(data)
    elif isinstance(data, pd.DataFrame):
        return MultipleFDT(data)
    elif data is None and freqs is not None:
        if isinstance(freqs, pd.Series | list):
            if kind is not None and kind != "numerical":
                raise TypeError(
                    "`freqs` (as pandas.Series | list) can only be used with `numerical` type FDTs"
                )
            return NumericalFDT(freqs=freqs, **kwargs)
        elif isinstance(freqs, dict):
            if kind is not None and kind != "categorical":
                raise TypeError(
                    "`freqs` (as dict) can only be used with `categorical` type FDTs"
                )
            return CategoricalFDT(freqs=freqs, **kwargs)
        else:
            raise TypeError(
                "`freqs` must be pandas.Series | list | dict when specified"
            )
    else:
        raise TypeError(
            "`data` must be list | pandas.Series | pandas.DataFrame | numpy.ndarray, or `freqs` must be specified"
        )

    kind = kind or deduce_fdt_kind(data_)
    if kind == "categorical":
        return CategoricalFDT(data_, **kwargs)
    elif kind == "numerical":
        return NumericalFDT(data_, **kwargs)
    else:
        raise TypeError(f"unexpected kind: {repr(kind)}")
