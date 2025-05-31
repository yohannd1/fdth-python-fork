import pandas as pd
import numpy as np

from .utils import deduce_fdt_kind
from .numerical_fdt import NumericalFDT
from .categorical_fdt import CategoricalFDT
from .frequency_distribution import FrequencyDistribution

class MultipleFDT:
    """Contains FDTs of all columns in dataset."""

    def __init__(self, data: pd.DataFrame | np.ndarray, **kwargs) -> None:
        # TODO: doc
        # TODO: arg for selecting only numeric or categorical columns (might use `df.select_dtypes(include=["category", "object"]).columns`)

        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = pd.DataFrame(data)
        else:
            raise TypeError("data must be pandas.DataFrame | numpy.ndarray")

        self.fdts = {k: self._auto_fdt(v, **kwargs) for (k, v) in self._data.items()}
        """A dictionary with the individual FDT objects for each column."""

        self.tables = {k: v.table for (k, v) in self.fdts.items()}
        """A dictionary with the individual tables for each column, skipping the classes."""

    @staticmethod
    def _auto_fdt(data: pd.Series, **kwargs) -> FrequencyDistribution:
        kind = deduce_fdt_kind(data)
        if kind == "categorical":
            return CategoricalFDT(data, **kwargs)
        elif kind == "numerical":
            return NumericalFDT(data, **kwargs)
        else:
            raise TypeError(f"unexpected kind: {repr(kind)}")
