from typing import Optional, Any

import pandas as pd
import numpy as np

from .utils import deduce_fdt_kind
from .numerical_fdt import NumericalFDT
from .categorical_fdt import CategoricalFDT
from .frequency_distribution import FrequencyDistribution


class MultipleFDT:
    """Contains FDTs of all columns in a data set (table or matrix)."""

    def __init__(self, data: pd.DataFrame | np.ndarray, **kwargs) -> None:
        # TODO: doc
        # TODO: arg for selecting only numeric or categorical columns (might use `df.select_dtypes(include=["category", "object"]).columns`)

        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = pd.DataFrame(data)
        else:
            raise TypeError("data must be pandas.DataFrame | numpy.ndarray")

        self.fdts_by_column = {k: self._auto_fdt(v, **kwargs) for (k, v) in self._data.items()}
        """A dictionary with the individual FDT objects for each column."""

        self.fdts_by_index = [self.fdts_by_column[k] for (k, _) in self._data.items()]
        """A list with the individual FDT objects for each column index."""

    def get_fdt(self, column: Any = None, index: Optional[int] = None) -> FrequencyDistribution:
        if column is not None and index is not None:
            raise ValueError("both `column` and `index` were specified - specify exactly one")
        elif column is not None:
            return self.fdts_by_column[column]
        elif index is not None:
            return self.fdts_by_index[index]
        else:
            raise ValueError("neither `column` nor `index` were specified - specify exactly one")

    def get_table(self, column: Any = None, index: Optional[int] = None) -> pd.DataFrame:
        return self.get_fdt(column, index).get_table()

    @staticmethod
    def _auto_fdt(data: pd.Series, **kwargs) -> FrequencyDistribution:
        kind = deduce_fdt_kind(data)
        if kind == "categorical":
            return CategoricalFDT(data, **kwargs)
        elif kind == "numerical":
            return NumericalFDT(data, **kwargs)
        else:
            raise TypeError(f"unexpected kind: {repr(kind)}")

    def __repr__(self) -> str:
        res = f"MultipleFDT ({len(self.fdts)} tables):\n\n"
        for k, v in self.fdts_by_column.items():
            res += f"{k}: {v}\n\n"
        return res
