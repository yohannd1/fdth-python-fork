from typing import Optional, Iterable, Any

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from .utils import deduce_fdt_kind
from .numerical_fdt import NumericalFDT
from .categorical_fdt import CategoricalFDT


class MultipleFDT:
    """Contains FDTs of all columns in a data set (table or matrix)."""

    fdts_by_column: dict[Any, NumericalFDT | CategoricalFDT]
    """A dictionary with the individual FDT objects for each column."""

    fdts_by_index: list[NumericalFDT | CategoricalFDT]
    """A list with the individual FDT objects for each column index."""

    def __init__(self, data: pd.DataFrame | NDArray, **kwargs) -> None:
        """
        Create a MultipleFDT based on tabular data (a dataframe or 2-dimensional array).

        :param kwargs: forwarded to each NumericalFDT and CategoricalFDT - one per column in `data`.
        """
        if isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = pd.DataFrame(data)
        else:
            raise TypeError("data must be pandas.DataFrame | numpy.ndarray")

        self.fdts_by_column = {
            k: self._auto_fdt(v, **kwargs) for (k, v) in self._data.items()
        }

        self.fdts_by_index = [self.fdts_by_column[k] for (k, _) in self._data.items()]

    def get_fdt(
        self,
        column: Any = None,
        index: Optional[int] = None,
    ) -> NumericalFDT | CategoricalFDT:
        if column is not None and index is not None:
            raise ValueError(
                "both `column` and `index` were specified - specify exactly one"
            )
        elif column is not None:
            return self.fdts_by_column[column]
        elif index is not None:
            return self.fdts_by_index[index]
        else:
            raise ValueError(
                "neither `column` nor `index` were specified - specify exactly one"
            )

    def get_table(
        self,
        column: Any = None,
        index: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.get_fdt(column, index).get_table()

    @staticmethod
    def _drop_keys(d: dict[Any, Any], keys: Iterable[Any]) -> None:
        for k in keys:
            if k in d:
                del d[k]

    @staticmethod
    def _auto_fdt(data: pd.Series, **kwargs) -> NumericalFDT | CategoricalFDT:
        kind = deduce_fdt_kind(data)
        if kind == "categorical":
            MultipleFDT._drop_keys(kwargs, ("binning", "start", "end", "h", "k"))
            return CategoricalFDT(data, **kwargs)
        elif kind == "numerical":
            MultipleFDT._drop_keys(kwargs, ("sort", "decreasing"))
            return NumericalFDT(data, **kwargs)
        else:
            raise TypeError(f"unexpected kind: {repr(kind)}")

    def __repr__(self) -> str:
        res = f"MultipleFDT ({len(self.fdts_by_index)} tables):\n\n"
        for k, v in self.fdts_by_column.items():
            res += f"{k}: {v}\n\n"
        return res
