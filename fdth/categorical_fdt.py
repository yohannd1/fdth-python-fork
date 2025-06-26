from typing import Optional, Any
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdth import FrequencyDistribution


class CategoricalFDT(FrequencyDistribution):
    """Stores information about a categorical frequency distribution, and allows related operations."""

    table: pd.DataFrame
    """The inner frequency distribution table. Columns:
        - "Category": the unique categories.
        - "f": the absolute frequency of each category.
        - "rf": the relative frequency of each category.
        - "rf(%)": the relative frequency expressed as a percentage.
        - "cf": the cumulative absolute frequency.
        - "cf(%)": the cumulative relative frequency expressed as a percentage.
    """

    def __init__(
        self,
        data: Optional[pd.Series | list] = None,
        *,
        freqs: Optional[pd.Series | dict[Any, int]] = None,
        sort: bool = True,
        decreasing: bool = False,
    ) -> None:
        """Create the frequency distribution class.

        Either `data` or `freqs` must be specified.

        :param data: a data set of which the frequency must be analyzed
        :param freqs: a pandas.Series with the value at a specific index being the absolute frequency of said category, or a dict with the key being a category and the value being the frequency
        :param sort: TODO
        :param decreasing: TODO
        """

        if data is not None:
            if freqs is not None:
                raise ValueError("`data` and `freqs` must not be both specified")

            if isinstance(data, list):
                data = pd.Series(data)
            elif isinstance(data, pd.Series):
                pass
            else:
                raise TypeError("`data` must be list | pandas.Series")

            # FIXME: don't save this in the class!
            self._data = data.astype("category")

            self.count, self.table = self._make_table_from_data(
                data, sort=sort, decreasing=decreasing
            )
        elif freqs is not None:
            if data is not None:
                raise ValueError("`data` and `freqs` must not be both specified")

            if isinstance(freqs, dict):
                freqs = pd.Series(freqs)
            elif isinstance(freqs, pd.Series):
                freqs = freqs
            else:
                raise TypeError("`freqs` must be dict | pandas.Series")

            self.count, self.table = self._make_table_from_frequencies(
                freqs, sort=sort, decreasing=decreasing
            )
        else:
            raise ValueError("one of `data` or `table` must be specified")

    def get_table(self) -> pd.DataFrame:
        return self.table

    def plot_histogram(self) -> None:
        category_counts = pd.Series(self._data).value_counts()

        # plotar o gráfico de barras
        category_counts.plot(kind="bar", color="skyblue", edgecolor="black")

        # definir título e rótulos
        plt.title("Histograma de Dados Categóricos")
        plt.xlabel("Categorias")
        plt.ylabel("Frequência")

        # rotacionar os rótulos das categorias para ficarem legíveis
        plt.xticks(rotation=0)

    @lru_cache(maxsize=1)
    def mfv(self) -> pd.Series:
        """Returns the most frequent values (modes) of the data set."""
        return self._data.mode().iloc[0:]

    @staticmethod
    def _make_table_from_frequencies(
        freqs: pd.Series | dict[Any, int], sort: bool, decreasing: bool
    ) -> tuple[int, pd.DataFrame]:
        """Make a frequency distribution table from a series of frequencies."""

        if isinstance(freqs, dict):
            freqs = pd.Series(freqs)

        if sort:
            # Sort by absolute frequencies
            freqs = freqs.sort_values(ascending=not decreasing)

        count = freqs.sum()

        # Calculate relative frequencies and cumulative frequencies
        rf = freqs / count  # Relative frequency
        rfp = rf * 100  # Relative frequency as a percentage
        cf = freqs.cumsum()  # Cumulative absolute frequency
        cfp = rfp.cumsum()  # Cumulative relative frequency as a percentage

        return count, pd.DataFrame({
            "Category": freqs.index,
            "f": freqs.values,
            "rf": rf.values,
            "rf(%)": rfp.values,
            "cf": cf.values,
            "cf(%)": cfp.values,
        }) # fmt: skip

    @staticmethod
    def _make_table_from_data(
        data: pd.Series, sort: bool, decreasing: bool
    ) -> tuple[int, pd.DataFrame]:
        """
        Create a frequency distribution table (FDT) for a set of categorical data.
        """

        # FIXME: is this needed? it would make using numbered categories impossible
        # if data.dtypes.name not in {"object", "category"}:
        #     raise ValueError(f"values must be strings or categorical (got {data.dtypes.name}).")

        count = len(data)

        # Convert data set to categorical type
        data = data.astype("category")

        # Check if there are valid categories
        if len(data.cat.categories) == 0:
            raise ValueError("No valid categories found in the data.")

        # Calculate absolute frequencies
        freqs = data.value_counts(sort=False)

        return CategoricalFDT._make_table_from_frequencies(
            freqs=freqs, sort=sort, decreasing=decreasing
        )

    def to_string(
        self,
        columns: list[str] | None = None,
        round: int = 2,
        right: bool = True,
        row_numbers: bool = False,
        max_lines: int | None = None,
    ) -> str:
        table = self.table

        if max_lines is not None:
            table = table.head(max_lines)

        # filter by columns if any were specified
        if columns is not None:
            table = pd.concat([table["Category"], table[columns]], axis="columns")

        # round the numbers in the table
        table = table.round(round)

        return table.to_string(index=row_numbers, justify="right" if right else "left")

    def __repr__(self) -> str:
        cat_count = self.table.shape[0]
        res = f"CategoricalFDT ({self.count} elements, {cat_count} categories):\n"
        res += self.to_string(max_lines=5) + "\n"
        if self.count > 5:
            res += f"... {self.count-5} more lines"

        return res
