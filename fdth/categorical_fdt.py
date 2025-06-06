from typing import Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdth import FrequencyDistribution


class CategoricalFDT(FrequencyDistribution):
    """Stores information about a categorical frequency distribution, and allows related operations."""

    def __init__(
        self,
        data: pd.Series | list,
        sort: bool = True,
        decreasing: bool = False,
    ):
        self._data: pd.Series
        self.table: pd.DataFrame

        self.sort = sort
        self.decreasing = decreasing

        if isinstance(data, list):
            self._data = pd.Series(data)
        elif isinstance(data, pd.Series):
            self._data = data
        else:
            raise TypeError(
                "data must be list | pandas.Series | pandas.DataFrame | numpy.ndarray"
            )

        self._data = self._data.astype("category")

        self.table = self._make_single_table(
            self._data, sort=sort, decreasing=decreasing
        )
        """The inner frequency distribution table."""

    def get_table(self) -> pd.DataFrame:
        # FIXME: deprecate this (in favor of `self.table`)
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

        plt.show()

    def mfv(self) -> Any:
        return self._data.mode().iloc[0]

    @staticmethod
    def _make_single_table(
        data: pd.Series, sort: bool, decreasing: bool
    ) -> pd.DataFrame:
        """
        Creates a frequency distribution table (FDT) for a set of categorical data.

        Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - "Category": The unique categories.
            - "f": The absolute frequency of each category.
            - "rf": The relative frequency of each category.
            - "rf(%)": The relative frequency expressed as a percentage.
            - "cf": The cumulative absolute frequency.
            - "cf(%)": The cumulative relative frequency expressed as a percentage.
        """

        if not (data.dtypes == "object" or data.dtypes.name == "category"):
            raise ValueError("Values must be strings or categorical.")

        # Convert to categorical type
        data = data.astype("category")

        # Check if there are valid categories
        if len(data.cat.categories) == 0:
            raise ValueError("No valid categories found in the data.")

        # Calculate absolute frequency
        f = data.value_counts(sort=False)

        if sort:
            # Sort by absolute frequencies
            f = f.sort_values(ascending=not decreasing)

        # Calculate relative frequencies and cumulative frequencies
        rf = f / f.sum()  # Relative frequency
        rfp = rf * 100  # Relative frequency as a percentage
        cf = f.cumsum()  # Cumulative absolute frequency
        cfp = rfp.cumsum()  # Cumulative relative frequency as a percentage

        return pd.DataFrame({
            "Category": f.index,
            "f": f.values,
            "rf": rf.values,
            "rf(%)": rfp.values,
            "cf": cf.values,
            "cf(%)": cfp.values,
        }) # fmt: skip

    def __repr__(self) -> str:
        res = f"CategoricalFDT (size {len(self._data)}, category count {self._data.nunique()}), head:\n"
        res += self.table.head().to_string(index=False)
        return res
