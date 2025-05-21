from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdth import FrequencyDistribution


class CategoricalFDT(FrequencyDistribution):
    def __init__(
        self,
        data: pd.DataFrame | pd.Series | list,
        sort: bool = True,
        decreasing: bool = False,
        column_name: Optional[str] = None,
    ):
        self.sort = sort
        self.decreasing = decreasing

        self.data: pd.Series
        if isinstance(data, list):
            self.data = pd.Series(data)
        elif isinstance(data, pd.Series):
            self.data = data
        elif isinstance(data, pd.DataFrame):
            raise NotImplementedError(
                "TODO: use `column_name` and `self._generate_fdt`"
            )
        else:
            raise TypeError("Data must be a list, a pandas.DataFrame or pandas.Series")

        self.data = self.data.astype("category")  # convert to category type
        self.table: pd.DataFrame | None = None

    def get_table(self) -> pd.DataFrame:
        """Get the frequency distribution table as a DataFrame."""
        if self.table is None:
            self.table = self._make_table(
                self.data, sort=self.sort, decreasing=self.decreasing
            )
        return self.table

    def plot_histogram(self) -> None:
        category_counts = pd.Series(self.data).value_counts()

        # plotar o gráfico de barras
        category_counts.plot(kind="bar", color="skyblue", edgecolor="black")

        # definir título e rótulos
        plt.title("Histograma de Dados Categóricos")
        plt.xlabel("Categorias")
        plt.ylabel("Frequência")

        # rotacionar os rótulos das categorias para ficarem legíveis
        plt.xticks(rotation=0)

        plt.show()

    def mean(self):
        raise NotImplementedError("TODO")

    def var(self):
        raise NotImplementedError("TODO")

    def mode(self):
        return self.data.mode().iloc[0]

    @staticmethod
    def _make_table(
        x: pd.Series, sort: bool = False, decreasing: bool = False
    ) -> pd.DataFrame:
        """
        Creates a frequency distribution table (FDT) for categorical data.

        :param x: the input data.
        :param sort: if True, sorts the table by frequency.
        :param decreasing: if True, sorts in descending order.

        Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - "Category": The unique categories.
            - "f": The absolute frequency of each category.
            - "rf": The relative frequency of each category.
            - "rf(%)": The relative frequency expressed as a percentage.
            - "cf": The cumulative absolute frequency.
            - "cf(%)": The cumulative relative frequency expressed as a percentage.
        """
        if not isinstance(x, (pd.Series, list)):
            raise TypeError("Input data must be a list or pandas Series.")

        # Convert to pandas Series if it"s a list
        x = pd.Series(x)

        if not (x.dtypes == "object" or x.dtypes.name == "category"):
            raise ValueError("Values must be strings or categorical.")

        # Convert to categorical type
        x = x.astype("category")

        # Check if there are valid categories
        if len(x.cat.categories) == 0:
            raise ValueError("No valid categories found in the data.")

        # Calculate absolute frequency
        f = x.value_counts(sort=False)

        if sort:
            # Sort by absolute frequencies
            f = f.sort_values(ascending=not decreasing)

        # Calculate relative frequencies and cumulative frequencies
        rf = f / f.sum()  # Relative frequency
        rfp = rf * 100  # Relative frequency as a percentage
        cf = f.cumsum()  # Cumulative absolute frequency
        cfp = rfp.cumsum()  # Cumulative relative frequency as a percentage

        # Ensure the result is returned as a DataFrame
        res = pd.DataFrame(
            {
                "Category": f.index,
                "f": f.values,
                "rf": rf.values,
                "rf(%)": rfp.values,
                "cf": cf.values,
                "cf(%)": cfp.values,
            }
        )
        return res

    @staticmethod
    def _generate_fdt(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        # FIXME: acho bom mover isso para outro lugar
        """
        Generate the frequency distribution table for the specified column in the DataFrame.

        :param df: the input dataframe
        :param column_name: the column name to generate the FDT for

        :return: a DataFrame with the frequency distribution table for the specified column.
        """

        # Check if the DataFrame has the specified column
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        # Ensure the column is categorical
        df[column_name] = df[column_name].astype("category")

        # Generate the FDT for the specified column
        return CategoricalFDT._make_table(
            df[column_name], sort=True, decreasing=True
        )
