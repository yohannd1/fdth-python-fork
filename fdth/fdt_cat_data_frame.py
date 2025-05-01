import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdth import FrequencyDistribution

class CategoricalFrequencyDistribution(FrequencyDistribution):
    def __init__(self, data: pd.Series | list):
        if isinstance(data, list):
            self.data = pd.Series(data)
        elif isinstance(data, pd.Series):
            self.data = data
        else:
            raise TypeError("Data must be a list or pandas.Series")

        self.data = self.data.astype("category") # convert to category type
        self.fdt: pd.DataFrame | None = None

    def ___str___(self) -> str:
        # FIXME: acho que isso aqui não funciona mais (e não por estar com ___ e não __)
        """
        Provide a formatted string representation of the FDT results.

        Returns:
            str: A formatted string containing the FDT for each column or group.
        """
        output = []
        for key, value in self.results.items():
            output.append(f"--- {key} ---")
            output.append("Table:")
            output.append(value.to_string(index=True))
            output.append("")
        return "\n".join(output)

    def get_table(self) -> pd.DataFrame:
        if self.fdt is None:
            self.fdt = self._make_table(self.data) # TODO: especificar sort & decreasing
        return self.fdt

    def plot_histogram(self) -> None:
        category_counts = pd.Series(self.data).value_counts()

        # Plotando o gráfico de barras
        category_counts.plot(kind="bar", color="skyblue", edgecolor="black")

        # Definindo título e rótulos
        plt.title("Histograma de Dados Categóricos")
        plt.xlabel("Categorias")
        plt.ylabel("Frequência")
        plt.xticks(rotation=0)  # Rotaciona os rótulos das categorias para ficarem legíveis
        plt.show()

    def mean(self):
        raise NotImplementedError("TODO")

    def var(self):
        raise NotImplementedError("TODO")

    def mode(self):
        return self.data.mode().iloc[0]

    @staticmethod
    def _make_table(x: list | pd.Series, sort: bool = False, decreasing: bool = False) -> pd.DataFrame:
        """
        Creates a frequency distribution table (FDT) for categorical data.

        Parameters:
        x (list or pd.Series): The input data, which must be a list or pandas Series.
        sort (bool): If True, sorts the table by frequency. Default is False.
        decreasing (bool): If sort is True, sorts in descending order if True, otherwise in ascending order. Default is False.

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

    def generate_fdt(self, df, column_name):
        # FIXME: acho bom mover isso para outro lugar. Não tenho certeza
        """
        This function will generate the frequency distribution table for the specified column in the dataframe.

        Parameters:
        df (pd.DataFrame): The input dataframe.
        column_name (str): The column name to generate the FDT for.

        Returns:
        pd.DataFrame: A DataFrame with the frequency distribution table for the specified column.
        """
        # Check if the DataFrame has the specified column
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        # Ensure the column is categorical
        df[column_name] = df[column_name].astype("category")

        # Generate the FDT for the specified column
        fdt_result = self.make_fdt_cat_simple(df[column_name], sort=True, decreasing=True)
        return fdt_result
