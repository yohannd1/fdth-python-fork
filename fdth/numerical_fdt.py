from typing import Literal, Sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdth import FrequencyDistribution

NumericalBin = Literal["Sturges", "Scott", "FD"]


class NumericalFDT(FrequencyDistribution):
    def __init__(
        self,
        data: pd.Series | list | np.ndarray,
        k: int | None = None,
        start: float | None = None,
        end: float | None = None,
        h: float | None = None,
        breaks: NumericalBin = "Sturges",
        right: bool = False,
        na_rm: bool = False,
    ):
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        elif isinstance(data, pd.Series):
            pass
        else:
            raise TypeError("Data must be a list, a pandas.Series or an numpy.ndarray")

        result = _fdt_numeric_simple(
            data,
            k=k,
            start=start,
            end=end,
            h=h,
            breaks=breaks,
            right=right,
            na_rm=na_rm,
        )
        self._fdt = result["table"]

        self.breaks_info = result["breaks"]
        """Information about the binning done in the creation of the FDT."""

    def get_table(self) -> pd.DataFrame:
        return self._fdt

    def mean(self):
        raise NotImplementedError("TODO")

    def median(self):
        raise NotImplementedError("TODO")

    def mode(self):
        raise NotImplementedError("TODO")

    def var(self):
        raise NotImplementedError("TODO")

    def plot_histogram(self) -> None:
        plt.hist(self.data, bins=self.bins, edgecolor="black")
        plt.title("Histograma")
        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.show()


def _fdt_numeric_simple(x, k, start, end, h, breaks, right, na_rm):
    x = np.array([np.nan if v is None else v for v in x], dtype=np.float64)

    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("The data vector must be numeric.")

    if na_rm:
        x = x[~np.isnan(x)]
    elif np.any(np.isnan(x)):
        raise ValueError("The data has <NA> values and na.rm=FALSE by default.")

    # Bin calculation based on specified method
    if k is None and start is None and end is None and h is None:
        if breaks == "Sturges":
            k = int(np.ceil(1 + 3.322 * np.log10(len(x))))
        elif breaks == "Scott":
            std_dev = np.std(x)
            k = int(
                np.ceil((x.max() - x.min()) / (3.5 * std_dev / (len(x) ** (1 / 3))))
            )
        elif breaks == "FD":
            iqr = np.percentile(x, 75) - np.percentile(x, 25)
            k = int(np.ceil((x.max() - x.min()) / (2 * iqr / (len(x) ** (1 / 3)))))
        else:
            raise ValueError("Invalid 'breaks' method.")

        start, end = x.min() - abs(x.min()) / 100, x.max() + abs(x.max()) / 100
        R = end - start
        h = R / k

    elif start is None and end is None and h is None:
        start, end = x.min() - abs(x.min()) / 100, x.max() + abs(x.max()) / 100
        R = end - start
        h = R / k

    elif k is None and h is None:
        R = end - start
        k = int(np.sqrt(abs(R)))
        k = max(k, 5)
        h = R / k

    elif k is None:
        pass

    else:
        raise ValueError("Please check the function syntax!")

    # Generate the frequency distribution table
    table = _make_fdt_simple(x, start, end, h, right)
    breaks_info = {"start": start, "end": end, "h": h, "right": int(right)}
    result = {"table": table, "breaks": breaks_info}

    return result


def _make_fdt_simple(
    x: Sequence[float], start: float, end: float, h: float, right: bool = False
) -> pd.DataFrame:
    """
    Create a simple frequency distribution table.

    Parameters:
    x (array-like): The data array.
    start (float): The starting point of the distribution range.
    end (float): The endpoint of the distribution range.
    h (float): The class interval width.
    right (bool): Whether to include the right endpoint in each interval.

    Returns:
    DataFrame: A frequency distribution table with class limits, frequencies, relative frequencies,
               cumulative frequencies, and cumulative percentages.
    """
    bins = np.arange(start, end + h, h)
    labels = [
        f"[{round(bins[i], 2)}, {round(bins[i + 1], 2)})" for i in range(len(bins) - 1)
    ]
    f = pd.cut(x, bins=bins, right=right, labels=labels).value_counts()
    rf = f / len(x)
    rfp = rf * 100
    cf = f.cumsum()
    cfp = (cf / len(x)) * 100

    table = pd.DataFrame({
        "Class limits": labels,
        "f": f.values,
        "rf": rf.values,
        "rf(%)": rfp.values,
        "cf": cf.values,
        "cf(%)": cfp.values,
    }) # fmt: skip

    table.index = np.arange(1, len(table) + 1)

    return table


## TIPO I ##

# class NumericalFDT(FrequencyDistribution):
#     def __init__(
#         self,
#         data: pd.Series | list | np.ndarray,
#         k: int | None = None,
#         start: float | None = None,
#         end: float | None = None,
#         h: float | None = None,
#         breaks: NumericalBin = "Sturges",
#         right: bool = False,
#         na_rm: bool = False,
#     ):
#         if isinstance(data, (list, np.ndarray)):
#             data = pd.Series(data)
#         elif isinstance(data, pd.Series):
#             pass
#         else:
#             raise TypeError("Data must be a list, a pandas.Series or an numpy.ndarray")

#         data = data.dropna()
#         self._data = pd.to_numeric(data, errors="coerce").dropna()

#         self.n = len(self.data)
#         self.k = int(
#             np.ceil(1 + 3.322 * np.log10(self.n))
#         )  # Regra de Sturges para determinar os intervalos
#         self.h = (self.data.max() - self.data.min()) / self.k
#         self.bins = np.arange(self.data.min(), self.data.max() + self.h, self.h)

#         self.freq, _ = np.histogram(self.data, bins=self.bins)
#         self.midpoints = 0.5 * (self.bins[:-1] + self.bins[1:])
#         self.cum_freq = np.cumsum(self.freq)

#         # Tabela de frequência
#         self.fdt = pd.DataFrame(
#             {
#                 "Intervalo": list(
#                     zip(np.round(self.bins[:-1], 2), np.round(self.bins[1:], 2))
#                 ),
#                 "Frequência": self.freq,
#                 "Frequência Acumulada": self.cum_freq,
#                 "Ponto Médio": np.round(self.midpoints, 2),
#             }
#         )

#     def get_table(self) -> pd.DataFrame:
#         return self.fdt

#     def mean(self):
#         return np.sum(self.midpoints * self.freq) / self.n

#     def median(self):
#         n_2 = self.n / 2
#         idx = np.where(self.cum_freq >= n_2)[0][0]

#         Li = self.bins[idx]  # Limite inferior da classe mediana
#         Fi = self.freq[idx]  # Frequência da classe mediana
#         F_acum_antes = self.cum_freq[idx - 1] if idx > 0 else 0

#         median = Li + ((n_2 - F_acum_antes) * self.h) / Fi
#         return median

#     def mode(self):
#         pass

#     def var(self):
#         return np.sum(((self.midpoints - self.mean()) ** 2) * self.freq) / (self.n - 1)

#     def plot_histogram(self) -> None:
#         plt.hist(self.data, bins=self.bins, edgecolor="black")
#         plt.title("Histograma")
#         plt.xlabel("Valor")
#         plt.ylabel("Frequência")
#         plt.show()


#### LIMITES NÃO APROXIMADOS ####

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# class NumericalFDT:
#     def __init__(self, data):
#         """
#         Inicializa a classe para dados numéricos.
#         """
#         if not isinstance(data, (pd.Series, list)):
#             raise ValueError("Dados devem ser uma lista ou uma Series do pandas.")

#         self.data = pd.Series(data) if isinstance(data, list) else data
#         self.data = pd.to_numeric(self.data, errors="coerce")  # Convertendo para numérico

#         self.breaks = self._define_breaks()
#         self.table = self._build_fdt_table()

#     def _define_breaks(self):
#         """
#         Define os limites dos intervalos automaticamente com base nos dados.
#         """
#         bins = np.histogram_bin_edges(self.data, bins="auto")
#         return {
#             "start": bins[0],
#             "end": bins[-1],
#             "h": bins[1] - bins[0],
#             "bins": bins
#         }

#     def _build_fdt_table(self):
#         """
#         Constrói a tabela de distribuição de frequência.
#         """
#         breaks = self.breaks["bins"]
#         freq, _ = np.histogram(self.data, bins=breaks)
#         cum_freq = np.cumsum(freq)
#         intervals = list(zip(breaks[:-1], breaks[1:]))

#         return pd.DataFrame({
#             "Interval": intervals,
#             "Frequency": freq,
#             "Cumulative Frequency": cum_freq
#         })

#     def make_fdt(self):
#         """
#         Retorna a tabela de distribuição de frequência.
#         """
#         return self.table

#     def mean_fdt(self):
#         """
#         Calcula a média da distribuição de frequência.
#         """
#         breaks = self.breaks["bins"]
#         mids = 0.5 * (breaks[:-1] + breaks[1:])
#         y = self.table["Frequency"].values
#         return np.sum(y * mids) / np.sum(y)

#     def var_fdt(self):
#         """
#         Calcula a variância da distribuição de frequência.
#         """
#         breaks = self.breaks["bins"]
#         mids = 0.5 * (breaks[:-1] + breaks[1:])
#         y = self.table["Frequency"].values
#         mean = self.mean_fdt()
#         return np.sum((mids - mean) ** 2 * y) / (np.sum(y) - 1)

#     def median_fdt(self):
#         """
#         Calcula a mediana da distribuição de frequência.
#         """
#         fdt = self.table
#         n = fdt["Cumulative Frequency"].iloc[-1]
#         posM = (n / 2 <= fdt["Cumulative Frequency"]).idxmax()

#         breaks = self.breaks["bins"]
#         liM = breaks[posM]  # limite inferior da classe mediana

#         if posM - 1 < 0:
#             sfaM = 0
#         else:
#             sfaM = fdt["Cumulative Frequency"].iloc[posM - 1]

#         fM = fdt["Frequency"].iloc[posM]
#         h = self.breaks["h"]

#         return liM + (((n / 2) - sfaM) * h) / fM

#     def plot_histogram(self):
#         """
#         Plota o histograma dos dados.
#         """
#         plt.hist(self.data, bins=self.breaks["bins"], edgecolor="black")
#         plt.title("Histograma de Dados Numéricos")
#         plt.xlabel("Valores")
#         plt.ylabel("Frequência")
#         plt.show()
