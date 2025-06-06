from typing import Literal, Sequence
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdth import FrequencyDistribution

NumericalBin = Literal["Sturges", "Scott", "FD"]


@dataclass
class BreaksInfo:
    pass


class NumericalFDT(FrequencyDistribution):
    """Stores information about a numerical frequency distribution, and allows related operations."""

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
            self._data = pd.Series(data)
        elif isinstance(data, pd.Series):
            self._data = data
        else:
            raise TypeError("Data must be a list, a pandas.Series or an numpy.ndarray")

        self._data_size = len(self._data)

        result = _fdt_numeric_simple(
            self._data,
            k=k,
            start=start,
            end=end,
            h=h,
            breaks=breaks,
            right=right,
            na_rm=na_rm,
        )

        self.table = result["table"]
        """The inner frequency distribution table."""

        self.breaks_info = result["breaks"]
        """Information about the binning done in the creation of the FDT."""

    @lru_cache(maxsize=1)
    def mean(self) -> float:
        """Calculate an approximate of the mean of the data represented by the FDT."""

        start = self.breaks_info["start"]
        end = self.breaks_info["end"]
        h = self.breaks_info["h"]

        # define class interval
        breaks = np.arange(start, end + h, h)

        # calculate midpoints of the class intervals
        mids = 0.5 * (breaks[:-1] + breaks[1:])

        # frequencies of each class
        y = self.table.loc[:, "f"]

        # return the weighted mean of the midpoints
        return np.sum(y * mids) / np.sum(y)

    @lru_cache(maxsize=1)
    def median(self) -> float:
        """Calculate an approximate of the median (50th percentile) of the data represented by the FDT."""

        start = self.breaks_info["start"]
        end = self.breaks_info["end"]
        h = self.breaks_info["h"]

        # Número total de observações
        n = self.table.iloc[-1, 4]

        # Posição da classe mediana
        posM = (n / 2 <= self.table.iloc[:, 4]).idxmax()

        brk = np.arange(start, end + h, h)

        # Limite inferior da classe mediana
        liM = brk[posM]

        # Frequência acumulada anterior à classe mediana
        if posM - 1 < 0:
            sfaM = 0
        else:
            sfaM = self.table.iloc[posM - 1, 4]

        # Frequência da classe mediana
        fM = self.table.iloc[posM, 1]

        return liM + (((n / 2) - sfaM) * h) / fM

    @lru_cache(maxsize=1)
    def var(self) -> float:
        """Calculate an approximate of the variance of the data represented by the FDT."""

        start = self.breaks_info["start"]
        end = self.breaks_info["end"]
        h = self.breaks_info["h"]

        # Definir intervalos de classe com base nos valores 'start', 'end' e 'h'
        breaks = np.arange(start, end + h, h)

        # Calcular pontos médios dos intervalos de classe
        mids = 0.5 * (breaks[:-1] + breaks[1:])

        # Frequências das classes
        y = self.table.loc[:, "f"]

        return np.sum((mids - self.mean()) ** 2 * y) / (np.sum(y) - 1)

    @lru_cache(maxsize=1)
    def sd(self) -> float:
        """Calculates the standard deviation (square root of the variance)."""
        return np.sqrt(self.var())

    @lru_cache(maxsize=1)
    def mfv(self) -> pd.Series:
        """Returns the most frequent values (modes) of the data set."""
        return self._data.mode()

    def get_table(self):
        # FIXME: deprecate in favor of `self.table`
        return self.table

    def plot_histogram(self) -> None:
        # FIXME: whoops. I think I (yohanan) messed this up
        plt.hist(self.table, bins=self.breaks_info["bins"], edgecolor="black")
        plt.title("Histograma")
        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.show()

    def __repr__(self):
        res = f"NumericalFDT (size {self._data_size}, class count {self.breaks_info['k']}, amplitude {round(self.breaks_info['h'], 4)}), head:\n"
        res += self.table.head().to_string(index=False)
        return res

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
    table, bins = _make_fdt_simple(x, start, end, h, right)
    breaks_info = {
        "start": start,
        "end": end,
        "h": h,
        "k": k,
        "right": int(right),
        "bins": bins,
    }
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

    return table, bins
