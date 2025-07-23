from typing import Optional, Literal, Sequence, Any, Callable
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdth import FrequencyDistribution

BinMode = Literal["Sturges", "Scott", "FD"]


@dataclass
class BreaksInfo:
    start: float
    end: float
    h: float
    k: int
    right: bool
    bins: np.ndarray


class NumericalFDT(FrequencyDistribution):
    """Stores information about a numerical frequency distribution, and allows related operations."""

    table: pd.DataFrame
    """
    The inner frequency distribution table. Columns:
    - `Class limits`: the limits of the class;
    - `f`: the absolute frequency of each class;
    - `rf`: the relative frequency of each class;
    - `rf(%)`: the relative frequency expressed as a percentage;
    - `cf`: the cumulative absolute frequency;
    - `cf(%)`: the cumulative relative frequency expressed as a percentage.
    """

    breaks_info: BreaksInfo
    """Information about the binning done in the creation of the FDT."""

    def __init__(
        self,
        data: Optional[pd.Series | list | np.ndarray] = None,
        *,
        freqs: Optional[pd.Series | dict] = None,
        k: int | None = None,
        start: float | None = None,
        end: float | None = None,
        h: float | None = None,
        breaks: BinMode = "Sturges",
        right: bool = False,
        na_rm: bool = False,
    ):
        if data is not None:
            if freqs is not None:
                raise ValueError("`data` and `freqs` must not be both specified")

            if isinstance(data, (list, np.ndarray)):
                data = pd.Series(data)
            elif isinstance(data, pd.Series):
                pass
            else:
                raise TypeError("`data` must be list | pandas.Series | numpy.ndarray")

            self.count = len(data)
            self.table, self.breaks_info = self._fdt_numeric_simple(
                data,
                k=k,
                start=start,
                end=end,
                h=h,
                breaks=breaks,
                right=right,
                na_rm=na_rm,
            )
        elif freqs is not None:
            if data is not None:
                raise ValueError("`data` and `freqs` must not be both specified")

            if isinstance(freqs, dict):
                freqs = pd.Series(freqs)
            elif isinstance(freqs, pd.Series):
                pass
            else:
                raise TypeError("`data` must be dict | pandas.Series")

            raise NotImplementedError("TODO")
        else:
            raise ValueError("one of `data` or `table` must be specified")

    @lru_cache
    def mean(self) -> float:
        """Calculate an approximate of the mean of the data represented by the FDT."""

        start = self.breaks_info.start
        end = self.breaks_info.end
        h = self.breaks_info.h

        # define class interval
        breaks = np.arange(start, end + h, h)

        # calculate midpoints of the class intervals
        midpoints = 0.5 * (breaks[:-1] + breaks[1:])

        # frequencies of each class
        y = self.table.loc[:, "f"]

        # return the weighted mean of the midpoints
        return np.sum(y * midpoints) / np.sum(y)

    @lru_cache
    def at(self) -> float:
        """Calculate the total amplitude of the data (estimate)."""
        h = self.breaks_info.h
        return self.breaks_info.end - self.breaks_info.start

    @staticmethod
    def quantile_to_percentile(x: float) -> str:
        return f"{x * 100:.2f}%"

    def quantiles(
        self,
        bins: Sequence[float] = np.arange(0.0, 1.0, 0.1),
        fmt_fn: Callable[[float], str] | None = None,
    ) -> pd.Series:
        """
        Calculate an approximate of multiple quantiles of the data represented by the FDT.

        :param bins: array of values between 0 and 1
        :param fmt_fn: function that maps the quantile number to a representation in the result. Defaults to percentile formatting.
        """
        if fmt_fn is None:
            fmt_fn = self.quantile_to_percentile
        return pd.Series({fmt_fn(b): self.quantile(b) for b in bins})

    def quantile(self, pos: float) -> float:
        """
        Calculate an approximate of a quantile of the data represented by the FDT.

        :param pos: position of the quantile - must be between 0 and 1.
        """
        if not (0.0 <= pos <= 1.0):
            raise ValueError(f"quantile position {pos} out of range - must be in [0, 1]")

        start = self.breaks_info.start
        end = self.breaks_info.end
        h = self.breaks_info.h

        pos_count = self.count * pos

        # Posição da classe mediana
        pos_m = np.where(pos_count <= self.table.iloc[:, 4])[0][0]

        breaks = np.arange(start, end + h, h)

        # Limite inferior da classe mediana
        li_m = breaks[pos_m]

        # Frequência acumulada anterior à classe mediana
        sfa_m = self.table.iloc[pos_m - 1, 4] if pos_m >= 1 else 0

        # Frequência da classe mediana
        f_m = self.table.iloc[pos_m, 1]

        return li_m + ((pos_count - sfa_m) * h) / f_m

    @lru_cache
    def median(self) -> float:
        """Calculate an approximate of the median (50th percentile) of the data represented by the FDT."""
        return self.quantile(0.5)

    @lru_cache
    def var(self) -> float:
        """Calculate an approximate of the variance of the data represented by the FDT."""

        start = self.breaks_info.start
        end = self.breaks_info.end
        h = self.breaks_info.h

        # Definir intervalos de classe com base nos valores 'start', 'end' e 'h'
        breaks = np.arange(start, end + h, h)

        # Calcular pontos médios dos intervalos de classe
        midpoints = 0.5 * (breaks[:-1] + breaks[1:])

        # Frequências das classes
        y = self.table.loc[:, "f"]

        return np.sum((midpoints - self.mean()) ** 2 * y) / (np.sum(y) - 1)

    @lru_cache
    def sd(self) -> float:
        """Calculate the standard deviation (square root of the variance)."""
        return np.sqrt(self.var())

    @lru_cache
    def mfv(self) -> pd.Series:
        """Calculate an approximation of the most frequent values (modes) of the data set."""

        freqs = self.table["f"].values
        bins = self.breaks_info.bins
        h = self.breaks_info.h

        # Czuber's formula
        def calculate_mfv(pos: int) -> float:
            lower_limit = bins[pos]
            num_rows = len(freqs)

            current_freq = float(freqs[pos])
            preceding_freq = float(0 if pos - 1 < 0 else freqs[pos - 1])
            succeeding_freq = float(0 if pos + 1 >= num_rows else freqs[pos + 1])

            d1 = current_freq - preceding_freq
            d2 = current_freq - succeeding_freq

            return float(lower_limit + (d1 / (d1 + d2)) * h)

        positions = np.where(freqs == freqs.max())[0]
        return pd.Series(calculate_mfv(pos) for pos in positions)

    def get_table(self) -> pd.DataFrame:
        return self.table

    def plot_histogram(self) -> None:
        # FIXME: whoops. I think I (yohanan) messed this up
        plt.hist(self.table, bins=self.breaks_info.bins, edgecolor="black")
        plt.title("Histograma")
        plt.xlabel("Valor")
        plt.ylabel("Frequência")

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
            table = pd.concat([table["Class limits"], table[columns]], axis="columns")

        # round the numbers in the table
        table = table.round(round)

        return table.to_string(index=row_numbers, justify="right" if right else "left")

    def __repr__(self) -> str:
        res = f"NumericalFDT ({self.count} elements, {self.breaks_info.k} classes, amplitude of {self.breaks_info.h:.2f}):\n"
        res += self.to_string()
        return res

    @staticmethod
    def _fdt_numeric_simple(
        data, k, start, end, h, breaks: BinMode, right, na_rm
    ) -> tuple[pd.DataFrame, BreaksInfo]:
        data = np.array([np.nan if v is None else v for v in data], dtype=np.float64)

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("The data vector must be numeric.")

        if na_rm:
            data = data[~np.isnan(data)]
        elif np.any(np.isnan(data)):
            raise ValueError("The data has <NA> values and na.rm=FALSE by default.")

        # Bin calculation based on specified method
        if k is None and start is None and end is None and h is None:
            if breaks == "Sturges":
                k = int(np.ceil(1 + 3.322 * np.log10(len(data))))
            elif breaks == "Scott":
                std_dev = np.std(data)
                k = int(
                    np.ceil(
                        (data.max() - data.min())
                        / (3.5 * std_dev / (len(data) ** (1 / 3)))
                    )
                )
            elif breaks == "FD":
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                k = int(
                    np.ceil(
                        (data.max() - data.min()) / (2 * iqr / (len(data) ** (1 / 3)))
                    )
                )
            else:
                raise ValueError("Invalid 'breaks' method.")

            start, end = (
                data.min() - abs(data.min()) / 100,
                data.max() + abs(data.max()) / 100,
            )
            R = end - start
            h = R / k

        elif start is None and end is None and h is None:
            start, end = (
                data.min() - abs(data.min()) / 100,
                data.max() + abs(data.max()) / 100,
            )
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
        table, bins = NumericalFDT._make_fdt_simple(data, start, end, h, right)

        breaks_info = BreaksInfo(
            start=start,
            end=end,
            h=h,
            k=k,
            right=int(right),
            bins=bins,
        )

        return table, breaks_info

    @staticmethod
    def _make_fdt_simple(
        data: pd.Series, start: float, end: float, h: float, right: bool = False
    ) -> pd.DataFrame:
        """
        Create a simple frequency distribution table.

        :param data: The data array.
        :param start: The starting point of the distribution range.
        :param end: The endpoint of the distribution range.
        :param h: The class interval width.
        :param right: Whether to include the right endpoint in each interval.

        :return a frequency distribution table with class limits, frequencies, relative frequencies, cumulative frequencies, and cumulative percentages.
        """
        bins = np.arange(start, end + h, h)
        labels = [
            f"[{round(bins[i], 2)}, {round(bins[i + 1], 2)})"
            for i in range(len(bins) - 1)
        ]
        f = pd.cut(data, bins=bins, right=right, labels=labels).value_counts()
        rf = f / len(data)
        rfp = rf * 100
        cf = f.cumsum()
        cfp = (cf / len(data)) * 100

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
