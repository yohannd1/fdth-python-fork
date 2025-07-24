from typing import Optional, Literal, Sequence, Any, Callable
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fdth import FrequencyDistribution

BinMode = Literal["Sturges", "Scott", "FD"]


@dataclass(frozen=True)
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
        """
        Create a simple frequency distribution table.

        :param data: the data array.
        :param freqs: the data array.
        :param start: the starting point of the distribution range.
        :param end: the endpoint of the distribution range.
        :param h: the class interval width.
        :param right: whether to include the right endpoint in each interval.

        :return a frequency distribution table with class limits, frequencies, relative frequencies, cumulative frequencies, and cumulative percentages.
        """

        if data is not None and freqs is not None:
            raise ValueError("exactly one of `data` or `table` must be specified")
        elif data is not None:
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

    @lru_cache
    def _breaks(self) -> np.ndarray:
        # FIXME: can we replace this with "bins"?
        start = self.breaks_info.start
        end = self.breaks_info.end
        h = self.breaks_info.h
        return np.arange(start, end + h, h)

    @lru_cache
    def _midpoints(self) -> np.ndarray:
        """Calculate the midpoints of the class intervals."""
        breaks = self._breaks()
        return 0.5 * (breaks[:-1] + breaks[1:])

    @lru_cache
    def mean(self) -> float:
        """Calculate an approximate of the mean of the data represented by the FDT."""
        return np.sum(self.table["f"] * self._midpoints()) / self.count

    @lru_cache
    def at(self) -> float:
        """Calculate the total amplitude of the data (estimate)."""
        h = self.breaks_info.h
        return self.breaks_info.end - self.breaks_info.start

    @staticmethod
    def fmt_percentile(x: float) -> str:
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
        fmt_fn = fmt_fn if fmt_fn is not None else self.fmt_percentile
        return pd.Series({fmt_fn(b): self.quantile(b) for b in bins})

    def quantile(self, pos: float) -> float:
        """
        Calculate an approximate of a quantile of the data represented by the FDT.

        :param pos: position of the quantile - must be between 0 and 1.
        """
        if not (0.0 <= pos <= 1.0):
            raise ValueError(
                f"quantile position {pos} out of range - must be in [0, 1]"
            )

        # calculate "position" where the desired class will be
        pos_count = self.count * pos

        # get quantile index
        idx = np.where(pos_count <= self.table["cf"])[0][0]

        breaks = self._breaks()
        h = self.breaks_info.h

        # quantile class lower limit
        ll = breaks[idx]

        # cumulative frequency of the previous class
        cf_prev = 0 if idx < 1 else self.table.iloc[idx - 1, 4]

        # frequency of the quantile class
        f_q = self.table.iloc[idx, 1]

        return ll + ((pos_count - cf_prev) * h) / f_q

    @lru_cache
    def median(self) -> float:
        """Calculate an approximate of the median (50th percentile) of the data represented by the FDT."""
        return self.quantile(0.5)

    @lru_cache
    def var(self) -> float:
        """Calculate an approximate of the variance of the data represented by the FDT."""
        return np.sum((self._midpoints() - self.mean()) ** 2 * self.table["f"]) / (
            self.count - 1
        )

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

    # TODO: `plot()` method

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
        data: pd.Series,
        k: int | None = None,
        start: float | None = None,
        end: float | None = None,
        h: float | None = None,
        breaks: BinMode = "Sturges",
        right: bool = False,
        na_rm: bool = False,
    ) -> tuple[pd.DataFrame, BreaksInfo]:
        data = np.array([np.nan if v is None else v for v in data], dtype=np.float64)

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("The data vector must be numeric.")

        if na_rm:
            data = data[~np.isnan(data)]
        elif np.any(np.isnan(data)):
            raise ValueError("The data has <NA> values and na.rm=FALSE by default.")

        n = len(data)

        # calculate bins based on the specified method
        if k is None and start is None and end is None and h is None:
            if breaks == "Sturges":
                k = int(np.ceil(1 + 3.322 * np.log10(n)))
            elif breaks == "Scott":
                std_dev = np.std(data)
                k = int(
                    np.ceil(
                        (data.max() - data.min()) / (3.5 * std_dev / (n ** (1 / 3)))
                    )
                )
            elif breaks == "FD":
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                k = int(np.ceil((data.max() - data.min()) / (2 * iqr / (n ** (1 / 3)))))
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

        # generate the frequency distribution table
        table, bins = NumericalFDT._make_table_from_data(
            data=data,
            start=start,
            end=end,
            h=h,
            right=right,
            class_round=2,  # FIXME: receive this as a parameter
        )

        breaks_info = BreaksInfo(
            start=start,
            end=end,
            h=h,
            k=k,
            right=int(right),
            bins=bins,
        )

        return (table, breaks_info)

    @staticmethod
    def _make_table_from_data(
        data: pd.Series,
        start: float,
        end: float,
        h: float,
        right: bool,
        class_round: Optional[int],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        bins = np.arange(start, end + h, h)
        freqs = pd.cut(data, bins=bins, right=right).value_counts()
        return NumericalFDT._make_table_from_frequencies(
            freqs=freqs, start=start, end=end, h=h, right=right, class_round=class_round
        )

    @staticmethod
    def _make_table_from_frequencies(
        freqs: pd.Series,
        start: float,
        end: float,
        h: float,
        right: bool,
        class_round: Optional[int],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        bins = np.arange(start, end + h, h)

        r = class_round if class_round is not None else 2

        classes = [
            NumericalFDT._format_class(a, b, round_=r, right=right)
            for (a, b) in zip(bins[:-1], bins[1:])
        ]

        n = freqs.sum()
        rf = freqs / n
        rfp = rf * 100
        cf = freqs.cumsum()
        cfp = cf / n * 100

        table = pd.DataFrame({
            "Class limits": classes,
            "f": freqs.values,
            "rf": rf.values,
            "rf(%)": rfp.values,
            "cf": cf.values,
            "cf(%)": cfp.values,
        }) # fmt: skip

        table.index = np.arange(1, len(table) + 1)

        return (table, bins)

    @staticmethod
    def _format_class(a, b, round_: int, right: bool) -> str:
        ra = round(a, round_)
        rb = round(b, round_)
        if not right:
            return f"[{ra}, {rb})"
        else:
            return f"({ra}, {rb}]"
