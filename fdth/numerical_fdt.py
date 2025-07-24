from __future__ import annotations

from typing import Optional, Sequence, Any, Callable
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .binning import Binning

BinFunc = Callable[[pd.Series[float]], Binning]
"""Type definition for a function that takes a dataset and returns a binning configuration for it."""


class NumericalFDT:
    """Stores information about a numerical frequency distribution, and provides relevant operations."""

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

    count: int
    """The amount of elements in the dataset."""

    binning: Binning
    """Information about the binning for this FDT."""

    def __init__(
        self,
        data: Optional[pd.Series[float] | Sequence[float]] = None,
        *,
        freqs: Optional[pd.Series[float] | Sequence[float]] = None,
        binning: Binning | BinFunc = Binning.from_sturges,
        right: bool = False,
        remove_nan: bool = False,
        round_: int = 2,
    ):
        """
        :param data: the data array;
        :param freqs: the frequency array - an alternative to the data array;
        :param binning: the binning, or a function that generates the binning based on the data;
        :param right: whether to include the right endpoint in each interval;
        :param round_: the rounding level for the numbers in the table;

        :return a frequency distribution table with class limits, frequencies, relative frequencies, cumulative frequencies, and cumulative percentages.
        """

        if data is not None and freqs is not None:
            raise ValueError("exactly one of `data` or `freqs` must be specified")
        elif data is not None:
            data = self._cleanup_data(data, remove_nan=remove_nan)
            self.count = len(data)

            b = binning(data) if callable(binning) else binning
            self.table = self._make_table_from_data(data, b, right, round_=round_)
            self.binning = b
        elif freqs is not None:
            if not isinstance(binning, Binning):
                raise ValueError(
                    "a ready-made binning must be specified when passing `freqs`"
                )

            freqs = pd.Series(freqs)
            self.count = int(freqs.sum())

            self.table = self._make_table_from_frequencies(
                freqs, binning, right, round_=round_
            )
            self.binning = binning

    @staticmethod
    def _cleanup_data(data: pd.Series[float] | Sequence[float], remove_nan: bool) -> pd.Series[float]:
        d = np.array([np.nan if v is None else v for v in data], dtype=np.float64)
        if not np.issubdtype(d.dtype, np.number):
            raise ValueError("input data must be numeric")

        if remove_nan:
            d = d[~np.isnan(d)]
        elif np.any(np.isnan(d)):
            raise ValueError("the data has NaN values")

        return pd.Series(d)

    @lru_cache
    def _midpoints(self) -> np.ndarray:
        """Calculate the midpoints of the class intervals."""
        bins = self.binning.bins
        return 0.5 * (bins[:-1] + bins[1:])

    @lru_cache
    def mean(self) -> float:
        """Calculate an approximate of the mean of the data represented by the FDT."""
        return np.sum(self.table["f"] * self._midpoints()) / self.count

    @lru_cache
    def at(self) -> float:
        """Calculate the total amplitude of the data (estimate)."""
        h = self.binning.h
        return self.binning.end - self.binning.start

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

        bins = self.binning.bins
        h = self.binning.h

        # quantile class lower limit
        ll = bins[idx]

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
        bins = self.binning.bins
        h = self.binning.h

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
        res = f"NumericalFDT ({self.count} elements, {self.binning.k} classes, amplitude of {self.binning.h:.2f}):\n"
        res += self.to_string()
        return res

    @staticmethod
    def _make_table_from_data(
        data: pd.Series,
        binning: Binning,
        right: bool,
        round_: int,
    ) -> pd.DataFrame:
        freqs = pd.cut(
            data.to_numpy(),  # XXX: converting it to numpy makes the order work. Why?
            bins=binning.bins,
            right=right,
        ).value_counts()

        return NumericalFDT._make_table_from_frequencies(
            freqs, binning, right=right, round_=round_
        )

    @staticmethod
    def _make_table_from_frequencies(
        freqs: pd.Series,
        binning: Binning,
        right: bool,
        round_: int,
    ) -> pd.DataFrame:
        bins = binning.bins
        r = round_ if round_ is not None else 2
        classes = binning.format_classes(round_=round_, right=right)

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

        table.index = np.arange(1, len(table) + 1)  # FIXME: do we need this?

        return table
