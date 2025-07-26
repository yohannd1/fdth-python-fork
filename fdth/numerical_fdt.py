from __future__ import annotations

from typing import Optional, Sequence, Any, Callable, cast, Iterable, Sequence
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .binning import Binning

BinFunc = Callable[["pd.Series[float]"], Binning]
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
    def _cleanup_data(
        data: pd.Series[float] | Sequence[float], remove_nan: bool
    ) -> pd.Series[float]:
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

    def quantile(
        self,
        pos: int | float | Iterable[int | float] | NDArray,
        by: int | float | Sequence[float] | NDArray = 1.0,
    ) -> float | list[float]:
        """
        Calculate approximates of one or more quantiles of the data represented by the FDT.

        :param pos: position(s) of the quantile(s) - if `by` is a set of data, it should be int.
        :param by: the divisor for the quantile, or an array of possible quantile positions (with each value in [0, 1]).
        """

        def single_quantile(a, b) -> float:
            if a < 0.0:
                raise ValueError(f"quantile position should be positive - got {a}")
            if a > b:
                raise ValueError(f"quantile position is too big - should be smaller than `by` ({a} > {b})")

            # calculate "position" where the desired class will be
            pos_count = self.count * (a / b)

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

        if isinstance(by, int | float):
            if isinstance(pos, int | float):
                return single_quantile(pos, by)
            else:
                return [single_quantile(x, by) for x in pos]
        else:
            if isinstance(pos, int):
                return single_quantile(by[pos], 1)
            else:
                return [single_quantile(by[i], 1) for i in pos]

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

        freqs = self.table["f"].to_numpy()
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

    def plot(
        self,
        type_: str = "fh",
        v: bool = False,
        v_round: int = 2,
        v_pos: int = 3,
        xlab: str = "Class limits",
        xlas: int = 0,
        ylab: Optional[str] = None,
        color: str = "gray",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        main: Optional[str] = None,
        edgecolor: str = "black",
        linewidth: int = 1,
        x_round: int = 2,
        show: bool = True,
        **kwargs,
    ) -> None:
        """
        Make one of a collection of plots.

        Supported plots:
        - `fh`: absolute frequency histogram;
        - `fp`: absolute frequency polygon;
        - `rfh`: relative frequency histogram;
        - `rfp`: relative frequency polygon;
        - `rfph`: relative frequency (%) histogram;
        - `rfpp`: relative frequency (%) polygon;
        - `d`: density;
        - `cdh`: cumulative density histogram;
        - `cdp`: cumulative density polygon;
        - `cfh`: cumulative frequency histogram;
        - `cfp`: cumulative frequency polygon;
        - `cfph`: cumulative frequency (%) histogram;
        - `cfpp`: cumulative frequency (%) polygon.

        :param kwargs: forwarded to the various plot functions;
        :param type_: type of plot to generate.
        :param v: if True, display values on the plot.
        :param v_round: decimal places for values displayed on the plot.
        :param v_pos: vertical position for value labels.
        :param xlab: label for the x-axis.
        :param xlas: rotation angle for x-axis labels. Defaults to 0.
        :param ylab: label for the y-axis.
        :param y2lab: label for the secondary y-axis (used in Pareto plot).
        :param y2cfp: percentage ticks for cumulative frequency y-axis in Pareto plot.
        :param col: color for plot elements. Default is '0.4' (gray).
        """

        bins = self.binning.bins
        mids = self._midpoints()

        if xlim is None:
            xlim = (self.binning.start, self.binning.end)

        def make_range_labels(ax, ylim_):
            ybot, ytop = ylim_
            yrange = ytop - ybot
            yticks = np.arange(ybot, ybot + 1.1 * yrange, 0.1)
            ylabels = [f"{k*100:.0f}%" for k in yticks]
            ax.set_yticks(yticks, ylabels)

        def aux_barplot(
            y,
            percent=False,
            default_ylab="Frequency",
        ) -> None:
            ylim_ = ylim or (0, y.max())
            fig, ax = plt.subplots()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim_)
            ax.set_title(main)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab or default_ylab)
            ax.set_xticks(bins)

            if percent:
                make_range_labels(ax, ylim_)

            ax.bar(
                x=mids,
                height=y,
                width=self.binning.h,
                edgecolor=edgecolor,
                linewidth=1,
                color=color,
                **kwargs,
            )

            if v:
                for xpos, ypos in zip(mids, y):
                    ax.text(
                        xpos,
                        ypos,
                        f"{ypos:.{v_round}f}",
                        va="bottom",
                        ha="center",
                        **kwargs,
                    )

            if show:
                plt.show()

        def aux_polyplot(
            y,
            percent=False,
            default_ylab="Frequency",
        ) -> None:
            ylim_ = ylim or (-0.1, y.max() + 0.1)
            fig, ax = plt.subplots()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim_)
            ax.set_title(main)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab or default_ylab)
            ax.set_xticks(bins)

            if percent:
                make_range_labels(ax, ylim_)

            ax.plot(mids, y, "o-", color=color, **kwargs)

            if v:
                for xpos, ypos in zip(mids, y):
                    ax.text(
                        xpos,
                        ypos,
                        f"{ypos:.{v_round}f}",
                        va="bottom",
                        ha="center",
                        **kwargs,
                    )

            if show:
                plt.show()

        if type_ == "fh":
            y = self.table["f"].to_numpy()
            aux_barplot(y)
        elif type_ == "fp":
            y = self.table["f"].to_numpy()
            aux_polyplot(y)
        elif type_ == "rfh":
            y = self.table["rf"].to_numpy()
            aux_barplot(y)
        elif type_ == "rfp":
            y = self.table["rf"].to_numpy()
            aux_polyplot(y)
        elif type_ == "rfph":
            y = self.table["rf"].to_numpy()
            aux_barplot(y, percent=True)
        elif type_ == "rfpp":
            y = self.table["rf"].to_numpy()
            aux_polyplot(y, percent=True)
        elif type_ == "d":
            y = self.table["rf"].to_numpy() / self.binning.h
            aux_barplot(y, default_ylab="Density")
        elif type_ == "cdh":
            y = self.table["cf"].to_numpy() / (self.count * self.binning.h)
            aux_barplot(y, default_ylab="Cumulative density")
        elif type_ == "cdp":
            y = self.table["cf"].to_numpy() / (self.count * self.binning.h)
            aux_polyplot(y, default_ylab="Cumulative density")
        elif type_ == "cfh":
            y = self.table["cf"].to_numpy()
            aux_barplot(y)
        elif type_ == "cfp":
            y = self.table["cf"].to_numpy()
            aux_polyplot(y)
        elif type_ == "cfph":
            y = self.table["cf"].to_numpy() / self.count
            aux_barplot(y, percent=True)
        elif type_ == "cfpp":
            y = self.table["cf"].to_numpy() / self.count
            aux_polyplot(y, percent=True)
        else:
            raise ValueError(f"unknown plot type {repr(type_)}")

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
            table = table[columns]

        # round the numbers in the table
        table = table.round(round)

        return table.to_string(index=row_numbers, justify="right" if right else "left")

    def __repr__(self) -> str:
        res = f"NumericalFDT ({self.count} elements, {self.binning.k} classes, class amplitude of {self.binning.h:.2f}):\n"
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
            data.to_numpy(),
            bins=cast(Sequence[float], binning.bins),
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

        return table
