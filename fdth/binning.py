from __future__ import annotations

from typing import Optional, Callable
from dataclasses import dataclass

import pandas as pd
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Binning:
    """
    Class for storing information about binning for numerical FDTs.

    Usually the desired result is to have a set of values following `h = (start - end) / k`, such that there are `k` class intervals and the start of the i-th class interval is at `start + h * (i - 1)`.
    """

    start: float
    """The start of the bin."""

    end: float
    """The end of the bin."""

    h: float
    """The amplitude/width of each bin/class."""

    k: int
    """The total amount of bins/classes."""

    bins: NDArray
    """An array with the start of each bin. Usually generated automatically."""

    @staticmethod
    def auto(
        start: Optional[float] = None,
        end: Optional[float] = None,
        h: Optional[float] = None,
        k: Optional[int] = None,
    ) -> Callable[[pd.Series], Binning]:
        """Build an automatic binning based on the passed start, end, h and k values."""

        all_none = lambda *x: all(xi is None for xi in x)
        no_none = lambda *x: all(xi is not None for xi in x)

        def inner(data, start, end, h, k) -> Binning:
            if all_none(start, end, h, k):
                raise ValueError("at least one of the arguments must be specified")
            elif h is None and k is not None:
                return Binning.linspace(data=data, k=k)
            elif no_none(start, end) and all_none(h, k):
                r = end - start
                k = int(np.sqrt(abs(r)))
                return Binning.linspace(k=max(k, 5), start=start, end=end)
            elif no_none(start, end, h) and k is None:
                # XXX: forcing h to potentially be a different value, as in to
                # maintain consistency, but if it is correct it won't be
                # changed
                k = np.ceil((end - start) / h)
                return Binning.linspace(k=k, start=start, end=end)
            else:
                raise ValueError("`h` and `k` must not be both specified")

        return lambda data: inner(data, start, end, h, k)

    @staticmethod
    def linspace(
        k: int,
        data: Optional[pd.Series] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Binning:
        """Linear binning, dividing the entire range into `k` equal spaces."""
        if start is None:
            if data is None:
                raise ValueError("`data` is None when `start` was not specified")
            start = data.min() - abs(data.min()) / 100
        if end is None:
            if data is None:
                raise ValueError("`data` is None when `end` was not specified")
            end = data.max() + abs(data.max()) / 100
        h = (end - start) / k
        bins = np.arange(start, end + h, h)
        return Binning(k=k, start=start, end=end, h=h, bins=bins)

    @staticmethod
    def from_sturges(data: pd.Series) -> Binning:
        """Sturges method for calculating the binning."""
        # FIXME: doesn't seem to be accurate anymore? do more testing here.
        n = len(data)
        k = int(np.ceil(1 + 3.322 * np.log10(n)))
        return Binning.linspace(data=data, k=k)

    @staticmethod
    def from_scott(data: pd.Series) -> Binning:
        """Scott method for calculating the binning."""
        n = len(data)
        sd = np.std(data)
        at = data.max() - data.min()
        k = int(np.ceil(at / (3.5 * sd / (n ** (1 / 3)))))
        return Binning.linspace(data=data, k=k)

    @staticmethod
    def from_fd(data: pd.Series) -> Binning:
        """Freedman-Diaconis method for calculating the binning."""
        n = len(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        k = int(np.ceil((data.max() - data.min()) / (2 * iqr / (n ** (1 / 3)))))
        return Binning.linspace(data=data, k=k)

    def format_classes(self, round_: int, right: bool) -> list[str]:
        delims = ("(", "]") if right else ("[", ")")
        bins = self.bins

        def fmt(a, b) -> str:
            ra = str(round(a, round_))
            rb = str(round(b, round_))
            return "{}{}, {}{}".format(delims[0], ra, rb, delims[1])

        return [fmt(a, b) for (a, b) in zip(bins[:-1], bins[1:])]
