from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Binning:
    """Class for storing information about binning for numerical FDTs."""

    start: float
    """The start of the bin."""

    end: float
    """The end of the bin."""

    h: float
    """The width/size of each bin."""

    k: int
    """The total amount of bins."""

    bins: np.ndarray
    """An array with the start of each bin."""

    @staticmethod
    def auto(
        start: Optional[float] = None,
        end: Optional[float] = None,
        h: Optional[float] = None,
        k: Optional[int] = None,
    ) -> Callable[[pd.Series], Binning]:
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
                raise ValueError("invalid arguments for auto")

        return lambda data: inner(data, start, end, h, k)

    @staticmethod
    def linspace(
        k: int,
        data: Optional[pd.Series] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> Binning:
        """Linear binning, dividing the entire range into `k` equal spaces."""
        start = start if start is not None else data.min() - abs(data.min()) / 100
        end = end if end is not None else data.max() + abs(data.max()) / 100
        h = (end - start) / k
        bins = np.arange(start, end + h, h)
        return Binning(k=k, start=start, end=end, h=h, bins=bins)

    @staticmethod
    def from_sturges(data: pd.Series) -> Binning:
        n = len(data)
        k = int(np.ceil(1 + 3.322 * np.log10(n)))
        return Binning.linspace(data=data, k=k)

    @staticmethod
    def from_scott(data: pd.Series) -> Binning:
        n = len(data)
        sd = np.std(data)
        at = data.max() - data.min()
        k = int(np.ceil(at / (3.5 * sd / (n ** (1 / 3)))))
        return Binning.linspace(data=data, k=k)

    @staticmethod
    def from_fd(data: pd.Series) -> Binning:
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
