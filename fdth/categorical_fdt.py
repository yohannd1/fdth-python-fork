from typing import Optional, Any
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CategoricalFDT:
    """Stores information about a categorical frequency distribution, and provides relevant operations."""

    table: pd.DataFrame
    """
    The inner frequency distribution table. Columns:
    - `Category`: the unique categories;
    - `f`: the absolute frequency of each category;
    - `rf`: the relative frequency of each category;
    - `rf(%)`: the relative frequency expressed as a percentage;
    - `cf`: the cumulative absolute frequency;
    - `cf(%)`: the cumulative relative frequency expressed as a percentage.
    """

    def __init__(
        self,
        data: Optional[pd.Series | list] = None,
        *,
        freqs: Optional[pd.Series | dict[Any, int]] = None,
        sort: bool = True,
        decreasing: bool = False,
    ) -> None:
        """
        Create the frequency distribution class.

        Note that either `data` or `freqs` must be specified.

        :param data: a data set of which the frequency must be analyzed
        :param freqs: a pandas.Series with the value at a specific index being the absolute frequency of said category, or a dict with the key being a category and the value being the frequency
        :param sort: whether to sort the categories
        :param decreasing: (when sort is True) whether to sort the categories in decreasing order
        """

        if data is not None:
            if freqs is not None:
                raise ValueError("`data` and `freqs` must not be both specified")

            data = pd.Series(data).astype("category")

            self.count, self.table = self._make_table_from_data(
                data, sort=sort, decreasing=decreasing
            )
        elif freqs is not None:
            if data is not None:
                raise ValueError("`data` and `freqs` must not be both specified")

            if isinstance(freqs, dict):
                freqs = pd.Series(freqs)
            elif isinstance(freqs, pd.Series):
                freqs = freqs
            else:
                raise TypeError("`freqs` must be dict | pandas.Series")

            self.count, self.table = self._make_table_from_frequencies(
                freqs, sort=sort, decreasing=decreasing
            )
        else:
            raise ValueError("one of `data` or `table` must be specified")

    def get_table(self) -> pd.DataFrame:
        return self.table

    @lru_cache
    def mfv(self) -> pd.Series:
        """Returns the most frequent values (modes) of the data set."""
        freqs = self.table["f"].to_numpy()
        positions = np.where(freqs == freqs.max())[0]
        return pd.Series(self.table["Category"][i] for i in positions)

    @staticmethod
    def _make_table_from_frequencies(
        freqs: pd.Series | dict[Any, int], sort: bool, decreasing: bool
    ) -> tuple[int, pd.DataFrame]:
        """Make a frequency distribution table from a series of frequencies."""

        if isinstance(freqs, dict):
            freqs = pd.Series(freqs)

        if sort:
            # Sort by absolute frequencies
            freqs = freqs.sort_values(ascending=not decreasing)

        count = freqs.sum()

        # Calculate relative frequencies and cumulative frequencies
        rf = freqs / count  # Relative frequency
        rfp = rf * 100  # Relative frequency as a percentage
        cf = freqs.cumsum()  # Cumulative absolute frequency
        cfp = rfp.cumsum()  # Cumulative relative frequency as a percentage

        return count, pd.DataFrame({
            "Category": freqs.index,
            "f": freqs.values,
            "rf": rf.values,
            "rf(%)": rfp.values,
            "cf": cf.values,
            "cf(%)": cfp.values,
        }) # fmt: skip

    @staticmethod
    def _make_table_from_data(
        data: pd.Series, sort: bool, decreasing: bool
    ) -> tuple[int, pd.DataFrame]:
        """Create a frequency distribution table (FDT) for a set of categorical data."""

        # Convert data set to categorical type
        data = data.astype("category")

        # Check if there are valid categories
        if len(data.cat.categories) == 0:
            raise ValueError("No valid categories found in the data.")

        # Calculate absolute frequencies
        freqs = data.value_counts(sort=False)

        return CategoricalFDT._make_table_from_frequencies(
            freqs=freqs, sort=sort, decreasing=decreasing
        )

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
            table = pd.concat([table["Category"], table[columns]], axis="columns")

        # round the numbers in the table
        table = table.round(round)

        return table.to_string(index=row_numbers, justify="right" if right else "left")

    def __repr__(self) -> str:
        cat_count = self.table.shape[0]
        res = f"CategoricalFDT ({self.count} elements, {cat_count} categories):\n"
        res += self.to_string()
        return res

    def plot(
        self,
        plot_type: str = "fb",
        v: bool = False,
        v_round: int = 2,
        v_pos: int = 3,
        xlab: Optional[str] = None,
        xlas: int = 0,
        ylab: Optional[str] = None,
        y2lab: Optional[str] = None,
        y2cfp=np.arange(0, 101, 25),
        col: str = "0.4",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        main: Optional[str] = None,
        box: bool = False,
    ) -> None:
        """
        Make one of a series of FDT plots.

        Supported plots:
        - `fb`: bar plot with frequencies;
        - `fp`: polygon plot with frequencies;
        - `fd`: dot chart with frequencies;
        - `pa`: Pareto plot with cumulative frequencies;
        - `rfb`: bar plot with relative frequencies;
        - `rfp`: polygon plot with relative frequencies;
        - `rfd`: dot chart with relative frequencies;
        - `rfpb`: bar plot with relative frequencies in %;
        - `rfpp`: polygon plot with relative frequencies in %;
        - `rfpd`: dot chart with relative frequencies in %;
        - `cfb`: bar plot with cumulative frequencies;
        - `cfp`: polygon plot with cumulative frequencies;
        - `cfd`: dot chart with cumulative frequencies;
        - `cfpb`: bar plot with cumulative frequencies in %;
        - `cfpp`: polygon plot with cumulative frequencies in %;
        - `cfpd`: dot chart with cumulative frequencies in %.

        :param plot_type: type of plot to generate.
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
        x = self.get_table()

        # Helper function for bar plot
        def plot_b(y, categories):
            fig, ax = plt.subplots()
            bar_positions = np.arange(len(categories))

            ax.bar(bar_positions, y, color=col, edgecolor="black")
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(categories, rotation=xlas * 90)

            if xlab:
                ax.set_xlabel(xlab)
            if ylab:
                ax.set_ylabel(ylab)
            if main:
                ax.set_title(main)
            if box:
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)

            if v:
                for i, val in enumerate(y):
                    ax.text(i, val, f"{round(val, v_round)}", ha="center", va="bottom")

            plt.show()

        # Helper function for polygon plot
        def plot_p(y, categories):
            fig, ax = plt.subplots()
            ax.plot(range(len(categories)), y, "o-", color=col, markersize=5)

            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=xlas * 90)

            if xlab:
                ax.set_xlabel(xlab)
            if ylab:
                ax.set_ylabel(ylab)
            if main:
                ax.set_title(main)

            if v:
                for i, val in enumerate(y):
                    ax.text(i, val, f"{round(val, v_round)}", ha="center", va="bottom")

            plt.show()

        # Helper function for dotchart
        def plot_d(y, categories):
            fig, ax = plt.subplots()

            ax.plot(y, range(len(categories)), "o", color=col)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories)

            if xlab:
                ax.set_xlabel(xlab)
            if ylab:
                ax.set_ylabel(ylab)
            if main:
                ax.set_title(main)

            if v:
                for i, val in enumerate(y):
                    ax.text(val, i, f"{round(val, v_round)}", ha="right")

            plt.show()

        # Helper function for pareto plot
        def plot_pa(y, cf, cfp, categories):
            fig, ax1 = plt.subplots()

            bar_positions = np.arange(len(categories))

            # Bar plot
            ax1.bar(bar_positions, y, color=col, edgecolor="black")
            ax1.set_xticks(bar_positions)
            ax1.set_xticklabels(categories, rotation=xlas * 90)

            if xlab:
                ax1.set_xlabel(xlab)
            if ylab:
                ax1.set_ylabel(ylab)
            if main:
                ax1.set_title(main)

            # Set y-axis limit based on cumulative frequency
            ax1.set_ylim(0, max(cf) * 1.1)

            # Cumulative frequency plot
            ax2 = ax1.twinx()
            ax2.plot(
                bar_positions, cf, color="blue", marker="o", linestyle="-", markersize=5
            )
            ax2.set_ylabel(y2lab)
            ax2.set_ylim(
                0, max(cf) * 1.1
            )  # Ensure y-axis limit for cumulative frequency

            plt.show()

        # Call appropriate plot type based on `plot_type` argument
        categories = x["Category"]
        if plot_type == "fb":
            y = x.iloc[:, 1]
            xlab = xlab or "Category"
            ylab = ylab or "Frequency"
            ylim = ylim or (0, max(y) * 1.3)
            plot_b(y, categories)

        elif plot_type == "fp":
            y = x.iloc[:, 1]
            xlab = xlab or "Category"
            ylab = ylab or "Frequency"
            ylim = ylim or (0, max(y) * 1.2)
            plot_p(y, categories)

        elif plot_type == "fd":
            y = x.iloc[:, 1]
            xlab = xlab or "Frequency"
            plot_d(y, categories)

        elif plot_type == "pa":
            y = x.iloc[:, 1]
            cf = x.iloc[:, 4]  # Cumulative frequency
            cfp = x.iloc[:, 5]  # Cumulative frequency percentage
            xlab = xlab or "Category"
            ylab = ylab or "Frequency"
            y2lab = y2lab or "Cumulative frequency, (%)"
            ylim = ylim or (0, sum(y) * 1.1)
            plot_pa(y, cf, cfp, categories)

        elif plot_type == "rfb":
            y = x.iloc[:, 2]
            xlab = xlab or "Category"
            ylab = ylab or "Relative frequency"
            plot_b(y, categories)

        elif plot_type == "rfp":
            y = x.iloc[:, 2]
            xlab = xlab or "Category"
            ylab = ylab or "Relative frequency"
            ylim = ylim or (0, max(y) * 1.2)
            plot_p(y, categories)

        elif plot_type == "rfd":
            y = x.iloc[:, 2]
            xlab = xlab or "Relative frequency"
            plot_d(y, categories)

        elif plot_type == "rfpb":
            y = x.iloc[:, 3]
            xlab = xlab or "Category"
            ylab = ylab or "Relative frequency (%)"
            plot_b(y, categories)

        elif plot_type == "rfpp":
            y = x.iloc[:, 3]
            xlab = xlab or "Category"
            ylab = ylab or "Relative frequency (%)"
            ylim = ylim or (0, max(y) * 1.2)
            plot_p(y, categories)

        elif plot_type == "rfpd":
            y = x.iloc[:, 3]
            xlab = xlab or "Relative frequency (%)"
            plot_d(y, categories)

        elif plot_type == "cfb":
            y = x.iloc[:, 4]
            xlab = xlab or "Category"
            ylab = ylab or "Cumulative frequency"
            plot_b(y, categories)

        elif plot_type == "cfp":
            y = x.iloc[:, 4]
            xlab = xlab or "Category"
            ylab = ylab or "Cumulative frequency"
            ylim = ylim or (0, max(y) * 1.2)
            plot_p(y, categories)

        elif plot_type == "cfd":
            y = x.iloc[:, 4]
            xlab = xlab or "Cumulative frequency"
            plot_d(y, categories)

        elif plot_type == "cfpb":
            y = x.iloc[:, 5]
            xlab = xlab or "Category"
            ylab = ylab or "Cumulative frequency (%)"
            plot_b(y, categories)

        elif plot_type == "cfpp":
            y = x.iloc[:, 5]
            xlab = xlab or "Category"
            ylab = ylab or "Cumulative frequency (%)"
            ylim = ylim or (0, max(y) * 1.2)
            plot_p(y, categories)

        elif plot_type == "cfpd":
            y = x.iloc[:, 5]
            xlab = xlab or "Cumulative frequency (%)"
            plot_d(y, categories)
