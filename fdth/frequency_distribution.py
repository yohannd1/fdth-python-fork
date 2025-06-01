from abc import abstractmethod, ABC

import pandas as pd
import numpy as np


class FrequencyDistribution(ABC):
    """Stores frequency distribution data for some dataset."""

    @abstractmethod
    def get_table(self) -> pd.DataFrame: ...

    @abstractmethod
    def plot_histogram(self) -> None: ...
