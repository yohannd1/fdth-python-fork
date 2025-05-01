from abc import abstractmethod

import pandas as pd
import numpy as np

class FrequencyDistribution:
    @abstractmethod
    def get_table(self) -> pd.DataFrame: ...

    @abstractmethod
    def mean(self): ...

    @abstractmethod
    def mode(self):
        """
        Calcula a moda (valor mais frequente) dos dados.
        """
        pass

    @abstractmethod
    def median(self): ...

    @abstractmethod
    def var(self): ...

    @abstractmethod
    def plot_histogram(self) -> None: ...
