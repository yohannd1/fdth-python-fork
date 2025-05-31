from typing import Literal, Optional

import pandas as pd

FdtKind = Literal["categorical", "numerical"]

def deduce_fdt_kind(data: pd.Series) -> FdtKind:
    is_categorical = data.dtype == "object" or isinstance(data.iloc[0], str)
    return "categorical" if is_categorical else "numerical"
