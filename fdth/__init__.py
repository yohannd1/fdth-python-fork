"""
This module includes classes and functions for creating frequency distribution
tables (FDTs) and generating associated graphs.

There are three main classes for FDTs:

- `fdth.numerical_fdt.NumericalFDT`: used for numerical data;

- `fdth.categorical_fdt.CategoricalFDT`: used for categorical data;

- `fdth.multiple_fdt.MultipleFDT`: used for creating FDTs for all columns in a
matrix or dataframe.

The `fdth.auto_fdt.fdt` function can be used to automatically detect the type
of the data, or as a shorthand for creating FDTs of any data type.

```python
from fdth import fdt
import pandas as pd

data = [1, 5, 8, 3, 12]
fd = fdt(data)
print(fd.__class__) # NumericalFDT

data = ["foo", "bar", 5, "baz"]
fd = fdt(data)
print(fd.__class__) # CategoricalFDT

df = pd.DataFrame({
    "numerical": [1, 5, 3, 2],
    "categorical": ["foo", "bar", "baz", "abc"],
})
fd = fdt(df)
print(fd.__class__) # MultipleFDT
print(fd.get_fdt("numerical").__class__) # MultipleFDT
print(fd.get_fdt("categorical").__class__) # CategoricalFDT
```

See the `examples/` folder for more usage examples.
"""

from .frequency_distribution import FrequencyDistribution
from .numerical_fdt import NumericalFDT, NumericalBin
from .categorical_fdt import CategoricalFDT
from .multiple_fdt import MultipleFDT
from .auto_fdt import fdt
