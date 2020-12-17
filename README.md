```
        _________
.__    /    |  |_)
|__\__/\___ |  | \
|   /       
```

A python implementation of Factorization Machines / Field-aware Factorization Machines with a simple interface.

Installation:
```shell script
pip install py-ctr
``` 

Basic example:
```python
import pandas as pd
from pyctr import PyCTR
training_params = {'epochs': 2, 'reg_lambda': 0.002}
pyctr = PyCTR(model="ffm", training_params=training_params)

file_path = 'path/to/csv/file'
df_in = pd.read_csv(file_path, index_col=0)
df_in.rename(columns={'label': 'click'}, inplace=True)

pyctr.train(df_in)
preds = pyctr.predict(df_in)


```