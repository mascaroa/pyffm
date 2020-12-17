```
        _________  _.
.__    |__ |__ | \/ |
|__\__/|   |   |    |
|   /       
```

A python implementation of Factorization Machines / Field-aware Factorization Machines with a simple interface.

Installation:
```shell script
pip install pyffm
``` 

Basic example:
```python
import pandas as pd
from pyffm import PyFFM
training_params = {'epochs': 2, 'reg_lambda': 0.002}
pyffm = PyFFM(model="ffm", training_params=training_params)

file_path = 'path/to/csv/file'
df_in = pd.read_csv(file_path)
# Make sure your file has a label column, default name is 'click' but you can either rename it or pass in label
df_in.rename(columns={'label': 'click'}, inplace=True)

pyffm.train(df_in)
preds = pyffm.predict(df_in)


```