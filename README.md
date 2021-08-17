```
        _________  _.
.__    |__ |__ | \/ |
|__\__/|   |   |    |
|   /       
```

A python implementation of Factorization Machines / Field-aware Factorization Machines with a simple interface.

(FFM fully functioning/tested, FM still not fully tested)

Supports classification and regression.

#### Project Structure
```
pyffm
 ├── engine
 │   ├── model
 │   │   ├── __init__.py
 │   │   ├── base_model.py
 │   │   ├── ffm_model.py
 │   │   └── fm_model.py
 │   ├── __init__.py
 │   ├── base_engine.py
 │   ├── ffm_engine.py
 │   └── fm_engine.py
 ├── test
 │   ├── data
 │   │   ├── __init__.py
 │   │   ├── bigdata.te.txt
 │   │   ├── bigdata.tr.txt
 │   │   └── small_sample_train.csv
 │   ├── __init__.py
 │   ├── test_ctrengine.py
 │   ├── test_pyffm.py
 │   └── test_utils.py
 ├── __init__.py
 ├── pyffm.py
 └── util.py
LICENSE
README.md
setup.py
```


#### Installation:
```shell script
pip install pyffm
``` 

#### Basic example:
```python
import pandas as pd
from pyffm import PyFFM

training_params = {'epochs': 2, 'reg_lambda': 0.002}
pyffm = PyFFM(model='ffm', training_params=training_params)

from pyffm.test.data import sample_df  # Small training data sample 

# Make sure your file has a label column, default name is 'click' but you can either rename it or pass in label=label_column_name

# Balance the dataset so we get some non-zero predictions (very small training sample)
balanced_df = sample_df[sample_df['click'] == 1].append(sample_df[sample_df['click'] == 0].sample(n=1000)).sample(frac=1)

train_data = balanced_df.sample(frac=0.9)
predict_data = balanced_df.drop(train_data.index)

pyffm.train(train_data)
preds = pyffm.predict(predict_data.drop(columns='click'))


```

Sample data from:  
https://github.com/ycjuan/libffm  
and:  
https://www.kaggle.com/c/criteo-display-ad-challenge

Created using the algorithm described in the original paper:  
https://www.csie.ntu.edu.tw/~cjlin/libffm/

