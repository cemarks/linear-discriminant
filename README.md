# Linear Discriminant Analysis (LDA) in Python


This repository contains minimal code implementations of LDA. For more information on LDA see pp. 106--119 and pp 440--455 of Hastie, T., Tibshirani, R., & Friedman, J. H. (2009), *The elements of statistical learning: data mining, inference, and prediction,* 2nd ed. New York: Springer, pp 106--119 and pp 440--455, available [online](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf).

This implementation allows for LDA as described on pp 106--119 (Hastie, et al) as well as Flexible Discriminant Analysis (FDA) as described on pp 440--445 (ibid).  In particular, in the latter case this module enables the user to substitute an arbitrary kernel into the underlying regression.

## Installation

### Set up a virtual environment
I recommend using a python virtual environment.

```bash
# Shell
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Install the linear-discriminant-analysis package

```bash
# Shell
pip install ./python-lda
```

Alternatively, if you want to develop, install the package with the -e flag

```bash
# Shell
pip install -e ./python-lda
```

### Testing

The `python-tests/test_lda-example.py` provides test methods that can be executed using `pytest` or that can be references as examples.  To run the test scripts, `pytest` and `pandas` must be installed.

```bash
# Shell
pip install -r dev-requirements.txt
pytest -s python-tests
```

## Usage

The python-tests/test_lda_example.py file contains examples on the vowel data comparable to the development in Chapters 4.4 and 12.5 in Hastie, et al.  The code below reproduces the analysis leading to the plot in Figure 4.4 on p. 107.  Note that the resulting plot might have one or both axes reversed from the text due to arbitrary differences in Eigenvector decomposition.

The code reads the vowel data from the data folder in this repository, assumed to be at `./data` or `${PROJECT_ROOT}/data`

```python
# Python
import numpy as np
import os
import lda as LDA

# Not a dependency, but used in this example
import pandas as pd


# Point to data path
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or "."

DATA_PATH = os.path.join(PROJECT_ROOT,"data")
TRAIN_PATH = os.path.join(DATA_PATH,"vowel_train.csv")
TEST_PATH = os.path.join(DATA_PATH,"vowel_test.csv")

# Read data
train_data_df = pd.read_csv(TRAIN_PATH)
x_cols = [col for col in list(train_data_df.columns) if 
    col[0:2] == "x."]

X = train_data_df.loc[:,x_cols].values.astype('float')
y = train_data_df['y'].values.astype('int32')

test_data_df = pd.read_csv(TEST_PATH)
x_cols = [col for col in list(test_data_df.columns) if 
    col[0:2] == "x."]

X_test = test_data_df.loc[:,x_cols].values.astype('float')
y_test = test_data_df['y'].values.astype('int32')


ldm = LDA.LinearDisciminant()
ldm.fit(*(X,y))
ldm.plot(0,1)

# Error rate on test set.
# Compare with linear model result, Table 12.3, p. 444,
#     Hastie, et al.

test_predict = ldm.predict(X_test)
count_correct = len(
    [i for i in range(len(test_predict)) if 
        test_predict[i] == y_test[i]]
)
misclass_rate = 1 - count_correct / len(test_predict)

print(f"Test misclassification rate: {misclass_rate: 1.2f}")
```

## Implementation in R

For a similar implementation in R, see the [klda](https://github.com/cemarks/klda) package.

