# Linear Discriminant Analysis (LDA)


This repository contains minimal code implementations of LDA. For more information on LDA see pp. 106--119 and pp 440--455 of Hastie, T., Tibshirani, R., & Friedman, J. H. (2009), *The elements of statistical learning: data mining, inference, and prediction,* 2nd ed. New York: Springer, pp 106--119 and pp 440--455, available [online](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf).


## Python

### Set up a virtual environment
I recommend using a python virtual environment.

```bash
$ python -m venv env
$ source env/bin/activate
$ pip install --upgrade pip
$ pip install -r python-requirements.txt
```

### Install the linear-discriminant-analysis package

```bash
$ pip install ./python-linear-discriminant
```

Alternatively, if you want to develop, install the package with the -e flag

```bash
$ pip install -e ./python-linear-discriminant
```

## Testing

The `python-tests/test_lda-example.py` provides test methods that can be executed using `pytest` or that can be references as examples.  To run the test scripts, `pytest` and `pandas` must be installed.

```bash
$ pip install -r python-dev-requirements.txt
$ pytest -s python-tests
```

## Coming Soon: Documentation of the LDA in R

The `R` folder contains linear discriminant analysis scripts that run in `R`, but need to be refactored and documented.


