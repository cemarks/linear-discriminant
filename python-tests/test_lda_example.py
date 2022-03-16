import pytest
import pandas as pd
import numpy as np
import os
import lda as LDA


pth = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or \
    os.path.split(pth)[0]

DATA_PATH = os.path.join(PROJECT_ROOT,"data")
TRAIN_PATH = os.path.join(DATA_PATH,"vowel_train.csv")
TEST_PATH = os.path.join(DATA_PATH,"vowel_test.csv")

@pytest.fixture(scope = 'module')
def train_data():
    train_data_df = pd.read_csv(TRAIN_PATH)
    x_cols = [col for col in list(train_data_df.columns) if 
        col[0:2] == "x."]

    X = train_data_df.loc[:,x_cols].values.astype('float')
    y = train_data_df['y'].values.astype('int32')
    return X, y

@pytest.fixture(scope = 'module')
def test_data():
    test_data_df = pd.read_csv(TEST_PATH)
    x_cols = [col for col in list(test_data_df.columns) if 
        col[0:2] == "x."]

    X = test_data_df.loc[:,x_cols].values.astype('float')
    y = test_data_df['y'].values.astype('int32')
    return X, y

@pytest.fixture(scope = 'module')
def quadratic_kernel_function():

    def quadratic_kf(x1,x2):
        return np.power(np.dot(x1,np.transpose(x2)) + 1,2)
    
    return quadratic_kf


def test_kernel_model(
    train_data,
    test_data,
    quadratic_kernel_function
):
    km = LDA.KernelModel(quadratic_kernel_function, reg_coef = 1000)
    ldm = LDA.LinearDisciminant(km)
    ldm.fit(*train_data)
    y_hat_train = ldm(train_data[0])
    training_precision = np.equal(y_hat_train,train_data[1])
    training_error_rate = 1 - sum(training_precision) / \
        len(training_precision)

    print(f"Training error rate: {100*training_error_rate:1.2f}%\n")
    y_hat_test = ldm(test_data[0])
    test_precision = np.equal(y_hat_test,test_data[1])
    test_error_rate = 1-sum(test_precision)/len(test_precision)
    print(f"Test error rate: {100*test_error_rate:1.2f}%\n")


@pytest.mark.parametrize(
    ("x_coord","y_coord"),
    [
        (1, 2), # p. 107
        (1, 3), # p. 115
        (2, 3), # p. 115
        (1, 7), # p. 115
        (9, 10) # p. 115
    ]
)
def test_train_plot(
    train_data,
    test_data,
    x_coord,
    y_coord
):
    """The goal of this test is to reproduce the plots (or close
    variants of them) shown on pp. 107 & 115 of Hastie, et al.

    Note some coordinates might be reversed due to variations in 
    computation of eigenvectors.
    """

    ldm = LDA.LinearDisciminant()
    ldm.fit(*train_data)
    ldm.plot(x_coord - 1, y_coord - 1)

@pytest.mark.parametrize(
    ("reg_coef",),
    [
        (0,),
        (1,),
        (100,),
        (350,),
        (500,)
    ]
)
def test_regularization(
    train_data,
    reg_coef
):
    ldm = LDA.LinearDisciminant(reg_coef=reg_coef)
    ldm.fit(*train_data)
    ldm.plot(1,2)