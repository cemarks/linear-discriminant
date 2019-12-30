#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 08:18:24 2018

@author: cemarks

Python module to perform general linear discrimenant analysis.
"""

from sklearn.preprocessing import StandardScaler,OneHotEncoder
import numpy as np
from scipy.special import softmax


class LinearModel:
    """Class to standardize regression models for use in LDA"""
    def __init__(self):
        self.PARAM = None
    def __call__(self,X):
        X = self.preprocess(X)
        return np.dot(X,self.PARAM)
    def check_array(self,X):
        X = np.array(X)
        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        return(X)
    def preprocess(self,X):
        X = self.check_array(X)
        X = np.concatenate(
            (
                np.array([[1]]*X.shape[0]),
                X
            ),
            axis=1
        )
        return X
    def fit(self,X,Y,regularize_coef):
        X = self.preprocess(X)
        self.PARAM = np.dot(np.linalg.inv(np.dot(np.transpose(X),X) + np.diag([regularize_coef]*X.shape[1])),np.dot(np.transpose(X),Y))
        return np.dot(X,self.PARAM)
    def weight_param(self,THETA):
        self.PARAM = np.dot(self.PARAM,THETA)


class KernelModel:
    """Class to standardize nonparametric (kernel) regression models for use in LDA
    The kernel function must input two matrices:
        x1 = (p x n)
        x2 = (q x n)
    and output a (p x q) kernel matrix
    """
    def __init__(self,kernel_function = None):
        self.PARAM = None
        self.X_TNG = None
        if kernel_function is None:
            def kf(x1,x2):
                return(np.dot(x1,np.transpose(x2)))
            self.kernel_function = kf
        else:
            self.kernel_function = kernel_function
    def __call__(self,X):
        K = self.preprocess(X)
        return np.dot(K,self.PARAM)
    def check_array(self,X):
        X = np.array(X)
        if len(X.shape) < 2:
            X = X.reshape(-1,1)
        return(X)
    def preprocess(self,X):
        X = self.check_array(X)
        K = self.kernel_function(X,self.X_TNG)
        return K
    def fit(self,X,Y,regularize_coef):
        self.X_TNG = self.check_array(X)
        K = self.preprocess(X)
        self.PARAM = np.dot(np.linalg.inv(K + np.diag([regularize_coef]*K.shape[1])),Y)
        return np.dot(K,self.PARAM)
    def weight_param(self,THETA):
        self.PARAM = np.dot(self.PARAM,THETA)



class LDA:
    """
    Class to perform linear discrimenant analysis
    X is a 2-dimensional array.
    y is a one dimensional array of target classes
    regression_model must have the following:
        -a fit(X,y,regularize_coef) method that fits the model.
        -a weight_param(THETA) method that transforms the regression output 
            according to THETA (i.e. YH --> np.dot(YH,THETA))
        -a __call__ method that inputs an X array and returns corresponding
           (transformed) estimates
        
    """
    def __init__(self,X,y,regression_model=None,reg_coef=1):
        self.X_raw = X
        self.y_raw = y
        self.reg_coef = reg_coef
        self.normalize()
        if regression_model is None:
            self.regression_model = LinearModel()
        else:
            self.regression_model = regression_model
    def __call__(self,new_X,dims=None,probs=False):
        X = self.scale_new_x(new_X)
        distance_matrix = self.m_distance(X,dims)
        if probs:
            sm = softmax(-1*distance_matrix,axis=1)
            return sm
        else:
            class_ints = np.argmin(distance_matrix,axis=1)
            class_names = [self.y_list[i] for i in class_ints]
            return class_names
    def normalize(self):
        self.scalar = StandardScaler()
        self.scalar.fit(self.X_raw)
        self.X = self.scalar.transform(self.X_raw)
        y_set = set(self.y_raw)
        self.y_list = list(y_set)
        self.y_dict = {self.y_list[i]:i for i in range(len(self.y_list))}
        Y_int = np.array([self.y_dict[i] for i in self.y_raw],dtype="int32")
        ohe = OneHotEncoder(sparse=False)
        self.Y = ohe.fit_transform(Y_int.reshape(-1,1))
    def scale_new_x(self,new_X):
        if len(new_X.shape) < 2:
            new_X = new_X.reshape(1,-1)
        X = self.scalar.transform(new_X)
        return X
    def fit(self):
        YH = self.regression_model.fit(self.X,self.Y,self.reg_coef)
        YTYH = np.dot(np.transpose(self.Y),YH)
        l,W = np.linalg.eig(YTYH)
        order = np.argsort(-1*l)
        W_sorted = W[:,order]
        l_sorted = l[order]
        D_pi = np.dot(np.transpose(self.Y),self.Y)/self.Y.shape[0]
        A = np.diag(1/np.sqrt(np.diag(np.dot(np.transpose(W_sorted),np.dot(D_pi,W_sorted)))))
        self.THETA = np.dot(W_sorted,A)
        self.regression_model.weight_param(self.THETA)
        centroids = np.zeros((len(self.y_list),len(self.y_list)))
        for i in range(len(self.y_list)):
            cl = self.y_list[i]
            eta = self.regression_model(self.X[np.where(self.y_raw==cl)])
            centroids[i] = np.mean(eta,axis=0)
        self.centroids = centroids[:,1:]
        SS_NORM = l_sorted[0]
        RS = l_sorted[1:]/SS_NORM
        self.w = np.sqrt(1/(RS*(1-RS)))
    def eta_minus_etameans_squared(self,single_eta):
        if len(single_eta.shape) < 2:
            eta = single_eta.reshape(1,-1)[:,1:]
        else:
            eta = single_eta[:,1:]
        eta_matrix = eta.repeat(len(self.centroids),axis=0)
        eta_diff = eta_matrix - self.centroids
        eta_diff_squared = np.power(eta_diff,2)
        return(eta_diff_squared)
    def m_distance(self,scaled_X,d=None):
        eta = self.regression_model(scaled_X)
        w = self.w.copy()
        if (d is not None) and (d < len(w)):
            for i in range(d,len(w)):
                w[i] = 0
        o = np.zeros(eta.shape)
        for i in range(len(eta)):
            e = eta[i]
            D = self.eta_minus_etameans_squared(e)
            o[i,:] = np.dot(D,w)
        return(o)
    def predict(self,X):
        pass
    def plot(self,coordx,coordy,new_X = None):
        from matplotlib import pyplot as plt
        if new_X is not None:
            X = self.scale_new_x(new_X)
        else:
            X = self.X
        eta = self.regression_model(X)
        eta = eta[:,1:]
        colors = [((1/(len(self.y_list)*2)+i/len(self.y_list)),0.5-0.5*(1/(len(self.y_list)*2)+i/len(self.y_list)),1-(1/(len(self.y_list)*2)+i/len(self.y_list))) for i in range(len(self.y_list))]
        plot_colors = [colors[self.y_dict[i]] for i in self.y_raw]
        plt.scatter(
            self.w[coordx]*eta[:,coordx],
            self.w[coordy]*eta[:,coordy],
            s = 2,
            marker = ".",
            c = plot_colors
        )
        plt.scatter(
            self.w[coordx]*self.centroids[:,coordx],
            self.w[coordy]*self.centroids[:,coordy], 
            s = 6,
            c = '#FFFFFF',
            linewidths = 3,
            edgecolors = colors
        )
        plt.show()
        plt.clf()
        plt.close()


if __name__ == "__main__":
    train_file = "/home/cemarks/Documents/python-lda/data/vowel_train.csv"
    with open(train_file,"r") as f:
        r = f.readlines()
    a = [i.rstrip("\n").split(",") for i in r]
    Xy = [[float(j) for j in i[1:len(i)]] for i in a[1:len(r)] if len(i) > 1]
    X = np.array([[j for j in i[1:len(i)]] for i in Xy],dtype="float")
    y = np.array([i[0] for i in Xy],dtype="int32")
    def quadratic_kf(x1,x2):
        return np.power(np.dot(x1,np.transpose(x2)) + 1,2)
    km = KernelModel(quadratic_kf)
    l = LDA(X,y,regression_model=km,reg_coef=1000)
    l.fit()
    y_hat_train = l(X)
    training_precision = np.equal(y_hat_train,y)
    training_error_rate = 1-sum(training_precision)/len(training_precision)
    print("Training error rate: {0:1.2f}%\n".format(100*training_error_rate))
    test_file = "/home/cemarks/Documents/python-lda/data/vowel_test.csv"
    with open(test_file,"r") as f:
        r_new = f.readlines()
    a_new = [i.rstrip("\n").split(",") for i in r_new]
    Xy_new = [[float(j) for j in i[1:len(i)]] for i in a_new[1:len(r)] if len(i) > 1]
    X_new = np.array([[j for j in i[1:len(i)]] for i in Xy_new],dtype="float")
    y_new = np.array([i[0] for i in Xy_new],dtype="int32")
    y_hat_test = l(X_new)
    test_precision = np.equal(y_hat_test,y_new)
    test_error_rate = 1-sum(test_precision)/len(test_precision)
    print("Test error rate: {0:1.2f}%\n".format(100*test_error_rate))





