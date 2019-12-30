#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 08:18:24 2018

@author: cemarks

Python module to perform general linear discrimenant analysis.
"""

from sklearn.preprocessing import StandardScaler,OneHotEncoder
import numpy as np

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
    """Class to standardize regression models for use in LDA"""
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
    def normalize(self):
        self.scalar = StandardScaler()
        self.scalar.fit(X)
        self.X = self.scalar.transform(self.X_raw)
        y_set = set(self.y_raw)
        self.y_list = list(y_set)
        self.y_dict = {self.y_list[i]:i for i in range(len(self.y_list))}
        Y_int = np.array([self.y_dict[i] for i in self.y_raw],dtype="int32")
        ohe = OneHotEncoder(sparse=False)
        self.Y = ohe.fit_transform(Y_int.reshape(-1,1))
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
        w = np.zeros(len(self.y_list))
        for i in range(len(self.y_list)):
            cl = self.y_list[i]
            eta = self.regression_model(X[np.where(self.y_raw==cl)])
            centroids[i] = np.mean(eta,axis=0)
        self.centroids = centroids

        self.centroids = np.mean(eta[:,1:eta.shape[1]],axis=0)
    def predict(self,X):

    def plot(self,coordx,coordy):
        from matplotlib import pyplot as plt
        eta = np.dot(self.X[:,0:self.X.shape[1]],self.new_B)
        eta_means = np.zeros((len(self.y_list),eta.shape[1]))
        print(eta.shape)
        print(eta_means.shape)
        for i in range(len(self.y_list)):
            inds = [j for j in range(eta.shape[0]) if self.y_dict[self.y_raw[j]]==i]
            eta_means[i,:] = np.mean(eta[inds,:],axis=0)
        colors = [((1/(len(self.y_list)*2)+i/len(self.y_list)),0.5-0.5*(1/(len(self.y_list)*2)+i/len(self.y_list)),1-(1/(len(self.y_list)*2)+i/len(self.y_list))) for i in range(len(self.y_list))]
        plot_colors = [colors[self.y_dict[i]] for i in self.y_raw]
        plt.scatter(
            eta[:,coordx],
            eta[:,coordy],
            s = 2,
            marker = ".",
            c = [colors[self.y_dict[i]] for i in self.y_raw]
        )
        plt.scatter(
            eta_means[:,coordx],
            eta_means[:,coordy], 
            s = 6,
            c = '#FFFFFF',
            linewidths = 3,
            edgecolors = colors
        )
        plt.show()
        plt.clf()
        plt.close()
    def predict(self,new_X,new_Y,coordx,coordy):
        from matplotlib import pyplot as plt
        X = np.concatenate(
            (
                [[1]] * new_X.shape[0],
                self.scalar.transform(new_X),
            ),
            axis = 1
        )
        eta = np.dot(X[:,0:X.shape[1]],self.new_B)
        eta_means = np.zeros((len(self.y_list),eta.shape[1]))
        print(eta.shape)
        print(eta_means.shape)
        for i in range(len(self.y_list)):
            inds = [j for j in range(eta.shape[0]) if self.y_dict[self.y_raw[j]]==i]
            eta_means[i,:] = np.mean(eta[inds,:],axis=0)
        colors = [((1/(len(self.y_list)*2)+i/len(self.y_list)),0.5-0.5*(1/(len(self.y_list)*2)+i/len(self.y_list)),1-(1/(len(self.y_list)*2)+i/len(self.y_list))) for i in range(len(self.y_list))]
        plot_colors = [colors[self.y_dict[i]] for i in self.y_raw]
        plt.scatter(
            eta[:,coordx],
            eta[:,coordy],
            s = 2,
            marker = ".",
            c = [colors[self.y_dict[i]] for i in new_Y]
        )
        plt.scatter(
            eta_means[:,coordx],
            eta_means[:,coordy], 
            s = 6,
            c = '#FFFFFF',
            linewidths = 3,
            edgecolors = colors
        )
        plt.show()
        plt.clf()
        plt.close()

if __name__ == "__main__":
    train_file = "/home/cemarks/Desktop/vowel_train.csv"
    with open(train_file,"r") as f:
        r = f.readlines()
    a = [i.rstrip("\n").split(",") for i in r]
    Xy = [[float(j) for j in i[1:len(i)]] for i in a[1:len(r)] if len(i) > 1]
    X = np.array([[j for j in i[1:len(i)]] for i in Xy],dtype="float")
    y = np.array([i[0] for i in Xy],dtype="int32")
    l = LDA(X,y)
    l.fit()
    train_file = "/home/cemarks/Desktop/vowel_test.csv"
    with open(train_file,"r") as f:
        r_new = f.readlines()
    a_new = [i.rstrip("\n").split(",") for i in r_new]
    Xy_new = [[float(j) for j in i[1:len(i)]] for i in a_new[1:len(r)] if len(i) > 1]
    X_new = np.array([[j for j in i[1:len(i)]] for i in Xy_new],dtype="float")
    y_new = np.array([i[0] for i in Xy_new],dtype="int32")
    l.plot(0,6)





