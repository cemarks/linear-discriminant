#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Python module to perform general Linear Discriminant Analysis (LDA).

See Hastie, T., Tibshirani, R., & Friedman, J. H. (2009), *The elements
    of statistical learning: data mining, inference, and prediction,* 
    2nd ed. New York: Springer, pp 106--119 and pp 440--455.

Available at https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf

This module extends the flexible linear discrimimant analysis 
methods described in Hastie, et. al. to use user-defined kernel functions.
It closely follows the computational steps on p. 445.

Created on Sat May  5 08:18:24 2018

@author: cemarks

Dependencies
------------
scipy
scikit-learn
numpy
matplotlib

Classes
------------
LinearModel
KernelModel
LDA

"""


from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from scipy.special import softmax
from matplotlib import pyplot as plt


class LinearModel:
    """Class to standardize regression models for use in LDA

    Attributes
    ----------
    PARAM : numpy.ndarray
        The model parameter vector.

    Methods
    ----------
    fit(X, Y, reg_coef)
        Fits the model to data, updates PARAM vector, and returns
        predicted values.
    weight_param(THETA)
        Rotate the model by matrix THETA. See computation step 
        3 on p. 445 of Hastie, et. al.
    """

    def __init__(self, reg_coef : float = 0):
        """Initialize LinearModel Object

        Parameters
        ----------
        reg_coef : float (optional)
            L2 Regularization coefficient.  Defaults to 0.     
        """

        self.PARAM = None
        self.reg_coef_ = reg_coef

    def __call__(self, X : np.ndarray) -> np.ndarray:
        """Run the linear model on a given predictor array.

        Parameters
        ----------
        X : numpy.ndarray
            A predictor array.

        Returns
        ----------
        numpy.ndarray
            The predicted values, i.e., the matrix product
            X * THETA, where THETA is the model parameter vector.
        """

        X = self._preprocess(X)
        return np.dot(X, self.PARAM)

    def _check_array(self, X : np.ndarray) -> np.ndarray:
        """Reshape predictor array if necessary.

        Parameters
        ----------
        X : numpy.ndarray
            The predictor array.

        Returns
        ----------
        numpy.ndarray
            Array with correct dimensions.
        """

        X = np.array(X)
        if len(X.shape) < 2:
            X = X.reshape(-1, 1)

        return(X)

    def _preprocess(self, X : np.ndarray) -> np.ndarray:
        """Reshapes input array and adds constant vector

        Parameters
        ----------
        X : numpy.ndarray
            Predictor array.

        Returns
        ----------
        numpy.ndarray
            Reformatted predictor array concatenated with column of
            ones.
        """

        X = self._check_array(X)
        X = np.concatenate(
            (
                np.array([[1]] * X.shape[0]),
                X
            ),
            axis=1
        )
        return X

    def fit(self, X : np.ndarray, Y : np.ndarray) -> None:
        """Fit the Linear Discriminant model

        This method fits the model to data and updates the PARAM
        attribute.

        Parameters
        ----------
        X : numpy.ndarray
            Predictor array
        Y : numpy.ndarray
            Categorical response values, one-hot encoded.

        Returns
        ----------
        numpy.ndarray
            Predicted values
        """

        X = self._preprocess(X)
        self.PARAM = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) \
                     + np.diag([self.reg_coef_] * X.shape[1])), 
                     np.dot(np.transpose(X), Y))

        return np.dot(X, self.PARAM)

    def weight_param(self, THETA : np.ndarray) -> None:
        """Rotate the model parameters by matrix THETA

        This function implements computation step 3, p. 445 in 
        Hastie, et. al., Elements of Statistical Learning and Data
        Mining, 2nd Edition (available at 
        https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf)

        Because the model is linear, rotating the parameter 
        vector by matrix THETA rotates the resulting prediction
        function.  This function performs this operation, updating
        the model parameters (PARAM) vector.

        Parameters
        ----------
        THETA ; numpy.ndarray
            Normalized eigenvector matrix found in computation step 2
            as outlined on p. 445 of Hastie, et. al.
        
        Returns
        ----------
        None

        """

        self.PARAM = np.dot(self.PARAM,THETA)


class KernelModel(LinearModel):
    """Class to standardize kernel regression models for use in LDA

    The kernel function must input two matrices:
        x1 = (p x n)
        x2 = (q x n)
    and must output a (p x q) kernel matrix.

    Attributes
    ----------
    PARAM : numpy.ndarray
        Model parameters (must be fit, initialized as None).
    X_TNG : numpy.ndarray
        Training array
    reg_coef_ : float
        L2 regularization coefficient (applied to kernel)
    kernel_function : callable
        Kernel function (must take two arguments as described above)

    Methods
    ----------
    fit(X, Y)
        Fit the model parameters and return predicted values
    weight_param(THETA)
        Rotate the model by matrix THETA. See computation step 
        3 on p. 445 of Hastie, et. al.
        
    """

    def __init__(
            self,
            kernel_function : callable = None,
            reg_coef : float = 1
        ):
        """Initialize KernelModel Object

        Parameters
        ----------
        kernel_function : callable (optional)
            Function that takes two vectors or matrices as
            numpy arrays and performs the kernel operation on
            them, i.e.,
                x1 = (p x n)
                x2 = (q x n)
                kernel_function(x1, x2) -> (p x q)
            If omitted a linear kernel is used.
        reg_coef : float (optional)
            L2 Regularization coefficient (applied to kernel).
            Defaults to 1.

        """

        self.PARAM = None
        self.X_TNG = None
        self.reg_coef_ = reg_coef
        if kernel_function is None:
            def kf(x1, x2):
                return(np.dot(x1,np.transpose(x2)))
            self.kernel_function = kf
        else:
            self.kernel_function = kernel_function

    def __call__(self, X : np.ndarray) -> np.ndarray:
        """Run the linear model on a given predictor array.

        Parameters
        ----------
        X : numpy.ndarray
            A predictor array.

        Returns
        ----------
        numpy.ndarray
            The predicted values.
        """

        K = self._preprocess(X)
        return np.dot(K, self.PARAM)

    def _preprocess(self, X : np.ndarray) -> np.ndarray:
        """Produce the kernel matrix from an input predictor array.

        Parameters
        ----------
        X : numpy.ndarray
            Predictor array.

        Returns
        ----------
        numpy.ndarray
            kernel matrix
        """

        X = self._check_array(X)
        K = self.kernel_function(X, self.X_TNG)
        return K

    def fit(self, X : np.ndarray, Y : np.ndarray) -> None:
        """Fit the Kernel Regression model

        This method fits the model to data and updates the PARAM
        attribute.

        Parameters
        ----------
        X : numpy.ndarray
            Predictor array
        Y : numpy.ndarray
            Categorical response values, one-hot encoded.

        Returns
        ----------
        numpy.ndarray
            Predicted values
        """

        self.X_TNG = self._check_array(X)
        K = self._preprocess(X)
        self.PARAM = np.dot(np.linalg.inv(K + \
            np.diag([self.reg_coef_] * K.shape[1])), Y)

        return np.dot(K, self.PARAM)



class LinearDisciminant:
    """Class to perform linear discriminant analysis

    This class implements the computation steps given on p. 445 in
    Hastie, et. al., Elements of Statistical Learning and Data
    Mining, 2nd Edition (available at 
    https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf)
        
    Attributes
    -----------
    regression_model : object
        The underlying regression model to use.  Defaults to LinearModel
        unless provided by user during initialization.
    X_raw : numpy.ndarray
        The unscaled training predictor data.
    y_raw : numpy.ndarray
        The integer-encoded target classes.
    X : numpy.ndarray
        The scaled training predictor data.
    Y : numpy.ndarray
        The one-hot encoded training class array.
    scalar_ : sklearn.preprocessing._data.StandardScaler
        The scalar fit on the training data.
    THETA_: numpy.ndarray
        The normalized eigenvector matrix.  See computation step 2 on
        p. 445 in Hastie, et. al.  These are essentially the LDA model
        parameters
    centroids_: numpy.ndarray
        The class centroids.  Denoted as bar(`eta`) in Hastie et. al.
        See equation 12.54 and following text on p. 441.
    w_: numpy.ndarray
        Coordinate weights.  See equation 12.55, p. 441 in 
        Hastie, et. al.


    Methods
    -----------
    fit(X, y)
        Fit the model to supplied training data.
    predict(X, n_dims, probs)
        Make class predictions for predictor data X.
    plot(x_coord,y_coord,X)
        Plot regressed data and centroids along supplied coordinates.
    """
    
    def __init__(
            self,
            regression_model : object = None,
            reg_coef : float = 0
        ):
        """
        Parameters
        ----------
        regression_model : Object (optional)
            An object that has the following methods:
                * fit(X,y) method that fits the model.
                * weight_param(THETA) method that transforms the 
                    regression output according to THETA 
                    (i.e. YH --> np.dot(YH,THETA); see step 3, p. 445
                    in Hastie, et. al.).
                * __call__ method that inputs an X array and returns
                    corresponding (transformed) estimates
            Defaults to basic linear model.
        reg_coef : float (optional)
            Regularization coefficient to be passed to LinearModel.
            Only used if no regression_modl is provided. Defaults to 0.
        """

        # Initialize model attributes / parameters to None
        self.X_raw = None
        self.y_raw = None
        self.scaler_ = StandardScaler()
        self.X = None
        self._y_list = None
        self._y_dict = None
        self.Y = None
        self.THETA_ = None
        self.centroids_ = None
        self.w_ = None

        if regression_model is None:
            self.regression_model = LinearModel(reg_coef = reg_coef)
        else:
            self.regression_model = regression_model


    def __call__(
            self,
            X : np.ndarray,
            n_dims : int = None,
            probs : bool = False
        ) -> np.ndarray:
        """Predict classes for data array X

        Parameters
        ----------
        X : numpy.ndarray
            Predictor array
        n_dims : int (optional)
            Number of dimensions to include in centroid distance
            computations (euclidean distances).  Defaults to using
            all dimensions.
        probs : bool (optional)
            If True, will return class probability matrix instead
            of class predictions. Probabilities are determined 
            directly from centroid distances. Defaults to False.
        
        Returns
        -----------
        numpy.ndarray
            Predicted classes, or predicted class probabilities, if 
            `probs` is True
        """
        
        return self.predict(X, n_dims, probs)


    def _normalize(self) -> None:
        """Normalize and prepare training data

        Performs standard scaling of training X data and one-hot
        encodes training y data.

        Parameters
        ----------
        (None)

        Returns
        ----------
        None
        """
        
        self.scaler_.fit(self.X_raw)
        self.X = self.scaler_.transform(self.X_raw)
        y_set = set(self.y_raw)
        self._y_list = list(y_set)
        self._y_dict = {self._y_list[i]:i for i \
            in range(len(self._y_list))}

        Y_int = np.array([self._y_dict[i] for i \
            in self.y_raw], dtype = "int32")

        ohe = OneHotEncoder(sparse=False)
        self.Y = ohe.fit_transform(Y_int.reshape(-1, 1))


    def _scale_X(self, X : np.ndarray) -> np.ndarray:
        """Apply the training scaler to predictor data

        Note: this function is intended for new predictor data, i.e.,
            the standard scaler must already have been fit on the 
            training data.
        
        Parameters
        ----------
        X : numpy.ndarray
            New predictor data
        
        Returns
        ----------
        numpy.ndarray
            Scaled predictor data, according to the training scaler. 
            Dimensions will be added as well if required.
        """
        
        if len(X.shape) < 2:
            X_reshape = X.reshape(1, -1)
        else:
            X_reshape = X

        X_rescale = self.scaler_.transform(X_reshape)
        return X_rescale


    def fit(self, X : np.ndarray, y : np.ndarray) -> None:
        """Fit the Linear Discriminent Model.

        This method does not return anything but fits the model and
        updates the parameters in the `regression_model` using that
        object's `weight_model` method.  Updates X, Y, scaler_, THETA_,
        centroids_, and w_ object attributes.

        Parameters
        -----------
        X : numpy.ndarray
            A 2-dimensional array.  The training predictor matrix.
        y : numpy.ndarray
            A one dimensional array of target classes encoded as 
            integers.  The training response classes.
        
        Returns
        ------------
        None
        """

        # Prep data
        self.X_raw = X
        self.y_raw = y
        self._normalize()

        """Step 1 (p. 445, Hastie, et. al.)"""
        YH = self.regression_model.fit(self.X, self.Y)

        """Step 2 (p. 445, Hastie, et. al.)"""
        YTYH = np.dot(np.transpose(self.Y), YH)
        eig_vals, EIG_VECTORS = np.linalg.eig(YTYH)
        print(eig_vals)
        order = np.argsort(-1 * eig_vals)
        EIG_VECTORS_SORTED = EIG_VECTORS[:, order]
        eig_vals_sorted = eig_vals[order]
        D_pi = np.dot(np.transpose(self.Y), self.Y) / self.Y.shape[0]
        
        # Normalization to get THETA
        NORM_MATRIX = np.diag(1 / np.sqrt(np.diag(np.dot(np.transpose(
            EIG_VECTORS_SORTED), np.dot(D_pi, EIG_VECTORS_SORTED)))))

        self.THETA_ = np.dot(EIG_VECTORS_SORTED, NORM_MATRIX)

        """Step 3 (p. 445, Hastie, et. al.).  Model update by Theta
        Because original model is linear, we only need to weight
        the parameters."""
        self.regression_model.weight_param(self.THETA_)

        """Compute the centroids and coordinate weights. See p. 141
        in Hastie, et. al."""
        centroids = np.zeros((len(self._y_list), len(self._y_list)))
        for i in range(len(self._y_list)):
            cl = self._y_list[i]
            eta = self.regression_model(
                self.X[np.where(self.y_raw==cl)])

            centroids[i] = np.mean(eta,axis=0)
        
        self.centroids_ = centroids[:, 1:]
        SS_NORM = eig_vals_sorted[0]
        RS = eig_vals_sorted[1:] / SS_NORM
        self.w_ = np.sqrt(1 / (RS * (1 - RS)))
        

    def _eta_minus_etameans_squared(
            self,
            single_eta : np.ndarray
        ) -> np.ndarray:
        """Compute coordinate squared differences from centroids.

        See equation 12.54, p. 441 in Hastie, et. al.  This method 
        computes the squared differences.

        Parameters
        ----------
        single_eta : numpy.ndarray
            Point in regression-transormed (eta) space.
        
        Returns
        ----------
        numpy.ndarray
            Vector of squared differences from centroids.
        """
        
        if len(single_eta.shape) < 2:
            eta = single_eta.reshape(1, -1)[:, 1:]
        else:
            eta = single_eta[:, 1:]

        eta_matrix = eta.repeat(len(self.centroids_), axis = 0)
        eta_diff = eta_matrix - self.centroids_
        eta_diff_squared = np.power(eta_diff, 2)
        return eta_diff_squared


    def _mahalanobis_distance(
            self,
            X_scaled : np.ndarray,
            n_dims : int = None
        ) -> np.ndarray:
        """Mahalanobis distances for a set of predictors

        See eq. 12.54, p. 441 of Hastie, et. al.  This method
        returns a matrix o where 
            o[i,k] = \delta_{J}(X[i,:],\hat{\mu}_{k}).
        
        Parameters
        ----------
        X_scaled : numpy.ndarray
            Scaled predictor matrix
        n_dims : int (optional)
            Number of dimensions to include (defaults is all
            dimensions).
        
        Returns
        ----------
        numpy.ndarray
            Matrix of Mahalanobis distances (minus D(x), see
            equation 12.54, p. 441 in Hastie, et al).
        
        """
        
        eta_matrix = self.regression_model(X_scaled)
        w = self.w_.copy()
        if (n_dims is not None) and (n_dims < len(w)):
            for i in range(n_dims, len(w)):
                w[i] = 0
        
        distances = np.zeros(eta_matrix.shape)

        for i, eta in enumerate(eta_matrix):
            dist_sq = self._eta_minus_etameans_squared(eta)
            distances[i, :] = np.dot(dist_sq, w)

        return distances


    def predict(
            self,
            X : np.ndarray,
            n_dims : int = None,
            probs : bool = False
        ) -> np.ndarray:
        """Predict classes for data array X

        Parameters
        ----------
        X : numpy.ndarray
            Predictor array
        n_dims : int (optional)
            Number of dimensions to include in centroid distance
            computations (euclidean distances).  Defaults to using
            all dimensions.
        probs : bool (optional)
            If True, will return class probability matrix instead
            of class predictions. Probabilities are determined 
            directly from centroid distances. Defaults to False.
        
        Returns
        -----------
        numpy.ndarray
            Predicted classes, or predicted class probabilities, if 
            `probs` is True
        """
        
        X_scaled = self._scale_X(X)
        distance_matrix = self._mahalanobis_distance(X_scaled, n_dims)
        if probs:
            sm = softmax(-1 * distance_matrix, axis = 1)
            return sm
        else:
            class_ints = np.argmin(distance_matrix, axis = 1)
            class_names = [self._y_list[i] for i in class_ints]
            return class_names


    def plot(
            self,
            x_coord : int,
            y_coord : int,
            X : np.ndarray = None
        ) -> None:
        """Plot regressed data and model centroids on two dimensions.

        Coordinate indices are 0-indexed, with `0` used to identify the
        direction associated with the largest eigenvalue.  See 
        development in Chapter 4.3 in Hastie, et al., with particular
        attention to Figure 4.4, p. 107.

        This method creates and shows the plot, but does not return
        anything.

        Parameters
        ----------
        x_coord : int
            Coordinate index to use along x-axis.  
        y_coord : int
            Coordinate index to use along y-axis.
        X : numpy.ndarray (optional)
            Predictor data to plot.  If omitted the training data will
            be used.
        
        Returns
        ----------
        None
        """
        
        if X is not None:
            X = self._scale_X(X)
        else:
            X = self.X

        eta = self.regression_model(X)
        eta = eta[:, 1:]
        colors = [
            (
                (1 / (len(self._y_list) * 2) + i / len(self._y_list)), # Red
                0.5 - 0.5 * (1 / (len(self._y_list) * 2) + i / len(self._y_list)), # Green
                1 - (1 / (len(self._y_list) * 2) + i / len(self._y_list)) # Blue
            ) for i in range(len(self._y_list))
        ]
        plot_colors = [colors[self._y_dict[i]] for i in self.y_raw]
        plt.scatter(
            self.w_[x_coord] * eta[:, x_coord],
            self.w_[y_coord] * eta[:, y_coord],
            s = 2,
            marker = ".",
            c = plot_colors
        )
        plt.scatter(
            self.w_[x_coord] * self.centroids_[:, x_coord],
            self.w_[y_coord] * self.centroids_[:, y_coord], 
            s = 6,
            c = '#FFFFFF',
            linewidths = 3,
            edgecolors = colors
        )
        plt.xlabel(f"Coordinate {x_coord + 1}")
        plt.ylabel(f"Coordinate {y_coord + 1}")
        plt.show()
        plt.clf()
        plt.close()
