#!/usr/bin/env Rscript

#' @author Christopher E. Marks, \email{cemarks@@alum.mit.edu}
#' @references \url{https://github.com/cemarks/linear-discriminant}
#' @seealso \code{\link{mda}}
#' @keywords mda
#' ...
#' @importFrom mda fda
#' @importFrom kernlab kernelMatrix
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 geom_point
#' @importFrom ggplot2 scale_shape_manual
#' @importFrom ggplot2 scale_size_manual
#' @importFrom ggplot2 guides

# See Hastie, T., Tibshirani, R., & Friedman, J. H. (2009), *The elements
#     of statistical learning: data mining, inference, and prediction,*
#     2nd ed. New York: Springer, pp 106--119 and pp 440--455.

# Available at
# https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf

# This module extends the flexible linear discrimimant analysis
# methods described in Hastie, et. al. to use user-defined kernel functions.
# It relies on the CRAN "mda" library, originally developed by Hastie, et al.

#' Flexible Discriminant Analysis with user-defined kernel
#'
#' Builds a regularized flexible discriminant model from user supplied
#' data, kernel function, and regularization parameter.
#'
#' @param formula formula of the form ‘y~x’ it describes the response and the
#' predictors.  The formula can be more complicated, such as
#' ‘y~log(x)+z’ etc (see ‘formula’ for more details).  The
#' response should be a factor representing the response
#' variable, or any vector that can be coerced to such (such as
#' a logical variable).
#' @param data data.frame containing predictor and response data.  The predictor
#' data should be centered and scaled (normalized).
#' @param kerneldot method from \code{kernlab} \code{dot} methods or
#' user-defined kernel function with the same signature.  Default is
#' linear kernel.
#' @param lambda numeric regularization coefficient.  Default is 1.
#' @param ... additional parameters passed to kernel.dot
#'
#' @return \code{mda} \code{fda} object.
#'
#' @seealso \code{\link[mda]{fda}}, \code{\link[kernlab]{dots}},
#' \code{\link{linear_fda}},
#' \url{https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf}
#' @export
#'
#' @examples
#' data(iris)
#'
#' # Normalize data
#' means <- apply(iris[,1:4],2,mean)
#' stddevs <- apply(iris[,1:4],2,sd)
#' iris_normalized <- iris
#' for (col in 1:4) {
#'   iris_normalized[,col] <- (iris[,col] - means[col])/stddevs[col]
#' }
#'
#' model <- kernel_fda(
#'   Species ~ .,
#'   iris_normalized,
#'   kerneldot = kernlab::rbfdot,
#'   lambda = 1,
#'   sigma = 0.1
#' )
kernel_fda <- function(
    formula,
    data,
    kerneldot = NULL,
    lambda = 1,
    ...
) {
    if (is.null(kerneldot)) {
        kerneldot <- lineardot
    }
    model <- mda::fda(
        formula = formula,
        data = data,
        method = kernel_reg_model,
        kernel_function = kerneldot(...),
        lambda = lambda
    )
    return(model)
}


#' Flexible Discriminant Analysis with linear regression model
#'
#' Builds a regularized flexible discriminant model from user supplied
#' data and L2-regularization parameter, using linear regression model.
#' This method is useful for quickly comparing more complicated models
#' to a linear model and for debugging more complicated methods.  By
#' itself, it does not provide functionality beyond what is already
#' provided in the \code{mda} package, and might be less efficient
#' than equivalent methods there.
#'
#' @param formula formula of the form ‘y~x’ it describes the response and the
#' predictors.  The formula can be more complicated, such as
#' ‘y~log(x)+z’ etc (see ‘formula’ for more details).  The
#' response should be a factor representing the response
#' variable, or any vector that can be coerced to such (such as
#' a logical variable).
#' @param data data.frame containing predictor and response data.  The
#' predictor data should be centered and scaled (normalized).
#' @param lambda numeric regularization coefficient.  Default is 1.
#'
#' @return \code{mda} \code{fda} object.
#'
#' @seealso \code{\link[mda]{fda}}, \code{\link[kernlab]{dots}},
#' \code{\link{kernel_fda}},
#' \url{https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf}
#' @export
#'
#' @examples
#' data(iris)
#'
#' # Normalize data
#' means <- apply(iris[,1:4],2,mean)
#' stddevs <- apply(iris[,1:4],2,sd)
#' iris_normalized <- iris
#' for (col in 1:4) {
#'   iris_normalized[,col] <- (iris[,col] - means[col])/stddevs[col]
#' }
#'
#' model <- linear_fda(
#'   Species ~ .,
#'   iris_normalized,
#'   lambda = 1
#' )
linear_fda <- function(
    formula,
    data,
    lambda = 1
) {
    model <- mda::fda(
        formula = formula,
        data = data,
        method = linear_reg_model,
        lambda = lambda
    )
    return(model)
}



#' Linear kernel method
#'
#' Generate linear kernel matrix (i.e., dot product).  This
#' function is meant to be used to build a kernel_model object,
#' and not meant to be used explicity.
#'
#' @param addOnes logical if TRUE will add an intercept column
#' (i.e. a column of 1's) to input matricies
#'
#' @return function that takes to matrices and returns the
#' dot product
#'
#' @seealso \code{\link{kernel_fda}}, \code{\link[kernlab]{dots}}
lineardot <- function(add_ones = FALSE) {
    addones <- add_ones
    linear_kernel <- function(x1, x2) {
        x1_matrix <- x1
        x2_matrix <- x2
        if (addones) {
            x1_matrix <- append(1, x1_matrix)
            x2_matrix <- append(1, x2_matrix)
        }
        return(x1_matrix %*% x2_matrix)
    }

    return(linear_kernel)
}


#' Kernel Regression
#'
#' Construct a linear regression model using the user-supplied
#' kernel function.  This function is called by \code{kernel_fda} and
#' is not intended to be used outside of that context.
#'
#' @param x matrix of predictor data.
#' @param y vector or matrix of response data.
#' @param w numeric not used.  Intended to be weights to be applied to training
#' data.
#' @param kernel_function function to compute the kernel.
#' @param lambda numeric regularization coefficient.  Default is 1.
#' @return function that takes to matrices and returns the
#' dot product
#' @seealso \code{\link{kernel_fda}}
kernel_reg_model <- function(
    x,
    y,
    w,
    kernel_function,
    lambda = 1
) {
    if (is.null(kernel_function)) {
        kernel_function <- lineardot()
    }
    kernel_matrix <- kernlab::kernelMatrix(kernel = kernel_function,
        x = as.matrix(x))
    alpha <- solve(kernel_matrix + diag(lambda, nrow = nrow(kernel_matrix)), y)
    y_hat <- kernel_matrix %*% alpha
    model <- structure(
        list(
            x = x,
            y = y,
            alpha = alpha,
            lambda = lambda,
            kernel_fn = kernel_function,
            fitted.values = y_hat
        ),
        class = "kernel_regression")
    return(model)
}


#' Linear Regression
#'
#' Construct a L2-regularized linear regression model.
#' This function is called by \code{kernel_fda} and
#' is not intended to be used outside of that context.
#'
#' @param x matrix of predictor data.
#' @param y vector or matrix of response data.
#' @param w numeric not used.  Intended to be weights to be applied to training
#' data.
#' @param lambda numeric regularization coefficient.  Default is 1.
#' @return function that takes to matrices and returns the
#' dot product
#' @seealso \code{\link{linear_fda}}
linear_reg_model <- function(
    x,
    y,
    w,
    lambda = 1
) {
    b <- solve(
        t(x) %*% x + diag(lambda, nrow = ncol(x)),
        t(x) %*% y
    )
    y_hat <- x %*% b
    model <- structure(
        list(
            x = x,
            y = y,
            beta = b,
            lambda = lambda,
            fitted.values = y_hat
        ),
        class = "linear_regression"
    )
    return(model)
}



#' Predict from kernel regression model
#'
#' Estimate response values using a kernel regression model.  This method
#' gets called implicitly in the \code{predict.fda} method when the
#' fda model is generated by the \code{kernel_fda} method.
#'
#' @param model_obj kernel regression model object.
#' @param newdata matrix containing new predictor data.  If not provided,
#' training data will be used.
#'
#' @return matrix of predicted values
#' @seealso \code{\link{kernel_reg_mod}}
predict.kernel_regression <- function(model_obj, newdata) {
    if (missing(newdata)) {
        if (is.null(model_obj$fitted.values)) {
            kernel_matrix <- kernlab::kernelMatrix(kernel = model_obj$kernel_fn,
                x = as.matrix(model_obj$x))
            alpha <- solve(kernel_matrix + diag(model_obj$lambda,
                nrow = nrow(kernel_matrix)), model_obj$y)
            y_hat <- kernel_matrix %*% alpha
            return(y_hat)
        } else {
            return(model_obj$fitted.values)
        }
    } else {
        kernel_matrix <- kernlab::kernelMatrix(model_obj$kernel_fn,
            newdata, as.matrix(model_obj$x))
        alpha <- model_obj$alpha
        return(kernel_matrix %*% alpha)
    }
}

#' Predict from linear regression model
#'
#' Estimate response values using a linear regression model.  This method
#' gets called implicitly in the \code{predict.fda} method when the
#' fda model is generated by the \code{linear_fda} method.
#'
#' @param model_obj linear regression model object.
#' @param newdata matrix containing new predictor data.  If not provided,
#' training data will be used.
#'
#' @return matrix of predicted values
#' @seealso \code{\link{linear_reg_mod}}
predict.linear_regression <- function(model_obj, newdata) {
    if (missing(newdata)) {
        if (is.null(model_obj$fitted.values)) {
            return(model_obj$x %*% model_obj$beta)
        } else {
            return(model_obj$fitted.values)
        }
    } else {
        return(newdata %*% model_obj$beta)
    }
}


#' Plot a flexible linear discriminant model
#'
#' Plots centroids and cannonical variates in specified coordinates.
#' The coordinate values must be strictly less than the number of classes.
#' (See p. 445, Hastie, et al.)
#' in the model.  The first coordinate correspondes to the largest eigenvalue.
#'
#' @param model_obj \code{mda} \code{fda} object.
#' @param newdata data frame new data to be plotted.  Must have same names as
#' data used for fitting.  If omitted, training data will be plotted.
#' @param xcoord integer coordinate to use on x axis.
#' @param ycoord integer coordinate to use on y axis.
#'
#' @return \code{ggplot2} \code{ggplot} object
#'
#' @seealso \code{\link[ggplot2]{ggplot}}, \code{\link{kernel_fda}},
#' \code{\link{linear_fda}}
#'
#' @export
#' @examples
#' data(iris)
#'
#' # Normalize data
#' means <- apply(iris[,1:4], 2, mean)
#' stddevs <- apply(iris[,1:4], 2, sd)
#' iris_normalized <- iris
#' for (col in 1:4) {
#'   iris_normalized[, col] <- (iris[, col] - means[col]) /
#'     stddevs[col]
#' }
#'
#' model <- kernel_fda(
#'   Species ~ .,
#'   iris_normalized,
#'   kerneldot = kernlab::rbfdot,
#'   lambda = 1,
#'   sigma = 0.1
#' )
#'
#' plot(model)
plot.fda <- function(model_obj, newdata, xcoord = 1, ycoord=2) {
    if (missing(newdata)) {
        cannonical_variates <- predict(model_obj, type = "variates")
        classes <- predict(model_obj)
    } else{
        cannonical_variates <- predict(model_obj, newdata, type = "variates")
        classes <- predict(model_obj, newdata)
    }
    centroid_df <- data.frame(
        x = model_obj$means[, xcoord],
        y = model_obj$means[, ycoord],
        class = row.names(model_obj$means),
        type = rep("centroid", nrow(model_obj$means))
    )
    variate_df <- data.frame(
        x = cannonical_variates[, xcoord],
        y = cannonical_variates[, ycoord],
        class = as.character(classes),
        type = rep("variate", nrow(cannonical_variates))
    )
    df <- rbind(centroid_df, variate_df)
    g <- ggplot2::ggplot(
        data = df,
        mapping = ggplot2::aes(x = x, y = y, color = class,
          shape = type, size = type)
    ) +
      ggplot2::geom_point(stroke = 3) +
      ggplot2::scale_size_manual(
          values = c(variate = 0.5, centroid = 8),
          limits = c("variate", "centroid"),
          breaks = c("variate", "centroid"),
          guide = "none"
      ) +
      ggplot2::scale_shape_manual(
          values = c(variate = 20, centroid = 1),
          limits = c("variate", "centroid"),
          breaks = c("variate", "centroid"),
          guide = "none"
      ) +
      ggplot2::guides(color = "none")

    print(g)

    return(g)
}


#' Test model performance
#'
#' Compute the correct classification rate and misclassification rate
#' for a fit fda model.
#'
#' @param model_obj \code{mda} \code{fda} model.
#' @param newdata data frame providing data to investigate.  Must include
#' all of the predictor variables and the response variable.  If omitted,
#' training data will be used.
#' @param pring_results logical.  If TRUE, results will be outputted to
#' stdout
#'
#' @return numeric correct classification rate.
#'
#' @export
#'
#' @examples
#' data(iris)
#'
#' # Separate into training & test
#' data_permutation_order <- sample(nrow(iris))
#' train_cutoff <- round(0.75 * nrow(iris))
#' train_indices <- data_permutation_order[1:train_cutoff]
#' test_indices <- data_permutation_order[(train_cutoff + 1):nrow(iris)]
#' train_data_unnorm <- iris[train_indices, ]
#' test_data_unnorm <- iris[test_indices, ]
#'
#' # Normalize data
#' means <- apply(train_data_unnorm[,1:4],2,mean)
#' stddevs <- apply(train_data_unnorm[,1:4],2,sd)
#' train_data_norm <- train_data_unnorm
#' test_data_norm <- test_data_unnorm
#'
#' for (col in 1:4) {
#'   train_data_norm[, col] <- (train_data_unnorm[, col] -
#'     means[col]) / stddevs[col]
#'   test_data_norm[, col] <- (test_data_unnorm[, col] -
#'     means[col]) / stddevs[col]
#' }
#'
#' model <- kernel_fda(
#'   Species ~ .,
#'   train_data_norm,
#'   kerneldot = kernlab::rbfdot,
#'   lambda = 1,
#'   sigma = 0.1
#' )
#'
#' test_model(model)
#' test_model(model,test_data_norm)
test_model <- function(
    model_obj,
    newdata,
    print_results = TRUE
) {
    if (missing(newdata)) {
        conf_matrix <- model_obj$confusion
        correct <- sum(diag(conf_matrix))
        total <- sum(conf_matrix)
    } else {
        y_name <- as.character(model_obj$terms)[2]
        predictions <- predict(model_obj, newdata)
        y <- newdata[, y_name]
        correct <- sum(
            as.character(predictions) == as.character(y)
        )
        total <- length(predictions)
    }
    misclass <- total - correct
    if (print_results) {
        cat("\n")
        cat(sprintf("Total Classifications: %i\n", total))
        cat(sprintf("Correct: %i\n", correct))
        cat(sprintf("Misclass: %i\n", misclass))
        cat(sprintf("Classification rate: %1.2f\n", correct / total))
    }

    return(correct / total)
}
