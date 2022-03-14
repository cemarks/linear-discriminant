#!/usr/bin/Rscript

#' @author Christopher E. Marks, \email{cemarks@@alum.mit.edu}
#' @references \url{https://github.com/cemarks/linear-discriminant}

source("linear_discriminant.R")
vowel_train <- read.csv("../data/vowel_train.csv")
vowel_test <- read.csv("../data/vowel_test.csv")

vowel_train <- vowel_train[, which(names(vowel_train) != "row.names")]
vowel_test <- vowel_test[, which(names(vowel_test) != "row.names")]

## Preprocessing
predictor_vars <- paste("x", 1:10, sep = ".")
train_means <- apply(vowel_train[, predictor_vars], 2, mean)
train_stddevs <- apply(vowel_train[, predictor_vars], 2, sd)

## Helpful functions

normalize_x <- function(
    x,
    means,
    stddevs,
    predictor_vars
) {
    x_norm <- x
    for (pv in predictor_vars) {
        x_norm[, pv] <- (x[, pv] - means[pv]) / stddevs[pv]
    }
    return(x_norm)
}

## Verify linear model works

x <- vowel_train[, grep("x.", names(vowel_train), fixed = TRUE)]
y <- vowel_train[, "y"]
x_test <- vowel_test[, grep("x.", names(vowel_test), fixed = TRUE)]
y_test <- vowel_test[, "y"]

y_one_hot <- as.matrix(mltools::one_hot(
    data.table::as.data.table(as.factor(y))))
krm <- kernel_reg_model(
    x,
    y_one_hot,
    kernel_function = lineardot(),
    lambda = 1
)

train_predictions <- predict(krm)
p <- predict(krm, as.matrix(x_test))

cat("\n")
cat(paste("Train data dimension:", dim(x)[1], "x", dim(x)[2], "\n", sep = " "))
cat(paste("Number of classes: ", length(unique(y)), "\n", sep = ""))
if ((dim(train_predictions)[1] == dim(x)[1]) &&
    (dim(train_predictions)[2] == length(unique(y)))) {
    cat("Training prediction matrix dimensions are correct.\n")
} else {
    cat("Training prediction matrix dimensions are wrong.\n")
    cat(paste("Training predictions dimensions: ",
        dim(train_predictions)[1], " x ", dim(train_predictions)[2],
            "\n", sep = ""))
    cat(paste("Dimensions should be ", dim(x)[1], " x ",
        length(unique(y)), ".\n", sep = ""))
}
cat(paste("Test data dimension:", dim(x_test)[1], "x", dim(x_test)[2],
    "\n", sep = " "))
cat(paste("Predict matrix dimension: ", dim(p)[1], " x ", dim(p)[2],
    "\n", sep = ""))
if ((dim(p)[1] == dim(x_test[1])) && (dim(p)[2] == length(unique(y)))) {
    cat("Test prediction matrix dimensions are correct.\n")
} else {
    cat("Test prediction matrix dimensions are wrong.\n")
    cat(paste("Dimensions should be ", dim(x_test)[1], " x ",
        length(unique(y)), ".\n"))
}


# Verify linear_fda

vowel_train_norm <- normalize_x(
    vowel_train,
    train_means,
    train_stddevs,
    predictor_vars
)

f_linear <- linear_fda(
    y~.,
    vowel_train_norm
)

## Plot on p. 85
plot(f_linear, xcoord = 1, ycoord = 2)
test_model(f_linear)

vowel_test_norm <- normalize_x(
    vowel_test,
    train_means,
    train_stddevs,
    predictor_vars
)

## Plot on p.93.  Interestingly enough, the training data is plotted.
plot(f_linear, vowel_test_norm, 1, 3)
test_model(f_linear, vowel_test_norm)



# Verify kernel_fda with default linear kernel

f_linear2 <- kernel_fda(
    y~.,
    vowel_train_norm,
    lambda = 1,
)
plot(f_linear2, xcoord = 1, ycoord = 2)
test_model(f_linear2)
test_model(f_linear2, vowel_test_norm)



# Verify kernel_fda

f <- kernel_fda(
    y~.,
    vowel_train_norm,
    kerneldot = kernlab::rbfdot,
    lambda = 1,
    sigma = 0.1
)
test_model(f)
test_model(f, vowel_test_norm)





data(iris)

# Normalize data
means <- apply(iris[, 1:4], 2, mean)
stddevs <- apply(iris[, 1:4], 2, sd)
iris_normalized <- iris
for (col in 1:4) {
  iris_normalized[, col] <- (iris[, col] - means[col]) / stddevs[col]
}

model <- linear_fda(
  Species ~ .,
  iris_normalized,
  lambda = 1
)

plot(model)


# Separate into training & test
data_permutation_order <- sample(nrow(iris))
train_cutoff <- round(0.75 * nrow(iris))
train_indices <- data_permutation_order[1:train_cutoff]
test_indices <- data_permutation_order[(train_cutoff + 1):nrow(iris)]
train_data_unnorm <- iris[train_indices, ]
test_data_unnorm <- iris[test_indices, ]

# Normalize data
means <- apply(train_data_unnorm[, 1:4], 2, mean)
stddevs <- apply(train_data_unnorm[, 1:4], 2, sd)
train_data_norm <- train_data_unnorm
test_data_norm <- test_data_unnorm

for (col in 1:4) {
  train_data_norm[, col] <- (train_data_unnorm[, col] -
    means[col]) / stddevs[col]
  test_data_norm[, col] <- (test_data_unnorm[, col] -
    means[col]) / stddevs[col]
}

model <- kernel_fda(
  Species ~ .,
  train_data_norm,
  kerneldot = kernlab::rbfdot,
  lambda = 1,
  sigma = 0.1
)

test_model(model)
test_model(model, test_data_norm)
plot(model, test_data_norm)
