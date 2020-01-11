#### New kernel regression file for big kernel regressions

#### Requirements

## (1) Function to compute and store large kernel matrix
### (a) Need to determine X(kernel) (NK x p) and X(training) (NT x p)
###     Kernel matrix is NT x NK.  NK is the number of features in the regression model

## (2) Need data structure, storage, & stream for storage of kernel matrix (SQL or data.frame???)

## (3) Need streaming fit linear regression method.
### (a) Preferably handles multiple output vectors, but might not need to.
### (b) biglm or homemade (sgd?)
### (c) Needs a predict method.

library(biglm)
library(MASS)
# library(nloptr)
library(kernlab)

logloss<-function(predict.probs,Y){
	return(-1/nrow(predict.probs)*sum(sapply(1:nrow(predict.probs),function(i) return(log(predict.probs[i,Y[i]])))))
}

score <- function(predict.probs,Y){
	ll <- logloss(predict.probs,Y)
	return(100/(1+ll))
}


kernel.biglm <- function(X,y,w=1,kernel_function,x.kernel=NULL){
	if (is.null(x.kernel)){
		x.kernel <- X
	}
	# X_train <- compute_K_df(X,x.kernel,kernel_function)
	nms_x <- paste("V",1:nrow(x.kernel),sep="")
	y <- as.data.frame(y)
	K_ <- ncol(y)
	nms_y <- paste("y",0:(K_-1),sep="")
	names(y) <- nms_y
	X_train <- compute_K_df(X,x.kernel,kernel_function)
	names(X_train) <- nms_x
	X_train <-cbind(X_train,y)
	model.list <- list()
	y_hat <- matrix(nrow=nrow(X),ncol = K_)
	for(i in 1:K_){
		f <- formula(paste(nms_y[i]," ~ 0 + ",paste(nms_x,sep="",collapse="+"),sep=""))
		l.bignormal <- biglm(f,X_train)
		model.list[[nms_y[i]]] <- l.bignormal
		y_hat[,i] <- predict(l.bignormal,X_train)
	}
	model <- structure(
	list(
		x = X,
		y = y,
		x.kernel=x.kernel,
		kernel_function = kernel_function,
		K_ = K_,
		big.lm = model.list,
		fitted.values=y_hat
	),
	class="big_kernel_lm")
	return(model)
}

# b.lm <- kernel.biglm(X_tr,Y_tr,0,polydot(deg=1),X_tr[1:20,])
# modelObj <- b.lm
# str(modelObj$big.lm[[1]])
# summary(modelObj$big.lm[[1]])

# X <- X_tr
# y <- Y_tr
# w <- 1
# kernel_function <- polydot(deg=1)
# x.kernel <- X_tr[1:20,]


# kernel.biglm.func <- function(X,y,w=1,kernel_function,x.kernel=NULL){
# 	if (is.null(x.kernel)){
# 		x.kernel <- X
# 	}
# 	# X_train <- compute_K_df(X,x.kernel,kernel_function)
# 	nms_x <- paste("V",1:nrow(x.kernel),sep="")
# 	y <- as.data.frame(y)
# 	K_ <- ncol(y)
# 	nms_y <- paste("y",0:(K_-1),sep="")
# 	names(y) <- nms_y
# 	data_function_index <- 1
# 	data_function <- function(reset){
# 		if(reset){
# 			data_function_index <- 1
# 		}
# 		if(data_function_index > nrow(X)){
# 			return(NULL)
# 		}
# 		end.index <- min(data_function_index+5000,nrow(X))
# 		KM <- compute_K_df(X[data_function_index:end.index,],x.kernel,kernel_function)
# 		KM <- cbind(KM,y[data_function_index:end.index])
# 		names(KM) <- c(nms_x,nms_y)
# 		data_function_index <- end.index+1
# 		return(KM)
# 	}
# 	model.list <- list()
# 	y_hat <- matrix(nrow=nrow(X),ncol = K_)
# 	for(i in 1:K_){
# 		f <- formula(paste(nms_y[i]," ~ 0 + ",paste(nms_x,sep="",collapse="+"),sep=""))
# 		l.bignormal <- biglm(f,data_function)
# 		model.list[[nms_y[i]]] <- l.bignormal
# 		y_hat[,i] <- predict(l.bignormal,cbind(X,y[,nms_y[i]]))
# 	}
# 	model <- structure(
# 	list(
# 		x = X,
# 		y = y,
# 		x.kernel=x.kernel,
# 		kernel_function = kernel_function,
# 		K_ = K_,
# 		big.lm = model.list,
# 		fitted.values=y_hat
# 	),
# 	class="big_kernel_lm")
# 	return(model)
# }

predict.big_kernel_lm <- function(modelObj,newdata=NULL){
	if(is.null(newdata)){
		newdata = modelObj$x
	}
	X_fit <- compute_K_df(newdata,modelObj$x.kernel,modelObj$kernel_function)
	names(X_fit)<-paste("V",1:ncol(X_fit),sep="")
	X_fit <- cbind(X_fit,rep(0,nrow(X_fit)))
	p <- matrix(ncol=modelObj$K_,nrow=nrow(X_fit))
	for(i in 1:modelObj$K_){
		ynme <- paste("y",i-1,sep="")
		names(X_fit)[ncol(X_fit)] <- paste("y",i-1,sep="")
		p[,i] <- predict(modelObj$big.lm[[ynme]],X_fit)
	}
	return(p)
}

# predict(b.lm,X_test)

# l.bigkernel <- lm(Y_tr ~ X_new -1)
# pred.out <- cbind(pred.out,predict(l.bigkernel,X_test_new))



compute_K_df <- function(X,x,kernel_function,n=5000){
	#X_train <- kernelMatrix(kernel=kernel_function,x=as.matrix(X[1:min(n,nrow(X)),]),y=as.matrix(x))
	#current_ind <- nrow(X_train)
	# while(nrow(X_train) < nrow(X)){
	# 	X_train <- rbind(
	# 		X_train,
	# 		kernelMatrix(kernel=kernel_function,x=as.matrix(X[(current_ind+1):min((current_ind+n),nrow(X)),]),y=as.matrix(x))
	# 		)
	# 	current_ind <- nrow(X_train)
	# 	cat(sprintf("%i/%i",current_ind,nrow_X))
	# }
	X_train <- kernelMatrix(kernel=kernel_function,x=as.matrix(X),y=as.matrix(x))
	X_train <- as.data.frame(X_train)
	return(X_train)
}

# compute_K_df(X_tr,X_tr[1:10,],vanilladot(),kernel_param)






# d_train <- read.csv("/home/cemarks/Projects/modulation/code/newtrain.csv",header=FALSE)

# linear.kernel <- function(kernel_param=0){
# 	return(vanilladot)
# }


# X <- d_train[,1:(ncol(d_train)-1)]
# X <- cbind(rep(1,nrow(X)),X)
# names(X)[1] <- 'const'


# Y <- d_train[,ncol(d_train)]

# Y_m <- matrix(0,nrow=nrow(d_train),ncol=length(unique(Y)))
# for(i in 1:nrow(Y_m)){
# 	Y_m[i,Y[i]+1] <- 1
# }

# t.inds <- sample(1:nrow(X),104)

# X_tr <- X[t.inds,]
# Y_tr <- Y_m[t.inds,]

# test.inds <- sample(1:nrow(X),5)
# X_test <- X[test.inds,]
# Y_test <- Y_m[test.inds,1]

# X_test_biglm <- cbind(X_test,rep(0,nrow(X_test)))
# names(X_test_biglm)[ncol(X_test_biglm)] <- 'response'

# xdf <- cbind(X_tr,Y_tr)
# names(xdf)[ncol(xdf)] <- "response"

# #### Normal linear model

# l.normal <- lm(response ~ . -1,data=xdf)
# pred.out <- matrix(predict(l.normal,X_test),ncol=1)

# #### Big linear model

# f <- formula(paste("response ~ 0 + ",paste(names(xdf)[1:(ncol(xdf)-1)],collapse="+"),sep=""))
# l.bignormal <- biglm(f,xdf)
# pred.out <- cbind(pred.out,predict(l.bignormal,X_test_biglm,type='response'))

# #### Kernel, manual

# K <- kernelMatrix(kernel=vanilladot(),x=as.matrix(X_tr),y=as.matrix(X_tr))
# alpha <- solve(K,Y_tr[,1:2])
# K_new <- linear.kernel(as.matrix(X_test),as.matrix(X_tr))
# pred.out <- K_new %*% alpha


# #### Kernel linear model
# ######################## NEED TO REWORK THESE TO MEET DATA TYPE CONSTRAINTS OF METHODS

# # X_new <- linear.kernel(as.matrix(X_tr),as.matrix(X_tr))
# # X_test_new <- linear.kernel(as.matrix(X_test),as.matrix(X_tr))
# # l.kernel <- lm(Y_tr ~ X_new -1)
# # pred.out <- cbind(pred.out,predict(l.kernel,X_test_new))

# #### Kernel big model













