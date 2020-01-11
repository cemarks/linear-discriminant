####### Kernel regression
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

renorm <- function(predict.probs,deg=5){
	return(t(apply(predict.probs,1,function(x) return(x^deg/(sum(x^deg))))))
}


linear_kernel <- function(x1,x2){
	if(class(x1) != 'matrix'){
		if (!(is.null(dim(x1)))){
			x1 <- as.matrix(x1)
		}else{
			x1 <- matrix(x1,nrow=1)
		}	
	}
	if(class(x2) != 'matrix'){
		if (!(is.null(dim(x2)))){
			x2 <- as.matrix(x2)
		}else{
			x2 <- matrix(x2,nrow=1)
		}	
	}
	if(any(x1[,1] != 1)){
		x1 <- cbind(rep(1,nrow(x1)),x1)
	}
	if(any(x2[,1] != 1)){
		x2 <- cbind(rep(1,nrow(x2)),x2)
	}
	return(x1 %*% t(x2))
}

poly_kernel <- function(x1,x2,degree){
	if(class(x1) != 'matrix'){
		if (!(is.null(dim(x1)))){
			x1 <- as.matrix(x1)
		}else{
			x1 <- matrix(x1,nrow=1)
		}	
	}
	if(class(x2) != 'matrix'){
		if (!(is.null(dim(x2)))){
			x2 <- as.matrix(x2)
		}else{
			x2 <- matrix(x2,nrow=1)
		}	
	}
	return(((x1 %*% t(x2)) + 1)^degree)
}

poly_kernel_kernlab <- function(deg) return(polydot(degree=deg))

rbf_kernel <- function(x1,x2,bw){
	if(class(x1) != 'matrix'){
		if (!(is.null(dim(x1)))){
			x1 <- as.matrix(x1)
		}else{
			x1 <- matrix(x1,nrow=1)
		}	
	}
	if(class(x2) != 'matrix'){
		if (!(is.null(dim(x2)))){
			x2 <- as.matrix(x2)
		}else{
			x2 <- matrix(x2,nrow=1)
		}	
	}
	dist <- matrix(nrow=nrow(x1),ncol=nrow(x2))
	for (i in 1:nrow(x1)){
		for (j in 1:nrow(x2)){
			d <- x1[i,] - x2[j,]
			dist[i,j]=(t(d) %*% d)
		}
	}
	return(exp(-dist/(2*bw^2)))
}

rbf_kernel_kernlab <- function(sig) return(rbfdot(sigma=1/(2*sig^2)))


kernel_reg_expanded <- function(x,y,w=1,kernel_function=NULL,lambda=1){
	if (is.null(kernel_function)){
		kernel_function <- linear_kernel
	}
	K <- kernel_function(x,x)
	if(is.null(dim(y)) || ncol(y)==1){
		if(!(is.null(dim(y)))){
			y <- as.numeric(as.character(y[,1]))
		}
		Y_dummy <- matrix(0,nrow=nrow(x),ncol=length(unique(y)))
		uq <- unique(y)
		for (i in 1:length(uq)){
			Y_dummy[which(y==uq[i]),i]<-1
		}
	} else {
		Y_dummy <- y
	}
	y_hat <- K %*% solve(K + diag(lambda,nrow=nrow(K))) %*% Y_dummy
	model <- structure(list(x=x,y=y,lambda=lambda,fit.values=y_hat),
		class="kernel_reg_model")
	return(model)
}

kernel_reg <- function(x,y,w=1,kernel_function=NULL,lambda=1){
	if (is.null(kernel_function)){
		kernel_function <- linear_kernel
	}
	K <- kernel_function(x,x)
	K_l_inv <- solve(K + diag(lambda,nrow=nrow(K)))
	alpha <- K_l_inv %*% y
	y_hat <- K %*% alpha
	# y_hat <- as.numeric(y_hat[,1])
	model <- structure(
		list(
			x=x,
			y=y,
			alpha=alpha,
			lambda=lambda,
			kernel_fn=kernel_function,
			fitted.values=y_hat
		),
		class="kernel_reg_model")
	return(model)
}

kernel_reg_fast <- function(x,y,w=1,kernel_function=NULL,lambda=1){
	if (is.null(kernel_function)){
		kernel_function <- linear_kernel
	}
	K <- kernel_function(x,x)
	alpha <- solve(K + diag(lambda,nrow=nrow(K)),y)
	y_hat <- K %*% alpha
	model <- structure(
		list(
			x=x,
			y=y,
			alpha=alpha,
			lambda=lambda,
			kernel_fn=kernel_function,
			fitted.values=y_hat
		),
		class="kernel_reg_model")
	return(model)
}

kernel_reg_fast_kernlab <- function(x,y,w=1,kernel_function=NULL,lambda=1){
	if (is.null(kernel_function)){
		kernel_function <- linear_kernel
	}
	K <- kernelMatrix(kernel=kernel_function,x=as.matrix(x))
	alpha <- solve(K + diag(lambda,nrow=nrow(K)),y)
	y_hat <- K %*% alpha
	model <- structure(
		list(
			x=x,
			y=y,
			alpha=alpha,
			lambda=lambda,
			kernel_fn=kernel_function,
			fitted.values=y_hat
		),
		class="kernel_reg_model_kernlab")
	return(model)
}

kernel_reg_sqdist <- function(x,y,w=1,sigma=NULL,lambda=1){
	K <- -log(kernelMatrix(kernel=rbf_kernel_kernlab(sigma),x=as.matrix(x)))# *2*sigma^2
	alpha <- solve(K + diag(lambda,nrow=nrow(K)),y)
	y_hat <- K %*% alpha
	model <- structure(
		list(
			x=x,
			y=y,
			alpha=alpha,
			lambda=lambda,
			sigma=sigma,
			fitted.values=y_hat
		),
		class="kernel_reg_sqdist")
	return(model)
}



predict.kernel_reg_model <- function(modelObj,newdata){
	if(missing(newdata)){
		return(modelObj$fitted.values)
	} else {
		K_new <- modelObj$kernel_fn(newdata,modelObj$x)
		alpha <- modelObj$alpha
		return(K_new %*% alpha)
	}
}

predict.kernel_reg_model <- function(modelObj,newdata){
	if(missing(newdata)){
		return(modelObj$fitted.values)
	} else {
		K_new <- modelObj$kernel_fn(newdata,modelObj$x)
		alpha <- modelObj$alpha
		return(K_new %*% alpha)
	}
}

predict.kernel_reg_model_kernlab <- function(modelObj,newdata){
	if(missing(newdata)){
		return(modelObj$fitted.values)
	} else {
		K_new <- kernelMatrix(kernel=modelObj$kernel_fn,as.matrix(newdata),as.matrix(modelObj$x))
		alpha <- modelObj$alpha
		return(K_new %*% alpha)
	}
}

predict.kernel_reg_sqdist <- function(modelObj,newdata){
	if(missing(newdata)){
		return(modelObj$fitted.values)
	} else {
		K_new <- -log(kernelMatrix(kernel=rbf_kernel_kernlab(modelObj$sigma),as.matrix(newdata),as.matrix(modelObj$x)))# *2*modelObj$sigma^2
		alpha <- modelObj$alpha
		return(K_new %*% alpha)
	}
}

return_next <- function(low.mid.high,score.vector){
	low.score<-score.vector[1]
	mid.score <- score.vector[2]
	high.score <- score.vector[3]
	low <-low.mid.high[1]
	mid <- low.mid.high[2]
	high <- low.mid.high[3]
	if(high.score > mid.score && mid.score > low.score){
		return(c(low-(mid-low),low,mid))
	} else if(high.score > mid.score && low.score > mid.score){
		return(c((low+mid)/2,mid,(high+mid)/2))
	} else if(low.score > mid.score && mid.score > high.score){
		return(c(mid,high,high+(high-mid)))
	} else if (mid.score > high.score && mid.score > low.score){
		return(c(low,high,high+(low-high)))
	} else if(low.score==1 && mid.score==1){
		return(c(high,high+(high-mid),high+2*(high-mid)))
	} else if (high.score==1 && mid.score==1){
		return(c(low-2*(mid-low),low-(mid-low),low))
	} else {
		return(c(mid,mid,mid))
	}
}







