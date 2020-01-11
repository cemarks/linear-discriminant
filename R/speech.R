library(mda)

d <- read.csv("/Users/cemarks/Documents/modulation/code/speech.csv",header=FALSE)

tng<-d[which(d$V1==0),4:13]
Y <- d[which(d$V1==0),14]
Y_dummy <- t(sapply(1:length(Y),function(x) {z <- rep(0,11); z[Y[x]+1]<-1; return(z)}))
tng<-cbind(rep(1,nrow(tng)),tng)

f<-fda(Y~.,data=tng)

f <- fda(V14~.,
	data=d[which(d$V1==0),4:14],
	method=l,
	)


f$values
f$theta

pp <- polyreg(d[which(d$V1==0),4:13],d[which(d$V1==0),14])

Y_hat <- predict(pp)
D_pi <- diag(colSums(Y_dummy)/nrow(Y_dummy))

#  t(f$theta) %*% D_pi %*% f$theta
dot_product <- sum(Y*Y_hat)

tng <- data.frame(x1=c(1,0,0.5,0.33),x2=c(1,1,1.5,0),y=c(0,1,1,0))
f <- fda(y~.,data=tng)

l <- function(x,y,w){return(lm(y~x))}



####### Kernel regression

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
			dist[i,j]=sqrt(t(d) %*% d)
		}
	}
	return(dist/bw)
}


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
	K_l_inv <- chol2inv(chol(K + diag(lambda,nrow=nrow(K))))
	y_hat <- K %*% K_l_inv %*% y
	# y_hat <- as.numeric(y_hat[,1])
	model <- structure(
		list(
			x=x,
			y=y,
			# K=K,
			K_l_inv=K_l_inv,
			lambda=lambda,
			kernel_fn=kernel_function,
			fitted.values=y_hat
		),
		class="kernel_reg_model")
	return(model)
}



predict.kernel_reg_model <- function(modelObj,newdata){
	if(missing(newdata)){
		return(modelObj$fitted.values)
	} else {
		K_new <- modelObj$kernel_fn(newdata,modelObj$x)
		K_l_inv <- modelObj$K_l_inv

		return(K_new %*% K_l_inv %*% modelObj$y)
	}
}


f <- fda(V14~.,
	data=d[which(d$V1==0),4:14],
	method=kernel_reg,
	kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,0.1)),
	lambda = 0.000001
	)


test_data <- d[which(d$V1==1),4:13]
test_Y <- d[which(d$V1==1),14]
output <- data.frame(matrix(nrow=0,ncol=2))
names(output) <- c("lambda","misclass.rate")
for(ll in -2:2){
	l <- 10^ll
	f <- fda(V14~.,
		data=d[which(d$V1==0),4:14],
		method=kernel_reg,
		kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,0.5)),
		lambda = l
	)
	p_test <- predict(f,test_data)
	misclass_count <- length(which(as.character(p_test) != as.character(test_Y)))
	output<-rbind(output,data.frame(lambda=l,misclass.rate=misclass_count/nrow(test_data),stringsAsFactors=FALSE))
}

################ CAN IT BE DONE WITH THE WAVE DATA?

