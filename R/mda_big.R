
library(mda)


# if(Sys.info()['sysname']=="Linux"){
# 	source("/home/cemarks/Projects/modulation/code/kernel_reg_functions.R")
# 	d_train <- read.csv("/home/cemarks/Projects/modulation/code/train.csv",header=FALSE)
# 	d_val <- read.csv("/home/cemarks/Projects/modulation/code/val.csv",header=FALSE)
# 	d_test <- read.csv("/home/cemarks/Projects/modulation/code/test.csv",header=FALSE)
# } else {
# 	source("/Users/cemarks/Documents/modulation/code/kernel_reg_functions.R")
# 	d_train <- read.csv("/Users/cemarks/Documents/modulation/code/train.csv",header=FALSE)
# 	d_val <- read.csv("/Users/cemarks/Documents/modulation/code/val.csv",header=FALSE)
# 	d_test <- read.csv("/Users/cemarks/Documents/modulation/code/test.csv",header=FALSE)
# }
if(Sys.info()['sysname']=="Darwin"){
	source("/Users/cemarks/Documents/modulation/code/big_kernel_regression.R")
	X_tng <- read.csv("/Users/cemarks/Documents/modulation/code/train.csv",header=FALSE)
	X_val <- read.csv("/Users/cemarks/Documents/modulation/code/val.csv",header=FALSE)
	x.kernel <- read.csv("/Users/cemarks/Documents/modulation/code/kern.csv",header=FALSE)
}else{
	source("/home/cemarks/Projects/modulation/code/big_kernel_regression.R")
	X_tng <- read.csv("/home/cemarks/Projects/modulation/code/train.csv",header=FALSE)
	X_val <- read.csv("/home/cemarks/Projects/modulation/code/val.csv",header=FALSE)
	x.kernel <- read.csv("/home/cemarks/Projects/modulation/code/kern.csv",header=FALSE)
}

X_tng <- X_tng[sample(nrow(X_tng)),]

# kernel.size <- 200

# kernel.inds <- sample(1:nrow(d_train),kernel.size)
# x.kernel <- d_train[kernel.inds,1:(ncol(d_train)-1)]
# X_tng <- d_train[setdiff(1:nrow(d_train),kernel.inds),]


last.name <- names(X_tng)[ncol(X_tng)]
form <- as.formula(paste(last.name,".",sep="~"))

# t <- Sys.time()
# f <- fda(form,
# 	data=rbind(d_train,d_val,d_test),
# 	method=kernel_reg,
# 	kernel_function=function(x1,x2) return(poly_kernel(x1,x2,3)),
# 	lambda=1000000000000
# 	)
# total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
# f

# t <- Sys.time()
# f.fast <- fda(form,
# 	data=rbind(d_train),
# 	method=kernel_reg_fast,
# 	kernel_function=function(x1,x2) return(poly_kernel(x1,x2,3)),
# 	lambda=100000000000
# 	)
# total.time.fast <- as.numeric(difftime(Sys.time(),t,units="secs"))
# f.fast


# d_tng2 <- d_train[sample(nrow(d_train)),]
# p <- predict(f,d_tng2[,1:(ncol(d_val-1))])
# tng2.misclass <- length(which(p != d_tng2[,ncol(d_val)]))
# tng2.misclass
# tng2.misclass/nrow(d_tng2)



# p <- predict(f.fast,d_val[,1:(ncol(d_val-1))])
# val.misclass <- length(which(p != d_val[,ncol(d_val)]))
# val.misclass/nrow(d_val)


out.d <- data.frame(matrix(nrow=0,ncol=9),stringsAsFactors=FALSE)
names(out.d)<-c("kernel","kernel_param","train.misclass","val.misclass","tng.nrow","tng.ncol","fit.time","tng.score","score")
Y_val <- as.character(X_val[,ncol(X_val)])
Y_tng <- as.character(X_tng[,ncol(X_tng)])

s <- "poly"
deg <- 2

# out.d <- read.csv("/home/cemarks/Projects/modulation/code/init_results.csv")
for(s in c("poly","rbf")){
	if(s=="poly"){
		for(deg in 1:4){
			# t <- Sys.time()
			# f <- fda(form,
			# 	data=d_train[1:1000,],
			# 	method=kernel_reg_fast,
			# 	kernel_function=function(x1,x2) return(poly_kernel(x1,x2,deg)),
			# 	lambda=la
			# 	)
			# total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
			# p <- predict(f,d_val[,1:(ncol(d_val-1))])
			# val.misclass <- length(which(p != d_val[,ncol(d_val)]))
			# vmc <- val.misclass/nrow(d_val)
			# cat(sprintf("%s: deg=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,deg,la,nrow(d_train),ncol(d_train),total.time,vmc))
			t <- Sys.time()
			f <- fda(form,
				data=X_tng,
				method=kernel.biglm,
				kernel_function=polydot(deg),
				x.kernel = x.kernel
				)
			f <- fda(form,
				data=X_tng,
				method=polyreg,
				)
			total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
			p <- predict(f,d_val[,1:(ncol(d_val)-1)])
			val.misclass <- length(which(p != d_val[,ncol(d_val)]))
			ptng <- predict(f,d_train[,1:(ncol(d_train)-1)])
			tng.misclass <- length(which(ptng != d_train[,ncol(d_train)]))
			tmc <- tng.misclass/nrow(d_train)
			vmc <- val.misclass/nrow(d_val)
			pp <- predict(f,d_val[,1:(ncol(d_val)-1)],type="posterior");
			sr <- score(pp,Y_val)
			ppt <- predict(f,d_train[,1:(ncol(d_train)-1)],type="posterior");
			srt <- score(pp,Y_tng)
			out.d<-rbind(out.d,
				data.frame(kernel=s,
					kernel_param=deg,
					val.misclass=vmc,
					train.misclass=tmc,
					tng.nrow=nrow(d_train),
					tng.ncol=ncol(d_train),
					fit.time=total.time),
				    tng.score=srt,
				    score=sr,
				stringsAsFactors=FALSE)
			cat(sprintf("%s: deg=%1.2f, tng size=%i x %i, time = %1.1f s, tmc: %0.2f vmc: %0.2f, tng.score: %1.2f, score: %1.2f\n",s,deg,nrow(d_train),ncol(d_train),total.time,tmc,vmc,srt,sr))
		}
	} else if(s=="rbf") {
		# for(bw in 1){
			# for(l in 10^0){
		for(bw in c(1,5,10,20)){
			for(la in 10^((-6):(-1))){
				# t <- Sys.time()
				# f <- fda(form,
				# 	data=d_train[1:1000,],
				# 	method=kernel_reg_fast,
				# 	kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
				# 	# kernel_function=rbf_kernel_kernlab(bw),
				# 	lambda=la
				# 	)
				# total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
				# p <- predict(f,d_val[,1:(ncol(d_val-1))])
				# val.misclass <- length(which(p != d_val[,ncol(d_val)]))
				# vmc <- val.misclass/nrow(d_val)
				# cat(sprintf("%s: bw=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,vmc))
				t <- Sys.time()
				f <- fda(form,
					data=X_tng,
					method=kernel_reg_fast_kernlab,
					# kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
					kernel_function=polydot(deg=1)
					)
				total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
				p <- predict(f,d_val[,1:(ncol(d_val)-1)])
				pp <- predict(f,d_val[,1:(ncol(d_train)-1)],type="posterior");
				ptng <- predict(f,d_train[,1:(ncol(d_val-1))])
				tng.misclass <- length(which(ptng != d_train[,ncol(d_train)]))
				tmc <- tng.misclass/nrow(d_train)
				sr <- score(pp,Y_val)
				ppt <- predict(f,d_train[,1:(ncol(d_train)-1)],type="posterior");
				srt <- score(pp,Y_tng)
				val.misclass <- length(which(p != d_val[,ncol(d_val)]))
				vmc <- val.misclass/nrow(d_val)
				out.d<-rbind(out.d,
					data.frame(kernel=s,
						kernel_param=bw,
						lambda=la,
						val.misclass=vmc,
						train.misclass=tmc,
						tng.nrow=nrow(d_train),
						tng.ncol=ncol(d_train),
						fit.time=total.time),
					    tng.score=srt,
					    score=sr,
					stringsAsFactors=FALSE)
				cat(sprintf("%s: bw=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, tmc: %0.2f, vmc: %0.2f, tng score: %1.2f, score: %1.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,tmc,vmc,srt,sr))
			}
		}
	}
}

write.csv(out.d,"/home/cemarks/Projects/modulation/code/init_results.csv",row.names=FALSE)

# bw<-50
# la <- 50
# t.old <- Sys.time()
# f.old <- fda(form,
# 	data=d_train[1:1000,],
# 	method=kernel_reg_fast,
# 	kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
# 	# kernel_function=rbf_kernel_kernlab(sqrt(bw)),
# 	lambda=la
# 	)
# total.time <- as.numeric(difftime(Sys.time(),t.old,units="secs"))
# p.old <- predict(f.old,d_val[,1:(ncol(d_val-1))])
# val.misclass.old <- length(which(p.old != d_val[,ncol(d_val)]))
# vmc.old <- val.misclass.old/nrow(d_val)
# f.old
# vmc.old


# t <- Sys.time()
# f <- fda(form,
# 	data=d_train[1:1000,],
# 	method=kernel_reg_fast_kernlab,
# 	# kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
# 	kernel_function=rbf_kernel_kernlab(bw),
# 	lambda=la
# 	)
# total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
# p <- predict(f,d_val[,1:(ncol(d_val-1))])
# val.misclass <- length(which(p != d_val[,ncol(d_val)]))
# vmc <- val.misclass/nrow(d_val)
# f
# vmc


for(s in c("poly","rbf")){
	if(s=="poly"){
		for(deg in 1:4){
			la <- 10^(0:18)
			high <- length(la)
			low  <- 1
			la  <- la[low]
			t <- Sys.time()
			f <- fda(form,
				data=d_train,
				method=kernel_reg_fast_kernlab,
				kernel_function=poly_kernel_kernlab(deg),
				lambda=la
				)
			total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
			p <- predict(f,d_val[,1:(ncol(d_val-1))])
			val.misclass <- length(which(p != d_val[,ncol(d_val)]))
			vmc <- val.misclass/nrow(d_val)
			out.d<-rbind(out.d,
				data.frame(kernel=s,
					kernel_param=deg,
					lambda=la,
					val.misclass=vmc,
					tng.nrow=nrow(d_train),
					tng.ncol=ncol(d_train),
					fit.time=total.time),
				stringsAsFactors=FALSE)
			
			cat(sprintf("%s: deg=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,deg,la,nrow(d_train),ncol(d_train),total.time,vmc))
			}
		}
	} else if(s=="rbf") {
		# for(bw in 1){
			# for(l in 10^0){
		for(bw in c(10,20,40,60,100)){
			for(la in 10^((-3):2)){
				# t <- Sys.time()
				# f <- fda(form,
				# 	data=d_train[1:1000,],
				# 	method=kernel_reg_fast,
				# 	kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
				# 	# kernel_function=rbf_kernel_kernlab(bw),
				# 	lambda=la
				# 	)
				# total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
				# p <- predict(f,d_val[,1:(ncol(d_val-1))])
				# val.misclass <- length(which(p != d_val[,ncol(d_val)]))
				# vmc <- val.misclass/nrow(d_val)
				# cat(sprintf("%s: bw=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,vmc))
				t <- Sys.time()
				f <- fda(form,
					data=d_train,
					method=kernel_reg_fast_kernlab,
					# kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
					kernel_function=rbf_kernel_kernlab(bw),
					lambda=la
					)
				total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
				p <- predict(f,d_val[,1:(ncol(d_val-1))])
				val.misclass <- length(which(p != d_val[,ncol(d_val)]))
				vmc <- val.misclass/nrow(d_val)
				out.d<-rbind(out.d,
					data.frame(kernel=s,
						kernel_param=bw,
						lambda=la,
						val.misclass=vmc,
						tng.nrow=nrow(d_train),
						tng.ncol=ncol(d_train),
						fit.time=total.time),
					stringsAsFactors=FALSE)
				cat(sprintf("%s: bw=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,vmc))
			}
		}
	}
}


for(s in c("rbf")){
	if(s=="poly"){
		for(deg in 5:8){
			for(la in 10^(12:20)){
				# t <- Sys.time()
				# f <- fda(form,
				# 	data=d_train[1:1000,],
				# 	method=kernel_reg_fast,
				# 	kernel_function=function(x1,x2) return(poly_kernel(x1,x2,deg)),
				# 	lambda=la
				# 	)
				# total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
				# p <- predict(f,d_val[,1:(ncol(d_val-1))])
				# val.misclass <- length(which(p != d_val[,ncol(d_val)]))
				# vmc <- val.misclass/nrow(d_val)
				# cat(sprintf("%s: deg=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,deg,la,nrow(d_train),ncol(d_train),total.time,vmc))
				t <- Sys.time()
				f <- fda(form,
					data=d_train,
					method=kernel_reg_fast_kernlab,
					kernel_function=poly_kernel_kernlab(deg),
					lambda=la
					)
				total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
				p <- predict(f,d_val[,1:(ncol(d_val-1))])
				val.misclass <- length(which(p != d_val[,ncol(d_val)]))
				vmc <- val.misclass/nrow(d_val)
				out.d<-rbind(out.d,
					data.frame(kernel=s,
						kernel_param=deg,
						lambda=la,
						val.misclass=vmc,
						tng.nrow=nrow(d_train),
						tng.ncol=ncol(d_train),
						fit.time=total.time),
					stringsAsFactors=FALSE)
				cat(sprintf("%s: deg=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,deg,la,nrow(d_train),ncol(d_train),total.time,vmc))
			}
		}
	} else if(s=="rbf") {
		# for(bw in 1){
			# for(l in 10^0){
		for(bw in c(25,30,35,40)){
			for(la in c(0.0075,0.01,0.0125,0.02,0.0275)){
				# t <- Sys.time()
				# f <- fda(form,
				# 	data=d_train[1:1000,],
				# 	method=kernel_reg_fast,
				# 	kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
				# 	# kernel_function=rbf_kernel_kernlab(bw),
				# 	lambda=la
				# 	)
				# total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
				# p <- predict(f,d_val[,1:(ncol(d_val-1))])
				# val.misclass <- length(which(p != d_val[,ncol(d_val)]))
				# vmc <- val.misclass/nrow(d_val)
				# cat(sprintf("%s: bw=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,vmc))
				t <- Sys.time()
				f <- fda(form,
					data=d_train,
					method=kernel_reg_fast_kernlab,
					# kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
					kernel_function=rbf_kernel_kernlab(bw),
					lambda=la
					)
				total.time <- as.numeric(difftime(Sys.time(),t,units="secs"))
				p <- predict(f,d_val[,1:(ncol(d_val-1))])
				val.misclass <- length(which(p != d_val[,ncol(d_val)]))
				vmc <- val.misclass/nrow(d_val)
				out.d<-rbind(out.d,
					data.frame(kernel=s,
						kernel_param=bw,
						lambda=la,
						val.misclass=vmc,
						tng.nrow=nrow(d_train),
						tng.ncol=ncol(d_train),
						fit.time=total.time),
					stringsAsFactors=FALSE)
				cat(sprintf("%s: bw=%1.2f, lambda=%1.3f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,vmc))
			}
		}
	}
}


