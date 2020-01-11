
library(mda)


if(Sys.info()['sysname']=="Linux"){
	source("/home/cemarks/Projects/modulation/code/kernel_reg_functions.R")
	d_train <- read.csv("/home/cemarks/Projects/modulation/code/train.csv",header=FALSE)
	d_val <- read.csv("/home/cemarks/Projects/modulation/code/val.csv",header=FALSE)
	d_test <- read.csv("/home/cemarks/Projects/modulation/code/test.csv",header=FALSE)
	output.file.name <- ("/home/cemarks/Projects/modulation/code/score_results.csv")
} else {
	source("/Users/cemarks/Documents/modulation/code/kernel_reg_functions.R")
	d_train <- read.csv("/Users/cemarks/Documents/modulation/code/train.csv",header=FALSE)
	d_val <- read.csv("/Users/cemarks/Documents/modulation/code/val.csv",header=FALSE)
	d_test <- read.csv("/Users/cemarks/Documents/modulation/code/test.csv",header=FALSE)
	output.file.name <- ("/Users/cemarks/Documents/modulation/code/score_results.csv")
}

d_train <- d_train[sample(nrow(d_train)),]
Y_val <- as.character(d_val[,ncol(d_val)])

last.name <- names(d_train)[ncol(d_train)]
form <- as.formula(paste(last.name,".",sep="~"))


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


dt <- Sys.time()
f.copy.name <- strsplit(output.file.name,"/")[[1]]
f.original.name.last <- "extract_features.py"
f.original.name <- f.copy.name
f.original.name[length(f.original.name)] <- f.original.name.last
f.original.name <- paste(f.original.name,collapse="/")
# f.copy.name.last <- f.copy.name[length(f.copy.name)]
# f.copy.name.last <- substr(f.copy.name.last,1,nchar(f.copy.name.last)-4)
f.copy.name.last <- sprintf("features_%s.py",format(dt,format="%Y%m%d_%H-%M-%S"))
f.copy.name[length(f.copy.name)] <- "archive"
f.copy.name <- c(f.copy.name,f.copy.name.last)
f.copy.name <- paste(f.copy.name,collapse="/")
file.copy(f.original.name,f.copy.name)

if(file.exists(output.file.name)){
	out.d <- read.csv(output.file.name)
} else {
	out.d<-data.frame(matrix(nrow=0,ncol=9),stringsAsFactors=FALSE)
	names(out.d)<-c("datetime",
			"kernel",
			"kernel_param",
			"lambda",
			"val.misclass",
			"score",
			"tng.nrow",
			"tng.ncol",
			"fit.time")
}


for(s in c("poly","rbf")){
	if(s=="poly"){
		for(deg in 1:4){
			la.low <- 0
			la.mid <- 3
			la.high <- 6
			la.list <- list()
			la.vector <- c(la.low,la.mid,la.high)
			while(la.vector[3]-la.vector[1] > 1){
				w <- which((!(as.character(la.vector) %in% names(la.list))))
				for(i in w){
					la <- 10^(la.vector[i])
					t <- Sys.time()
					vmc <- 1
					sr <- 0
					total.time <- 0
					try({
						f <- fda(form,
							data=d_train,
							#method=kernel_reg_fast_kernlab,
							method=kernel_reg_fast,
							kernel_function=function(x1,x2) return (poly_kernel(x1,x2,deg)),
							lambda=la
							);
						total.time <- as.numeric(difftime(Sys.time(),t,units="secs"));
						p <- predict(f,d_val[,1:(ncol(d_val)-1)]);
						val.misclass <- length(which(p != d_val[,ncol(d_val)]));
						vmc <- val.misclass/nrow(d_val);
						pp <- predict(f,d_val[,1:(ncol(d_val)-1)],type="posterior");
						sr <- score(pp,Y_val)
					})
					out.d<-rbind(out.d,
						data.frame(datetime=format(dt),
							kernel=s,
							kernel_param=deg,
							lambda=la,
							val.misclass=vmc,
							score=sr,
							tng.nrow=nrow(d_train),
							tng.ncol=ncol(d_train),
							fit.time=total.time),
						stringsAsFactors=FALSE)
					cat(sprintf("%s: deg=%1.2f, lambda=%1.4f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f, score: %1.2f\n",s,deg,la,nrow(d_train),ncol(d_train),total.time,vmc,sr))
					if(is.na(sr)) sr <- 0
					la.list[[as.character(la.vector[i])]] <- list(lambda=10^(la.vector[i]),vmc=vmc,score=-sr)
				}
				score.vector <- c(la.list[[as.character(la.vector[1])]][["score"]],
					la.list[[as.character(la.vector[2])]][["score"]],
					la.list[[as.character(la.vector[3])]][["score"]])
				la.vector <- return_next(la.vector,score.vector)
			}
		}
	} else if(s=="rbf") {
		for(bw in c(10,20,40,60,100)){
			la.low <- -4
			la.mid <- -2
			la.high <- 0
			la.list <- list()
			la.vector <- c(la.low,la.mid,la.high)
			while(la.vector[3]-la.vector[1] > 1){
				w <- which((!(as.character(la.vector) %in% names(la.list))))
				for(i in w){
					la <- 10^(la.vector[i])
					t <- Sys.time()
					vmc <- 1
					total.time <- 0
					sr <- 0
					try({
						f <- fda(form,
							data=d_train,
							method=kernel_reg_fast_kernlab,
							# kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
							kernel_function=rbf_kernel_kernlab(bw),
							lambda=la
							);
						total.time <- as.numeric(difftime(Sys.time(),t,units="secs"));
						p <- predict(f,d_val[,1:(ncol(d_val)-1)]);
						val.misclass <- length(which(p != d_val[,ncol(d_val)]));
						vmc <- val.misclass/nrow(d_val)
						pp <- predict(f,d_val[,1:(ncol(d_val)-1)],type="posterior");
						sr <- score(pp,Y_val)
					})
					out.d<-rbind(out.d,
						data.frame(datetime=format(dt),
							kernel=s,
							kernel_param=bw,
							lambda=la,
							val.misclass=vmc,
							score=sr,
							tng.nrow=nrow(d_train),
							tng.ncol=ncol(d_train),
							fit.time=total.time),
						stringsAsFactors=FALSE)
					cat(sprintf("%s: bw=%1.2f, lambda=%1.4f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f, score: %1.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,vmc,sr))
					if(is.na(sr)) sr <- 0
					la.list[[as.character(la.vector[i])]] <- list(lambda=10^(la.vector[i]),vmc=vmc,score=-sr)
				}
				score.vector <- c(la.list[[as.character(la.vector[1])]][["score"]],
					la.list[[as.character(la.vector[2])]][["score"]],
					la.list[[as.character(la.vector[3])]][["score"]])
				la.vector <- return_next(la.vector,score.vector)
			}
		}
	}
}

write.csv(out.d,"/home/cemarks/Projects/modulation/code/score_results.csv",row.names=FALSE)





## Follow ups

# for(s in c("sqdist")){
# 	if(s=="sqdist"){
# 		for(bw in c(1,10,100)){
# 			la.low <- -8
# 			la.mid <- -4
# 			la.high <- 0
# 			la.list <- list()
# 			la.vector <- c(la.low,la.mid,la.high)
# 			while(la.vector[3]-la.vector[1] > 1){
# 				w <- which((!(as.character(la.vector) %in% names(la.list))))
# 				for(i in w){
# 					la <- 10^(la.vector[i])
# 					t <- Sys.time()
# 					vmc <- 1
# 					total.time <- 0
# 					try({
# 						f <- fda(form,
# 							data=d_train,
# 							method=kernel_reg_sqdist,
# 							sigma=bw,
# 							lambda=la
# 							);
# 						total.time <- as.numeric(difftime(Sys.time(),t,units="secs"));
# 						p <- predict(f,d_val[,1:(ncol(d_val-1))]);
# 						val.misclass <- length(which(p != d_val[,ncol(d_val)]));
# 						vmc <- val.misclass/nrow(d_val);
# 					})
# 					out.d<-rbind(out.d,
# 						data.frame(datetime=format(dt),
# 							kernel=s,
# 							kernel_param=bw,
# 							lambda=la,
# 							val.misclass=vmc,
# 							tng.nrow=nrow(d_train),
# 							tng.ncol=ncol(d_train),
# 							fit.time=total.time),
# 						stringsAsFactors=FALSE)
# 					cat(sprintf("%s: sigma=%1.2f, lambda=%1.4f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,vmc))
# 					la.list[[as.character(la.vector[i])]] <- list(lambda=10^(la.vector[i]),vmc=vmc)
# 				}
# 				score.vector <- c(la.list[[as.character(la.vector[1])]][['vmc']],
# 					la.list[[as.character(la.vector[2])]][['vmc']],
# 					la.list[[as.character(la.vector[3])]][['vmc']])
# 				la.vector <- return_next(la.vector,score.vector)
# 			}
# 		}
# 	} else if(s=="rbf") {
# 		for(bw in c(95,100,105,110)){
# 			la.low <- -6
# 			la.mid <- -4
# 			la.high <- -2
# 			la.list <- list()
# 			la.vector <- c(la.low,la.mid,la.high)
# 			while(la.vector[3]-la.vector[1] > 0.5){
# 				w <- which((!(as.character(la.vector) %in% names(la.list))))
# 				for(i in w){
# 					la <- 10^(la.vector[i])
# 					t <- Sys.time()
# 					vmc <- 1
# 					total.time <- 0
# 					try({
# 						f <- fda(form,
# 							data=d_train,
# 							method=kernel_reg_fast_kernlab,
# 							# kernel_function=function(x1,x2) return(rbf_kernel(x1,x2,bw)),
# 							kernel_function=rbf_kernel_kernlab(bw),
# 							lambda=la
# 							);
# 						total.time <- as.numeric(difftime(Sys.time(),t,units="secs"));
# 						p <- predict(f,d_val[,1:(ncol(d_val-1))]);
# 						val.misclass <- length(which(p != d_val[,ncol(d_val)]));
# 						vmc <- val.misclass/nrow(d_val)
# 					})
# 					out.d<-rbind(out.d,
# 						data.frame(datetime=format(dt),
# 							kernel=s,
# 							kernel_param=bw,
# 							lambda=la,
# 							val.misclass=vmc,
# 							tng.nrow=nrow(d_train),
# 							tng.ncol=ncol(d_train),
# 							fit.time=total.time),
# 						stringsAsFactors=FALSE)
# 					cat(sprintf("%s: bw=%1.2f, lambda=%1.4f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,vmc))
# 					la.list[[as.character(la.vector[i])]] <- list(lambda=10^(la.vector[i]),vmc=vmc)
# 				}
# 				score.vector <- c(la.list[[as.character(la.vector[1])]][['vmc']],
# 					la.list[[as.character(la.vector[2])]][['vmc']],
# 					la.list[[as.character(la.vector[3])]][['vmc']])
# 				la.vector <- return_next(la.vector,score.vector)
# 			}
# 		}
# 	}
# }

# for(s in c("tanh")){
# 	if(s=="tanh"){
# 		for(bw in c(1,10,100)){
# 			la.low <- -8
# 			la.mid <- -4
# 			la.high <- 0
# 			la.list <- list()
# 			la.vector <- c(la.low,la.mid,la.high)
# 			while(la.vector[3]-la.vector[1] > 1){
# 				w <- which((!(as.character(la.vector) %in% names(la.list))))
# 				for(i in w){
# 					la <- 10^(la.vector[i])
# 					t <- Sys.time()
# 					vmc <- 1
# 					total.time <- 0
# 					try({
# 						f <- fda(form,
# 							data=d_train,
# 							method=kernel_reg_fast_kernlab,
# 							kernel_function=tanhdot(scale=bw),
# 							lambda=la
# 							);
# 						total.time <- as.numeric(difftime(Sys.time(),t,units="secs"));
# 						p <- predict(f,d_val[,1:(ncol(d_val-1))]);
# 						val.misclass <- length(which(p != d_val[,ncol(d_val)]));
# 						vmc <- val.misclass/nrow(d_val);
# 					})
# 					out.d<-rbind(out.d,
# 						data.frame(datetime=format(dt),
# 							kernel=s,
# 							kernel_param=bw,
# 							lambda=la,
# 							val.misclass=vmc,
# 							tng.nrow=nrow(d_train),
# 							tng.ncol=ncol(d_train),
# 							fit.time=total.time),
# 						stringsAsFactors=FALSE)
# 					cat(sprintf("%s: sigma=%1.2f, lambda=%1.4f, tng size=%i x %i, time = %1.1f s, vmc: %0.2f\n",s,bw,la,nrow(d_train),ncol(d_train),total.time,vmc))
# 					la.list[[as.character(la.vector[i])]] <- list(lambda=10^(la.vector[i]),vmc=vmc)
# 				}
# 				score.vector <- c(la.list[[as.character(la.vector[1])]][['vmc']],
# 					la.list[[as.character(la.vector[2])]][['vmc']],
# 					la.list[[as.character(la.vector[3])]][['vmc']])
# 				la.vector <- return_next(la.vector,score.vector)
# 			}
# 		}
# 	}
# }


# # write.csv(out.d,"/home/cemarks/Projects/modulation/code/results.csv",row.names=FALSE)

