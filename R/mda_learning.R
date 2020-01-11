#!/usr/bin/env Rscript

library(mda)
library(RMySQL)

if(Sys.info()['sysname']=="Darwin"){
	dir.prefix <- '/Users/cemarks/Documents/modulation'
}else{
	dir.prefix <- '/home/cemarks/Projects/modulation'
}

source(paste(dir.prefix,"code/kernel_reg_functions.R",sep="/"))

dt_ext <- strftime(Sys.time(),"%Y-%m-%d_%H-%M-%S")

tng_size <- 3000
val_size <- 500

snr_df <- data.frame(
	snr = c(-10,-6,-2,2,6,10),
	snr_table = c("snr_n10","snr_n6","snr_n2","snr_2","snr_6","snr_10")
	)
cnx = dbConnect(drv=MySQL(),host="10.0.0.2",user="modu",password="!gO98aRMY&",dbname="mod_new")
# max_train_record <- 24*2000*14-1
max_train_record <- 480000
train_val_records <- sample(1:max_train_record,tng_size+val_size,replace=FALSE)
train_records <- train_val_records[1:tng_size]
val_records <-train_val_records[(tng_size+1):(tng_size+val_size)]

transform_pp <- function(pp,minval){
	tr <- t(apply(pp,1,function(x) return(sapply(x,function(y) return(max(y,minval))))))
	return(t(apply(tr,1,function(x) return(x/sum(x)))))
}

find_best_minval <- function(pp,Y,start.minval = 0.0005){
	best <- 0
	sr <- score(pp,Y)
	if(is.na(sr)) sr <- 0.00001
	minval <- start.minval/2
	best_minval <- 0
	while(sr > best){
		best_minval <- minval
		minval <- minval *2
		best <- sr
		newpp <- transform_pp(pp,minval)
		sr <- score(newpp,Y)
	}
	high <- best_minval * 2
	low <- best_minval/2
	while (high - low > start.minval){
		hm <- (high+best_minval)/2
		newpp <- transform_pp(pp,hm)
		sr <- score(newpp,Y)
		if(sr > best){
			low <- best_minval
			best_minval <- hm
			best <- sr
		} else {
			high <- hm
		}
		llm <- (low + best_minval)/2
		newpp <- transform_pp(pp,llm)
		sr <- score(newpp,Y)
		if(sr > best){
			high <- best_minval
			best_minval <- llm
			best <- sr
		} else {
			low <- llm
		}
	}
	return(list(best=best,best_minval=best_minval))
}

column_names <- read.csv(paste(dir.prefix,"feature_names_db.csv",sep="/"),stringsAsFactors=FALSE)


column_numbers_1 <- c(
    1:51
)

column_numbers_2 <- c(
    1:8,
    35:37,
    51:53,
    67:69,
    82:84,
    98:100
	)
form <- as.formula("y~.")
out.d <- data.frame(matrix(nrow=0,ncol=12),stringsAsFactors=FALSE)
names(out.d)<-c("snr","kernel","kernel_param","lambda","train.misclass","val.misclass","tng.nrow","tng.ncol","fit.time","tng.score","best.minval","best.score")



file.copy(paste(dir.prefix,"code/mda_learning.R",sep="/"),paste(dir.prefix,sprintf("code/archive/mda_learning_%s.R",dt_ext),sep="/"))


snr <- -10
s <- "rbf"
bw <- 20
la <- 10



for(snr in c(-10,-6,-2,2,6,10)){
	snr_table <- snr_df$snr_table[which(snr_df$snr==snr)]
	tab_1_cols <- paste(paste(snr_table,column_names$name.db[which(column_names$no %in% column_numbers_1 & column_names$table==1)],sep="."),collapse=",")
	tab_2_cols <- paste(paste(paste("new",snr_table,sep="_"),column_names$name.db[which(column_names$no %in% column_numbers_1 & column_names$table==2)],sep="."),collapse=",")
	train_query <- sprintf("SELECT %s,%s,%s.y FROM %s JOIN %s ON %s.record_no=%s.record_no WHERE %s.record_no IN (%s);",
		tab_1_cols,
		tab_2_cols,
		snr_table,
		snr_table,
		paste("new",snr_table,sep="_"),
		snr_table,
		paste("new",snr_table,sep="_"),
		snr_table,
		paste(train_records,collapse=",")
		)
	d_train <- dbGetQuery(cnx,train_query)
	d_train <- d_train[sample(nrow(d_train)),]
	val_query <- sprintf("SELECT %s,%s,%s.y FROM %s JOIN %s ON %s.record_no=%s.record_no WHERE %s.record_no IN (%s);",
		tab_1_cols,
		tab_2_cols,
		snr_table,
		snr_table,
		paste("new",snr_table,sep="_"),
		snr_table,
		paste("new",snr_table,sep="_"),
		snr_table,
		paste(val_records,collapse=",")
		)
	d_val <- dbGetQuery(cnx,val_query)
	d_val <- d_val[sample(nrow(d_val)),]
	x_mean=NULL
	x_sd=NULL
	for(i in 1:(ncol(d_train)-1)){
		x_mean = append(x_mean,mean(d_train[,i]))
		x_sd = append(x_sd,sd(d_train[,i]))
		d_train[,i] <- (d_train[,i] - x_mean[i])/x_sd[i]
		d_val[,i] <- (d_val[,i] - x_mean[i])/x_sd[i]
	}
	Y_val <- as.character(d_val$y)
	Y_tng <- as.character(d_train$y)
	for(s in c("poly","rbf")){
		if(s=="poly"){
			for(deg in 1:4){
				for(la in 10^((deg):(8+deg))){
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
					ptng <- predict(f,d_train[,1:(ncol(d_val-1))])
					tng.misclass <- length(which(ptng != d_train[,ncol(d_train)]))
					tmc <- tng.misclass/nrow(d_train)
					vmc <- val.misclass/nrow(d_val)
					pp <- predict(f,d_val[,1:(ncol(d_val)-1)],type="posterior");
					if(!any(is.na(pp))){
						sr <- score(pp,Y_val)
						ppt <- predict(f,d_train[,1:(ncol(d_train)-1)],type="posterior");
						srt <- score(ppt,Y_tng)
						bests <- find_best_minval(pp,Y_val)
						out.d<-rbind(out.d,
							data.frame(snr = snr,
								kernel=s,
								kernel_param=deg,
								lambda=la,
								val.misclass=vmc,
								train.misclass=tmc,
								tng.nrow=nrow(d_train),
								tng.ncol=ncol(d_train),
								fit.time=total.time,
							    tng.score=srt,
							    best.minval = bests$best_minval,
							    best.score=bests$best,
							stringsAsFactors=FALSE))
						cat(sprintf("snr: %i, %s: deg=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, tmc: %0.2f vmc: %0.2f, tng.score: %1.2f, best.minval: %1.4f, best.score: %1.2f\n",snr,s,deg,la,nrow(d_train),ncol(d_train),total.time,tmc,vmc,srt,bests$best_minval,bests$best))
					} else{
						cat(sprintf("Skipping %s, deg %1.2f, lambda %1.0f--prob overflow\n",s,deg,la))
					}
				}
			}
		} else if(s=="rbf") {
			# for(bw in 1){
				# for(l in 10^0){
			for(bw in c(1,5,10,20,40)){
				for(la in 10^((-5):(4))){
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
					pp <- predict(f,d_val[,1:(ncol(d_val)-1)],type="posterior");
					if(!any(is.na(pp))){
						ptng <- predict(f,d_train[,1:(ncol(d_val-1))])
						tng.misclass <- length(which(ptng != d_train[,ncol(d_train)]))
						tmc <- tng.misclass/nrow(d_train)
						sr <- score(pp,Y_val)
						ppt <- predict(f,d_train[,1:(ncol(d_train)-1)],type="posterior");
						srt <- score(ppt,Y_tng)
						val.misclass <- length(which(p != d_val[,ncol(d_val)]))
						vmc <- val.misclass/nrow(d_val)
						bests <- find_best_minval(pp,Y_val)
						out.d<-rbind(out.d,
							data.frame(snr = snr,
								kernel=s,
								kernel_param=bw,
								lambda=la,
								val.misclass=vmc,
								train.misclass=tmc,
								tng.nrow=nrow(d_train),
								tng.ncol=ncol(d_train),
								fit.time=total.time,
							    tng.score=srt,
							    best.minval = bests$best_minval,
							    best.score=bests$best,
							stringsAsFactors=FALSE))
					cat(sprintf("snr: %i, %s: bw=%1.2f, lambda=%1.0f, tng size=%i x %i, time = %1.1f s, tmc: %0.2f, vmc: %0.2f, tng score: %1.2f, best_minval: %1.3f, best score: %1.2f\n",
						snr,s,bw,la,nrow(d_train),ncol(d_train),total.time,tmc,vmc,srt,bests$best_minval,bests$best))
					} else{
						cat(sprintf("Skipping %s, bw %1.2f, lambda %1.0f--prob overflow\n",s,bw,la))
					}
				}
			}
		}
	}
}

write.csv(out.d,paste(dir.prefix,sprintf("results/MDA%s_%s.csv",snr_table,dt_ext),sep="/"),row.names=FALSE)
dbDisconnect(cnx)




# zt <- sapply(1:tng_size,function(i) return(ppt[i,Y_tng[i]]))
# zv <- sapply(1:val_size,function(i) return(pp[i,Y_val[i]]))
