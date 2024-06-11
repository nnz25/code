# Required packages
library(scales)
library(dplyr)
library(Rcpp)
library(rstan)
library(hBayesDM)
library(ggpubr)
library(ggplot2)
library(reshape2)

# Parallel computing allowed
options(mc.cores = parallel::detectCores())

# Set working directory
setwd("C:/Users/P312648/OneDrive/桌面/EXP3_bayes_loss")

######### arrange data list#######################
# Load data, all reward value is negative
dfHL <- read.table(file = 'KNSVHCLOSS.txt',header = TRUE)
dfPL <- read.table(file = 'KNSVPALOSS.txt',header = TRUE)

# healthy 
reward=matrix(dfHL$reward,nrow=45,ncol=50,byrow=FALSE)
effort=matrix(dfHL$effort,nrow=45,ncol=50,byrow=FALSE)
choice=matrix(dfHL$choice,nrow=45,ncol=50,byrow=FALSE)
Tsubj=rep(50,45)

data_list1 <- list(
  N = 45,
  T = 50,
  reward = -reward,
  effort = effort,
  choice = choice,
  Tsubj = Tsubj
)

# patient 
reward=matrix(dfPL$reward,nrow=34,ncol=50,byrow=FALSE)
effort=matrix(dfPL$effort,nrow=34,ncol=50,byrow=FALSE)
choice=matrix(dfPL$choice,nrow=34,ncol=50,byrow=FALSE)
Tsubj=rep(50,34)

data_list2 <- list(
  N = 34,
  T = 50,
  reward = -reward,
  effort = effort,
  choice = choice,
  Tsubj = Tsubj
)


#################### model fitting#############################
# linear model sv_linear=reward[i,t] - k[i]*effort[i,t];
m1 <- stan(
  file="egt_linear.stan",
  data=data_list1,
  chains = 4,
  iter = 4000
)

# parabolic model: sv_parabolic  = a[i]*reward[i,t]-k[i]*pow(effort[i, t], 2);
m2 <- stan(
  file="egt_parab.stan",
  data=data_list1,
  chains = 4,
  iter = 4000
)

# hyperbolic model: 
m3 <- stan(
  file="egt_hyper.stan",
  data=data_list1,
  chains = 4,
  iter = 4000
)

# exponential model:
m4 <- stan(
  file="egt_sigm.stan",
  data=data_list1,
  chains = 4,
  iter = 4000
)

# linear model sv_linear=reward[i,t] - k[i]*effort[i,t];
m5 <- stan(
  file="egt_linear.stan",
  data=data_list2,
  chains = 4,
  iter = 4000
)

# parabolic model: sv_parabolic  = a[i]*reward[i,t]-k[i]*pow(effort[i, t], 2);
m6 <- stan(
  file="egt_parab.stan",
  data=data_list2,
  chains = 4,
  iter = 4000
)

# hyperbolic model: 
m7 <- stan(
  file="egt_hyper.stan",
  data=data_list2,
  chains = 4,
  iter = 4000
)

# exponential model:
m8 <- stan(
  file="egt_sigm.stan",
  data=data_list2,
  chains = 4,
  iter = 4000
)


################### PPC and model comparison ###########################

# LOOC model compare:
L1=loo(m1)
L2=loo(m2)
L3=loo(m3)
L4=loo(m4)
L5=loo(m5)
L6=loo(m6)
L7=loo(m7)
L8=loo(m8)

loo_result_loss = data.frame(L1$looic,L2$looic,L3$looic,L4$looic,L5$looic,L6$looic,L7$looic,L8$looic)
loo_result_loss=melt(loo_result_loss)
save(loo_result_loss,file="loo_loss_June_Risnegative.RData")

looc=mutate(loo_result_loss,
            group=c('HC','HC','HC','HC','PA','PA','PA','PA'),
            models=c('linear','parab','hyper','exp','linear','parab','hyper','exp'),
            value=as.integer(value))

#theme(axis.text.x = element_text(angle = 45,hjust=1))
ggbarplot(data=looc,x="group",y="value", 
          fill='models',
          add.params =  list(alpha = 0.7, size = 0.8),
          position = position_dodge(0.8),
          ylab = "LOOC",
          label = TRUE)+
  scale_fill_manual(values =c('#bdbebd','#a9a9a9','#949494','#808080')
  )+
  theme_pubr(margin = T)+
  font('xylab', face = 'bold', size = 16)+ 
  font("legend.title",face = 'bold',size=16)+
  font("legend.text", face = 'bold',size=16)+
  theme(axis.line.y=element_line(linetype=1,color="black",size=0.5))+
  theme(axis.line.x=element_line(linetype=1,color="black",size=0.5))

################ parameter compare in wining model
# parameter compare: M3,M7 is winning model: hyperbolic shape
fitted_m1=extract(m3) # sigmoid shape with a
fitted_m2=extract(m7) # sigmoid shape with a
save(fitted_m1,file="fitm1_L.RData")
save(fitted_m2,file="fitm2_L.RData")


## PPC one subject example
ymeans <- apply(fitted_m1$y_pred, c(2, 3), mean)

tiff("PPC_HC.tif")
par(mfrow = c(4, 1))
# sub1
plot(data_list1$choice[1,], type="s", xlab="Trial", ylab="Choice (0 or 1)", col='blue', yaxt="n")
lines(ymeans[1,], col="red", lty=2)
axis(side=2, at = c(0,1))
legend("bottomleft", legend=c("True", "PPC"), cex=0.8, col=c("blue", "red"), lty=1:2)
title(main="subject01")
# sub2
plot(data_list1$choice[2,], type="s", xlab="Trial", ylab="Choice (0 or 1)", col='blue', yaxt="n")
lines(ymeans[2,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), cex=0.8, col=c("blue", "red"), lty=1:2)
title(main="subject02")
# sub3
plot(data_list1$choice[3,], type="s", xlab="Trial", ylab="Choice (0 or 1)", col='blue', yaxt="n")
lines(ymeans[3,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), cex=0.8, col=c("blue", "red"), lty=1:2)
title(main="subject03")
# sub4
plot(data_list1$choice[4,], type="s", xlab="Trial", ylab="Choice (0 or 1)", col='blue', yaxt="n")
lines(ymeans[4,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), cex=0.8, col=c("blue", "red"), lty=1:2)
title(main="subject04")

dev.off()

### PPC for PA
ymeans <- apply(fitted_m2$y_pred, c(2, 3), mean)

tiff("PPC_PA.tif")
par(mfrow = c(4, 1))
# sub1
plot(data_list2$choice[1,], type="s", xlab="Trial", ylab="Choice (0 or 1)", col='blue', yaxt="n")
lines(ymeans[1,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), cex=0.8, col=c("blue", "red"), lty=1:2)
title(main="subject01")
# sub2
plot(data_list2$choice[2,], type="s", xlab="Trial", ylab="Choice (0 or 1)", col='blue', yaxt="n")
lines(ymeans[2,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), cex=0.8, col=c("blue", "red"), lty=1:2)
title(main="subject02")
# sub3
plot(data_list2$choice[3,], type="s", xlab="Trial", ylab="Choice (0 or 1)", col='blue', yaxt="n")
lines(ymeans[3,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), cex=0.8, col=c("blue", "red"), lty=1:2)
title(main="subject03")
# sub4
plot(data_list2$choice[4,], type="s", xlab="Trial", ylab="Choice (0 or 1)", col='blue', yaxt="n")
lines(ymeans[4,], col="red", lty=2)
axis(side=2, at = c(0,1) )
legend("bottomleft", legend=c("True", "PPC"), cex=0.8, col=c("blue", "red"), lty=1:2)
title(main="subject04")

dev.off()

# trace plot
traceplot(m1,pars=c("mu_k","mu_beta"))
traceplot(m2,pars=c("mu_k","mu_beta"))

# density and rhat plot
rstan::stan_rhat(m1,bins=30)
rstan::stan_rhat(m2,bins=30)


# 保存当前环境中的所有对象到一个.RData文件中
save.image("my_workspace_loss_Rnegative.RData")


