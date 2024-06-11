library(hBayesDM)
library(ggplot2)
library(ggpubr)
library(patchwork)


###### HDI calculate: group difference in posterior distribution ##########

diff_k = fitted_m1$mu_k-fitted_m2$mu_k # group1 - group2
HDIofMCMC( diff_k )  # Compute the 95% Highest Density Interval (HDI).
plotHDI( diff_k,Title="95% HDI = [-0.06,-0.03]")  
ggsave(filename='kHDI.png',width=6,height=6,units="in",dpi=300)

diff_beta = fitted_m1$mu_beta-fitted_m2$mu_beta # group1 - group2
HDIofMCMC(diff_beta)  # Compute the 95% Highest Density Interval (HDI).
plotHDI( diff_beta,Title="95% HDI = [-0.06,0.01]")  
ggsave(filename='betaHDI.png',width=6,height=6,units="in",dpi=300)


################ posterior distribution of two groups #################
# plot posterior distribution of k 
dens1 <- density(fitted_m1[["mu_k"]]) # parameter a in HC
dens2 <- density(fitted_m2[["mu_k"]]) # parameter a in PA
#pdf("distribution_k.pdf",width = 6,height = 6)
pdf('distribution_k.pdf')
plot(dens1, main = "Posterior distribution of k",font.main=2,cex.main=1.5,
     xlim = range(c(dens1$x, dens2$x)),
     ylim = range(c(dens1$y, dens2$y)), 
     xlab = "effort sensitivity", ylab = "density",
     col = "black", lwd = 1.5,cex.lab=1.5,cex.axis=1.5,font.lab=2,font.axis=2)

polygon(dens1, col = "cyan4")
polygon(dens2, col = "brown")
lines(dens2, col = "black", lwd = 1.5)
legend("top", legend = c("HC", "PA"), col = c("cyan4", "brown"), lty = 1, lwd = 2,text.font=2,cex=1.5)
dev.off()

# plot posterior distribution beta
dens1 <- density(fitted_m1[["mu_beta"]]) # parameter a in HC
dens2 <- density(fitted_m2[["mu_beta"]]) # parameter a in PA
#pdf("distribution_beta.pdf",width = 6,height = 6)
pdf('distribution_beta.pdf')
plot(dens1, main = "Posterior distribution of beta",font.main=2,cex.main=1.5,
     xlim = c(0.1,0.3),
     ylim = range(c(dens1$y, dens2$y)), 
     xlab = "stochasticity", ylab = "density",
     col = "black", lwd = 1.5,cex.lab=1.5,cex.axis=1.5,font.lab=2,font.axis=2)
polygon(dens1, col = "cyan4")
polygon(dens2, col = "brown")
lines(dens2, col = "black", lwd = 1.5)
legend("topright",legend = c("HC", "PA"), col = c("cyan4", "brown"), lty = 1, lwd = 2,text.font=2,cex=1.5)
dev.off()

###############################################################################
################## plot correlation ###########################################
# why? it seems so complicated? because subID need be matched in each variable
# extract parameters
k_hc=apply(fitted_m1[["k"]],2,mean)
k_pa=apply(fitted_m2[["k"]],2,mean)
beta_hc=apply(fitted_m1[["beta"]],2,mean)
beta_pa=apply(fitted_m2[["beta"]],2,mean)

k_val=c(k_hc,k_pa)
beta_val=c(beta_hc,beta_pa)

# get ID info
rawID_hc=unique(dfHW$rawID)
rawID_pa=unique(dfPW$rawID)
subID_hc=unique(dfHW$subjID)
subID_pa=unique(dfPW$subjID)

rawID=c(rawID_hc,rawID_pa)
subID=c(subID_hc,subID_pa)

# get AES,anhedonia score
df_scale=read.csv("regression_AESANTCON.csv")

# generate AES data for correlation analysis
df_par=mutate(df_scale,
              AES=as.numeric(AES),
              ANT=as.numeric(ANT),
              CON=as.numeric(CON),
              ID=rawID,
              k=k_val,
              beta=beta_val)
df_par_1= subset(df_par, select = -c(ID, sub))
hc_par=df_par_1[1:45,]
pa_par=df_par_1[46:nrow(df_par), ]

## correlations
ggpairs(df_par_1)
ggpairs(hc_par)
ggpairs(pa_par)
#### ANT,CON and k (4 plots)
par_anh_HC=as.data.frame(par_anh_HC)%>%
  mutate(anhedonia=as.factor(V5),
         V6=as.numeric(V6),
         k_hc2=as.numeric(k_hc2))

p1=ggscatter(data=par_anh_HC,x = "V6", y = "k_hc2", color="anhedonia",
             add = "reg.line", conf.int = TRUE,
             xlab = "anhedonia score", ylab = "k",
             shape = 16, size = 1) +
  stat_cor(data = par_anh_HC,aes(color=anhedonia),method="pearson",fontface="bold")+
  scale_y_continuous(labels = scales::number_format(accuracy = 0.001))


par_anh_PA_cleaned <- na.omit(par_anh_PA)
par_anh_PA_cleaned=as.data.frame(par_anh_PA_cleaned)%>%
  mutate(anhedonia=as.factor(V5),
         V6=as.numeric(V6),
         k_pa2=as.numeric(k_pa2))
p2=ggscatter(data=par_anh_PA_cleaned,x = "V6", y = "k_pa2", color="anhedonia",
             add = "reg.line", conf.int = TRUE,
             xlab = "anhedonia score", ylab = "k",
             shape = 16, size = 1)+
  stat_cor(data = par_anh_PA_cleaned,aes(color=anhedonia),method="pearson",fontface="bold")+
  scale_y_continuous(labels = scales::number_format(accuracy = 0.001))

p=ggarrange(p1,p2,ncol=2,nrow=1,common.legend = T)+
  plot_annotation(title="Correlation between anhedonia and k")
print(p)
ggsave("anhedonia_k.tiff",plot=p,height=6,width=9)
