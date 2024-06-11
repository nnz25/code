library(dplyr)
library(lme4)
library(lmerTest)
library(ggplot2)
library(ggpubr)
library(bruceR)

############## data import and clean ###################
df=read.csv("longdata_for_rt_new_teps.csv")
str(df)
head(df)

df <- df %>%
  mutate(group = as.factor(group),
         Sex = as.factor(Sex),
         F = as.factor(F),
         AES=as.numeric(AES),
         CDI=as.numeric(CDI),
         TEPS_C=as.numeric(TEPS_C),
         ANT_C=as.numeric(ANT_C),
         CON_C=as.numeric(CON_C)) 


# task & group effect
m1=glmer(response~group*R*E+group*F+(1|sub_idx),family=binomial(link="logit"),data=df)
HLM_summary(m1)
model_summary(m1)
print_table(m1)

# apathy and anhedonia effect
m2=glmer(response~AES+ANT_C+CON_C+(1|sub_idx),family=binomial(link="logit"),data=df)
summary(m2)
anova(m2)

# task & group effect
m3=lmer(rt~group*R*E+group*F+(1|sub_idx),data=df)
HLM_summary(m3)
model_summary(m3)

# apathy and anhedonia effect
m4=lmer(rt~AES+ANT_C+CON_C+(1|sub_idx),data=df)
summary(m4)
anova(m4)

