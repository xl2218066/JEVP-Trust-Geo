

# environment and data setting  -------------------------------------------



library(glmmTMB)
library(sjstats)

dat <- read.csv('d.csv')

# code the variable
dat$gender_binary <- factor(dat$sex)
dat$marriage_binary <- factor(dat$marriage)

dat$pro <- match(dat$PROVCD18,unique(dat$PROVCD18))

# function for formatting the result of glmmTMB
glmmTMBformat <- function(glmm){
  values <- coef(summary(glmm))$cond
  coefficients <- values[,1]
  se <- values[,2]
  p.value <- values[,4]
  
  
  #odds ratio
  ors <- exp(confint(glmm,"beta_"))
  or <- ors[,3]
  or.l.ci <- ors[,1]
  or.h.ci <- ors[,2]
  
  # r2
  R2 <- MuMIn::r.squaredGLMM(glmm)
  R2m <-R2[1]
  R2c <- R2[3]
  
  # ICC
  icc <- performance::icc(glmm)[1,1]
  
  # AIC
  aic <- extractAIC(glmm)[2]
  
  
  coefficients['AIC']<- aic
  coefficients['R2_m'] <- R2m
  coefficients['R2_c'] <- R2c
  coefficients['ICC'] <- icc
  # 
  l <- length(coefficients) - length(se)
  df <- data.frame(Estimate = coefficients,
                   Standard_error = c(se, rep(NA,l)),
                   odds_ratio = c(or, rep(NA,l)),
                   or.low.ci = c(or.l.ci,rep(NA,l)),
                   or.high.ci = c(or.h.ci, rep(NA,l)),
                   p_value = c(p.value, rep(NA,l)))
  
  return(df)
  
}


# building models ---------------------------------------------------------

## Step-one model
m0.TMB <- glmmTMB(factor(trust) ~ 1 + age + education + gender_binary + marriage_binary 
                  +  income + Institutional.quality 
                  + (1|pro), 
                  data = dat, family = binomial)
r0 <- glmmTMBformat(m0.TMB)
print(r0)


## Step-two model
m1.TMB <- glmmTMB(factor(trust) ~  age + education + gender_binary + marriage_binary 
                  +  income + Institutional.quality 
                  + Latitude + Longitude +GDP.per.capita + (1|pro), 
                  data = dat, family = binomial)
r1 <- glmmTMBformat(m1.TMB)
print(r1)


## Step-three model
m2.TMB <- glmmTMB(factor(trust) ~ 1 + age + education + gender_binary + marriage_binary 
                  +  income + Institutional.quality 
                  + Latitude + Longitude +GDP.per.capita 
                  + Elevation_mean + Elevation_std + Elevation_CV  + (1|pro), 
                  data = dat, family = binomial)
r2 <- glmmTMBformat(m2.TMB)
print(r2)

