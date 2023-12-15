library(tidyverse, quietly = TRUE)
library(survival)
library(survminer)
ICI.data = read.csv('pred-response.csv')
fit <- survfit(Surv(time, state) ~ response_pred, data = ICI.data)
ggsurvplot(fit, pval = TRUE,
           surv.median.line = "hv", xlab = "Progression-free survival (months)",
           risk.table = TRUE,
           legend.labs = c("Predicted R", "Predicted NR")   
)

res_cox = coxph(Surv(time, state) ~ response_pred, data = ICI.data) 
summary(res_cox)       
