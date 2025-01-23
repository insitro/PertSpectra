fitted_norman_GSFA <- readRDS("~/GSFA/fitted_norman_GSFA.rds")

Z <- fitted_norman_GSFA$posterior_means$Z_pm
beta <- fitted_norman_GSFA$posterior_means$beta_pm
W <- fitted_norman_GSFA$posterior_means$W_pm
F <- fitted_norman_GSFA$posterior_means$F_pm
lsfr <- fitted_norman_GSFA$lfsr

write.csv(Z,"~/GSFA/norman_gsfa_outputs/Z.csv")
write.csv(beta,"~/GSFA/norman_gsfa_outputs/beta.csv")
write.csv(W,"~/GSFA/norman_gsfa_outputs/W.csv")
write.csv(F,"~/GSFA/norman_gsfa_outputs/F.csv")
write.csv(lsfr,"~/GSFA/norman_gsfa_outputs/lsfr.csv")
