fitted_rna565_GSFA <- readRDS("~/GSFA/fitted_rna565_GSFA.Rds")

Z <- fitted_rna565_GSFA$posterior_means$Z_pm
beta <- fitted_rna565_GSFA$posterior_means$beta_pm
W <- fitted_rna565_GSFA$posterior_means$W_pm
F <- fitted_rna565_GSFA$posterior_means$F_pm
lsfr <- fitted_rna565_GSFA$lfsr

write.csv(Z,"~/GSFA/rna565_gsfa_outputs/Z.csv")
write.csv(beta,"~/GSFA/rna565_gsfa_outputs/beta.csv")
write.csv(W,"~/GSFA/rna565_gsfa_outputs/W.csv")
write.csv(F,"~/GSFA/rna565_gsfa_outputs/F.csv")
write.csv(lsfr,"~/GSFA/rna565_gsfa_outputs/lsfr.csv")
