library(data.table)
library(tidyverse)
library(Matrix)
library(GSFA)
library(ggplot2)

install.packages('reticulate')
library(reticulate)
use_python("/usr/bin/python3")
py_discover_config()
np <- import("numpy")

npz <- np$load("inhouse_GSFA_inputs.npz", allow_pickle=TRUE)
Y <- npz$get("array1")
G <- npz$get("array2")

print(dim(Y))
print(dim(G))
print("loaded data")

dev_res <- deviance_residual_transform(Y)
top_gene_index <- select_top_devres_genes(dev_res, num_top_genes = 6000)
dev_res_filtered <- dev_res[, top_gene_index]

write.csv(top_gene_index, "inhouse_top_genes.csv")
rm(npz)
rm(Y)
print(dim(dev_res_filtered))
print("processed data")

set.seed(14314)
time_start = Sys.time()
num_cells = 5000
fit <- fit_gsfa_multivar(Y = dev_res_filtered[1:num_cells,], G = G[1:num_cells,],
                         K = 20,
                         prior_type = "mixture_normal",
                         init.method = "svd",
                         niter = 3000, used_niter = 1000,
                         verbose = T, return_samples = T)
print(Sys.time()-time_start)
rm(G)
rm(dev_res_filtered)
saveRDS(fit, file = "fitted_inhouse.rds")
