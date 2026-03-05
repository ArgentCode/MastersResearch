rm(list=ls())
gc()
library(MASS)
library(doParallel)
library(foreach)

source("ArtfimaCausal.R")
source("MultiStateSpaceFunctions.R")
source("Data_Creation_Funcs.R")
source("MCFuncs.R")

n_cores <- detectCores() -1 

cluster <- makeCluster(n_cores, timeout = 120, outfile = "")

clusterExport(cluster, ls())  # export all objects
invisible(clusterEvalQ(cluster, library(MASS)))  # needed for mvrnorm

doParallel::registerDoParallel(cluster)

true_theta <- list(
  d = 0,
  lambda = 0,
  phi = 0.45,
  theta = 0.3,
  sigma2_eta = 0.5,
  sigma2_w = 0,
  rho = 0.35
)

hat_theta <- c(
  phi = 0.5,
  theta = 0.5,
  sigma2_eta = 0.75,
  rho = 0.5
)

output_file <- "ARMA1.txt"
sink(output_file)

cat("Monte Carlo Study Results\n")
cat("===========================\n\n")
cat("Started at:", format(Sys.time()), "\n\n")

settings <- list(
  list(T_len = 50, n_side = 5, m = 5, iter = 30),
  list(T_len = 75, n_side = 7, m = 6, iter = 30)
  # list(T_len = 100, n_side = 10, m = 5, iter = 100),
  # list(T_len = 250, n_side = 10, m = 5, iter = 100)
)

for (s in settings) {
  run_one_setting(cluster, true_theta, hat_theta,
                  m = s$m,
                  iter = s$iter,
                  n_side = s$n_side,
                  T_len = s$T_len)
}
stopCluster(cluster)

cat("Finished at:", format(Sys.time()), "\n")
sink()