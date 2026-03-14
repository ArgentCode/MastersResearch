rm(list=ls())
gc()

# library(doParallel)
# library(foreach)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("../Base Functions/ArtfimaCausal.R")
source("../Base Functions/Data_Creation_Funcs.R")
source("TestPredictFunction.R")
source("MatrixMCRuns.R")

# n_cores <- 2
# 
# cluster <- makeCluster(n_cores, timeout = 120, outfile = "")
# 
# clusterExport(cluster, ls())  # export all objects
# invisible(clusterEvalQ(cluster, library(MASS)))  # needed for mvrnorm
# 
# doParallel::registerDoParallel(cluster)

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

T_len = 100
m = 5
n_side = 10
iter = 10

start_time <- Sys.time()
output_file <- "ARMA1.txt"
sink(output_file)

cat("Monte Carlo Study Results\n")
cat("===========================\n\n")
cat("Started at:", format(start_time), "\n\n")
cat("Settings:\n")
cat("T =", T_len, 
    " | n_side =", n_side,
    " | m =", m,
    " | iter =", iter, "\n\n")


results <- run_MC(true_theta, hat_theta, 
                  m, iter, n_side, T_len)

end_time <- Sys.time()
print(results)

cat("\nRuntime:",
    round(difftime(end_time, start_time, units = "mins"), 2),
    "minutes\n\n")

# stopCluster(cluster)

cat("Finished at:", format(Sys.time()), "\n")
sink(file=NULL)
