library(stats)

# TEMPORAL
psi_frac <- function(m, d, lambda) {
  psi <- numeric(m + 1)
  psi[1] <- 1
  for (k in 2:(m + 1)) {
    psi[k] <- psi[k - 1] * ((k - 2 + d) / (k - 1)) * exp(-lambda)
  }
  psi
}


psi_arma <- function(m, ar = numeric(0), ma = numeric(0)) {
  # ARMAtoMA returns psi_1,...,psi_m
  psi_tail <- ARMAtoMA(ar = ar, ma = ma, lag.max = m)
  c(1, psi_tail)
}


psi_artfima <- function(m, d, lambda, ar = numeric(0), ma = numeric(0)) {
  
  psi_f <- psi_frac(m, d, lambda)
  psi_a <- psi_arma(m, ar, ma)
  
  psi <- numeric(m + 1)
  
  for (k in 0:m) {
    psi[k + 1] <- sum(
      psi_a[1:(k + 1)] * rev(psi_f[1:(k + 1)])
    )
  }
  
  psi
}

### SPATIAL
# PLEASE NOTE: We use v = 1/2 so Model 1 is in consideration

build_grid <- function(n_side) {
  coords <- expand.grid(x = 1:n_side, y = 1:n_side)
  as.matrix(coords)
}

matern_exp_cov <- function(coords, sigma2_eta, rho) {
  
  D <- as.matrix(dist(coords))  # pairwise distances
  
  Sigma_eta <- sigma2_eta * exp(-D / rho)
  
  return(Sigma_eta)
}
