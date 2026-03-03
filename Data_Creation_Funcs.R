
library(MASS)
simulate_artfima_univariate <- function(T, true_theta) {
  
  A <- artsim(
    n = T,
    d = true_theta$d,
    lambda = true_theta$lambda,
    phi = true_theta$phi,
    theta = true_theta$theta,
    sigma2 = true_theta$sigma2_eta
  )
  
  Y <- A + rnorm(T, 0, sqrt(true_theta$sigma2_w))
  
  matrix(Y, nrow = 1)
}

simulate_artfima_spatial <- function(
    T_len,
    n_side,
    m,
    true_theta
) {
  
  N <- n_side^2
  
  # -------------------------
  # 1️⃣ Build spatial grid
  # -------------------------
  coords <- expand.grid(x = 1:n_side, y = 1:n_side)
  coords <- as.matrix(coords)
  
  D <- as.matrix(dist(coords))
  
  # -------------------------
  # 2️⃣ Spatial covariance (Matérn ν = 1/2)
  # -------------------------
  if (N == 1) {
    Sigma_eta <- matrix(true_theta$sigma2_eta, 1, 1)
  } else {
    Sigma_eta <- true_theta$sigma2_eta *
      exp(-D / true_theta$rho)
  }
  
  # -------------------------
  # 3️⃣ Temporal weights (ψ)
  # -------------------------
  psi <- psi_artfima(
    m = m,
    d = true_theta$d,
    lambda = true_theta$lambda,
    ar = true_theta$phi,
    ma = true_theta$theta
  )
  
  # -------------------------
  # 4️⃣ Generate innovations η_t
  # -------------------------
  
  eta <- matrix(0, N, T_len + m)
  
  for (t in 1:(T_len + m)) {
    eta[, t] <- mvrnorm(1,
                        mu = rep(0, N),
                        Sigma = Sigma_eta)
  }
  
  # -------------------------
  # 5️⃣ Build latent process X_t
  # -------------------------
  X <- matrix(0, N, T_len)
  
  for (t in 1:T_len) {
    for (j in 0:m) {
      X[, t] <- X[, t] + psi[j + 1] * eta[, t + m - j]
    }
  }
  
  # -------------------------
  # 6️⃣ Add measurement noise
  # -------------------------
  W <- matrix(
    rnorm(N * T_len, 0, sqrt(true_theta$sigma2_w)),
    nrow = N,
    ncol = T_len
  )
  
  Y <- X + W
  
  return(Y)
}
