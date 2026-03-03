# Prediction step: t|t -> t+1|t
predict_step <- function(X_t, Omega_t, F, H, sigma2_eta, n_loc) {
  
  Sigma_eta = diag(sigma2_eta, n_loc)
  
  # One-step ahead state prediction
  X_t1_given_t <- F %*% X_t
  
  # State covariance prediction
  # C^{HV} = H * Sigma_eta * H^T
  C_HV <- H %*% Sigma_eta %*% t(H)
  Omega_t1_given_t <- F %*% Omega_t %*% t(F) + C_HV
  
  
  return(list(
    X_pred = X_t1_given_t,
    Omega_pred = Omega_t1_given_t
  ))
}

# Update step: t+1|t -> t+1|t+1
update_step <- function(X_pred, Omega_pred, Y_obs,
                        G, M, beta, sigma2_w) {
  
  # Innovation
  Y_pred <- G %*% X_pred + M %*% beta
  innovation <- Y_obs - Y_pred
  
  # Measurement covariance
  R <- matrix(sigma2_w, 1, 1)
  
  # Innovation variance
  # Delta_t
  S_t <- G %*% Omega_pred %*% t(G) + R
  S_inv <- solve(S_t)
  
  # Kalman gain
  K_t <- Omega_pred %*% t(G) %*% S_inv
  
  # State update
  X_updated <- X_pred + K_t %*% innovation
  
  # Covariance update
  Omega_updated <- Omega_pred - K_t %*% S_t %*% t(K_t)
  
  # Log-likelihood contribution
  log_det <- as.numeric(determinant(S_t, logarithm = TRUE)$modulus)
  quad_form <- as.numeric(t(innovation) %*% S_inv %*% innovation)
  
  likeli <- -0.5 * (log_det + quad_form + log(2*pi))
  
  return(list(
    X_updated = X_updated,
    Omega_updated = Omega_updated,
    likeli = likeli
  ))
}

