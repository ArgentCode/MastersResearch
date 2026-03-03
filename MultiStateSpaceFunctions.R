create_state_space <- function(
    n_loc,
    m,
    sigma2_w,
    psi,
    Sigma_eta   # full N x N spatial covariance
) {
  
  state_dim <- (m + 1) * n_loc
  
  # State Innovation Noise
  R <- diag(sigma2_w, n_loc)
  
  # -------- G matrix --------
  # Observation Loading Matrix
  # G = [psi0 I_N  psi1 I_N  ...  psim I_N]
  # G_blocks <- lapply(psi[1:(m+1)], function(p) p * diag(n_loc))
  # G <- do.call(cbind, G_blocks)
  
  # -------- F matrix --------
  # State Transition
  F <- matrix(0, state_dim, state_dim)

  for (j in 2:(m+1)) {
    row_idx <- ((j-1)*n_loc + 1):(j*n_loc)
    col_idx <- ((j-2)*n_loc + 1):((j-1)*n_loc)
    F[row_idx, col_idx] <- diag(n_loc)
  }
  
  # -------- H matrix --------
  # Mean structure
  H <- matrix(0, state_dim, n_loc)
  H[1:n_loc, 1:n_loc] <- diag(n_loc)
  
  return(list(
    # G = G,
    F = F,
    H = H,
    R = R,
    psi = psi,
    Sigma_eta = Sigma_eta
  ))
}

predict_step <- function(X_t, Omega_t, F, H, Sigma_eta) {

  # One-step ahead state prediction
  X_pred_2 <- F %*% X_t
  #############################################
  # New F alternative
  
  # r <- (m+1)*n_loc
  # X_pred <- numeric(r)
  # 
  # # shift everything down by one block
  # X_pred[(n_loc+1):r] <- X_t[1:(r-n_loc)]
  
  ##############################################

  # Process noise contribution
  Q <- H %*% Sigma_eta %*% t(H)
  # State covariance prediction
  Omega_pred <- F %*% Omega_t %*% t(F) + Q
  ##############################################
  # Omega_pred <- matrix(0, r, r)
  # 
  # Omega_pred[(n_loc+1):r, (n_loc+1):r] <- Omega_t[1:(r-n_loc), 1:(r-n_loc)]
  # 
  # 
  # Omega_pred[1:n_loc, 1:n_loc] <- Sigma_eta

  ###############################################

  return(list(
    X_pred = X_pred,
    Omega_pred = Omega_pred
  ))
}



update_step <- function(X_pred, Omega_pred, Y_obs,
                        psi, R) {
  
  # Innovation
  ## For loops avoids the mega-matrix nonsense
  # Y_pred <- G %*% X_pred
  Y_pred <- numeric(n_loc)
  
  for (k in 0:m) {
    idx <- (k*n_loc + 1):((k+1)*n_loc)
    Y_pred <- Y_pred + psi[k+1] * X_pred[idx]
  }
  innovation <- Y_obs - Y_pred
  
  # Innovation covariance
  ## r = (m+1)N
  ## the one line code below runs in about O(Nr^2), the double for loops is O(m^2N^2)
  # S_t <- G %*% Omega_pred %*% t(G) + R
  S_t <- matrix(0, n_loc, n_loc)
  
  for (i in 0:m) {
    idx_i <- (i*n_loc + 1):((i+1)*n_loc)
    
    for (j in 0:m) {
      idx_j <- (j*n_loc + 1):((j+1)*n_loc)
      
      Omega_ij <- Omega_pred[idx_i, idx_j]
      S_t <- S_t + psi[i+1] * psi[j+1] * Omega_ij
    }
  }
  
  S_t <- S_t + R
  
  S_inv <- solve(S_t)
  
  # ---- Kalman Gain ----
  # with loop speed up to avoid mega G matrix
  r <- length(X_pred)
  K_t <- matrix(0, r, n_loc)
  
  for (i in 0:m) {
    idx_i <- (i*n_loc + 1):((i+1)*n_loc)
    
    temp <- matrix(0, n_loc, n_loc)
    
    for (j in 0:m) {
      idx_j <- (j*n_loc + 1):((j+1)*n_loc)
      Omega_ij <- Omega_pred[idx_i, idx_j]
      temp <- temp + psi[j+1] * Omega_ij
    }
    
    K_t[idx_i, ] <- temp %*% S_inv
  }
  
  # State update
  X_updated <- X_pred + K_t %*% innovation
  
  # Covariance update
  Omega_updated <- Omega_pred - K_t %*% S_t %*% t(K_t)
  
  # Log-likelihood
  log_det <- as.numeric(determinant(S_t, logarithm = TRUE)$modulus)
  quad_form <- as.numeric(t(innovation) %*% S_inv %*% innovation)
  
  n_loc <- length(Y_obs)
  likeli <- -0.5 * (log_det + quad_form + n_loc * log(2*pi))
  
  return(list(
    X_updated = X_updated,
    Omega_updated = Omega_updated,
    likeli = likeli
  ))
}


kalman_filter <- function(data, state_space, m) {
  
  likelihood <- 0
  n_loc <- nrow(data)
  T_len <- ncol(data)
  
  state_dim <- (m + 1) * n_loc
  
  # Initialize state
  X_t <- matrix(0, state_dim, 1)
  Omega_t <- diag(500, state_dim)
  
  for (t in 1:T_len) {
    
    # -------- Predict --------
    pred <- predict_step(
      X_t,
      Omega_t,
      state_space$F,
      state_space$H,
      state_space$Sigma_eta
    )
    
    X_pred <- pred$X_pred
    Omega_pred <- pred$Omega_pred
    
    # Observation vector (N x 1)
    Y_obs <- matrix(data[, t], ncol = 1)
    
    # -------- Update --------
    update <- update_step(
      X_pred,
      Omega_pred,
      Y_obs,
      state_space$psi,
      state_space$R
    )
    
    X_t <- update$X_updated
    Omega_t <- update$Omega_updated
    
    likelihood <- likelihood + update$likeli
  }
  
  return(as.numeric(likelihood))
}



neg_loglik <- function(par, data, m, n_side) {
  
  # ---- Extract parameters safely ----
  theta <- list(
    d = if ("d" %in% names(par)) par[['d']] else 0,
    lambda = if ("lambda" %in% names(par)) par[['lambda']] else 0,
    phi = if ("phi" %in% names(par)) par[['phi']] else 0,
    theta = if ("theta" %in% names(par)) par[['theta']] else 0,
    sigma2_eta = if ("sigma2_eta" %in% names(par)) par[['sigma2_eta']] else NA,
    sigma2_w = if ("sigma2_w" %in% names(par)) par[['sigma2_w']] else 0.000001,
    rho = if ("rho" %in% names(par)) par[['rho']] else NA
  )
  
  # ---- Hard constraints ----
  if (theta$sigma2_eta <= 0 ||
      theta$sigma2_w <= 0 ||
      abs(theta$phi) >= 0.999 ||
      abs(theta$theta) >= 0.999) {
    return(1e12)
  }
  
  # ---- Build psi (temporal structure) ----
  psi <- psi_artfima(
    m = m,
    d = theta$d,
    lambda = theta$lambda,
    ar = theta$phi,
    ma = theta$theta
  )
  
  # ---- Build spatial covariance (Matérn ν = 1/2) ----
  n_loc = nrow(data)
  if (n_loc == 1) {
    
    # Univariate case
    Sigma_eta <- matrix(theta$sigma2_eta, 1, 1)
    
  } else {
    
    # Multivariate spatial case
    if (is.na(theta$rho) || theta$rho <= 0)
      return(1e12)
    
    coords <- expand.grid(x = 1:n_side, y = 1:n_side)
    coords <- as.matrix(coords)
    
    D <- as.matrix(dist(coords))
    
    Sigma_eta <- theta$sigma2_eta * exp(-D / theta$rho)
  }
  
  
  # ---- Build state-space matrices ----
  state_space <- create_state_space(
    n_loc = nrow(data),
    m = m,
    sigma2_w = theta$sigma2_w,
    psi = psi,
    Sigma_eta = Sigma_eta
  )
  
  # ---- Run Kalman filter ----
  lik <- kalman_filter(
    data = data,
    state_space = state_space,
    m = m
  )
  
  # ---- Return negative log-likelihood ----
  return(-as.numeric(lik))
}
