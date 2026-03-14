create_state_space <- function(
    n_loc,
    m,
    psi,
    Sigma_eta,   # full N x N spatial covariance
    T_len,
    sigma2_w = 0.000001
) {
  
  state_dim <- (m + 1) * n_loc
  
  # State Innovation Noise
  R <- diag(sigma2_w, n_loc)
  
  # -------- G matrix --------
  # Observation Loading Matrix
  # G = [psi0 I_N  psi1 I_N  ...  psim I_N]
  G_blocks <- lapply(psi[1:(m+1)], function(p) p * diag(n_loc))
  G <- do.call(cbind, G_blocks)
  
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
    G = G,
    F = F,
    H = H,
    R = R,
    Sigma_eta = Sigma_eta,
    n_loc = n_loc,
    m = m,
    state_dim = state_dim,
    T_len = T_len
  ))
}

predict_step <- function(X_t, Omega_t, state_space, Y_prev) {
  
  F = state_space$F
  H = state_space$H
  G = state_space$G
  Sigma_eta = state_space$Sigma_eta
  
  
  # Ferreira pieces
  Delta_t = G%*% Omega_t %*% t(G)
  Delta_inv = solve(Delta_t)
  Theta_t = F %*% Omega_t %*% t(G)
  epsilon_t = Y_prev - G %*% X_t
  
  
  # ---- Shift state ----
  # One-step ahead state prediction
  X_pred <- F %*% X_t + Theta_t %*% Delta_inv %*% epsilon_t
  
  # Process noise contribution
  Q <- H %*% Sigma_eta %*% t(H)
  
  # ---- Shift covariance ----
  # State covariance prediction
  Omega_pred <- F %*% Omega_t %*% t(F) + Q - Theta_t %*% Delta_inv %*% t(Theta_t)
  
  # ---- Safety checks ----
  if (max(abs(Omega_pred - t(Omega_pred))) > 1e-8)
    stop("Omega_pred not symmetric")
  
  return(list(
    X_pred = X_pred,
    Omega_pred = Omega_pred
  ))
}

update_step <- function(state_space, X_pred, Omega_pred, Y_obs) {
  
  if (anyNA(Y_obs)) {
    message("Implement a missing value function!")
  }
  
  n_loc <- state_space$n_loc
  state_dim <- state_space$dim
  m <- state_space$m
  G = state_space$G
  R = state_space$R
  
  # Innovation
  Y_pred <- G %*% X_pred
  innovation <- Y_obs - Y_pred
  
  # Innovation covariance
  S_t <- G %*% Omega_pred %*% t(G) + R
  
  # S_inv <- solve(S_t)
  S_inv <- tryCatch(
    chol2inv(chol(S_t)),
    error = function(e) {
      message("chol failed")
      print(S_t)
      return(solve(S_t))
    }
  )

  # eig_vals <- eigen(S_t, symmetric = TRUE, only.values = TRUE)$values
  # 
  # if (any(eig_vals <= 0)) {
  #   stop("S_t not PD")
  # }
  
  # ---- Kalman Gain ----
  K_t <- Omega_pred %*% t(G) %*% S_inv
  ######################################################
  
  # State update
  X_updated <- X_pred + K_t %*% innovation
  
  # Covariance update
  Omega_updated <- Omega_pred - K_t %*% S_t %*% t(K_t)
  
  Omega_updated <- (Omega_updated + t(Omega_updated)) / 2 
  
  if (any(diag(Omega_updated) < -1e-10)) {
    stop("Omega_t lost positive definiteness")
  }
  
  # Log-likelihood
  log_det <- as.numeric(determinant(S_t, logarithm = TRUE)$modulus)
  quad_form <- as.numeric(t(innovation) %*% S_inv %*% innovation)
  
  n_loc <- length(Y_obs)
  likeli <- -0.5 * (log_det + quad_form + n_loc * log(2*pi))
  
  if (any(!is.finite(innovation))) {
    stop("Innovation not finite")
  }

  if (any(!is.finite(K_t))) {
    stop("Kalman gain not finite")
  }
  
  return(list(
    X_updated = X_updated,
    Omega_updated = Omega_updated,
    likeli = likeli,
    Y_pred = Y_pred
  ))
}


# This is the loop that iterates over the Predict and Update functions
kalman_filter <- function(data, state_space) {
  
  
  n_loc <- state_space$n_loc
  T_len <- ncol(data)
  state_dim <- state_space$state_dim
  m = state_space$m
  likelihood <- 0
  
  # Initialize state
  X_t <- matrix(0, state_dim, 1)
  Omega_t <- diag(500, state_dim)
  Y_obs = matrix(0, nrow= n_loc, ncol = 1)
  
  for (t in 1:T_len) {
    # print(paste("Iteration:", t))
    # -------- Predict --------
    pred <- predict_step(
      X_t,
      Omega_t,
      state_space,
      Y_obs
    )
    
    X_pred <- pred$X_pred
    Omega_pred <- pred$Omega_pred
    
    # Observation vector (N x 1)
    Y_obs <- matrix(data[, t], ncol = 1)
    
    # -------- Update --------
    update <- update_step(
      state_space,
      X_pred,
      Omega_pred,
      Y_obs
    )
    
    X_t <- update$X_updated
    Omega_t <- update$Omega_updated
    
    likelihood <- likelihood + update$likeli
  }
  
  return(as.numeric(likelihood))
}

neg_loglik <- function(par, data, m, D) {
  
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
    
    Sigma_eta <- theta$sigma2_eta * exp(-D / theta$rho)
  }
  
  
  # ---- Build state-space matrices ----
  state_space <- create_state_space(
    n_loc = n_loc,
    m = m,
    psi,
    Sigma_eta = Sigma_eta,
    T_len = ncol(T_len),
    sigma2_w = theta$sigma2_w
  )
  
  # ---- Run Kalman filter ----
  lik <- tryCatch(
    kalman_filter(
      data = data,
      state_space = state_space
    ),
    error = function(e) {
      cat("Kalman error:", conditionMessage(e), "\n")
      return(NA)
    }
  )
  
  if (!is.finite(lik)) {
    message("Non-finite likelihood\n")
    return(1e12)
  }
  
  # ---- Return negative log-likelihood ----
  return(-as.numeric(lik))
}