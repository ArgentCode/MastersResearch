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
  
  # # -------- F matrix --------
  # # State Transition
  # F <- matrix(0, state_dim, state_dim)
  # 
  # for (j in 2:(m+1)) {
  #   row_idx <- ((j-1)*n_loc + 1):(j*n_loc)
  #   col_idx <- ((j-2)*n_loc + 1):((j-1)*n_loc)
  #   F[row_idx, col_idx] <- diag(n_loc)
  # }
  
  # # -------- H matrix --------
  # # Mean structure
  # H <- matrix(0, state_dim, n_loc)
  # H[1:n_loc, 1:n_loc] <- diag(n_loc)
  
  return(list(
    # G = G,
    # F = F,
    psi = psi,
    # H = H,
    R = R,
    Sigma_eta = Sigma_eta
  ))
}


predict_step <- function(X_t, Omega_t, Sigma_eta, n_loc, state_dim, m) {
  
  # ---- Shift state ----
  # One-step ahead state prediction
  # X_pred <- F %*% X_t
  # O(m^2N^2) -> O(mN)
  X_pred <- matrix(0, state_dim, 1)
  
  for (j in 1:m) {
    dest <- (j*n_loc + 1):((j+1)*n_loc)
    src  <- ((j-1)*n_loc + 1):(j*n_loc)
    X_pred[dest] <- X_t[src]
  }
  
  # ---- Shift covariance ----
  # State covariance prediction
  # Omega_pred <- F %*% Omega_t %*% t(F) + Q
  # O(m^3N^3) -> O(m^2N^2)
  Omega_pred <- matrix(0, state_dim, state_dim)
  
  for (i in 1:m) {
    for (j in 1:m) {
      
      row_new <- (i*n_loc + 1):((i+1)*n_loc)
      col_new <- (j*n_loc + 1):((j+1)*n_loc)
      
      row_old <- ((i-1)*n_loc + 1):(i*n_loc)
      col_old <- ((j-1)*n_loc + 1):(j*n_loc)
      
      Omega_pred[row_new, col_new] <-
        Omega_t[row_old, col_old]
    }
  }
  
  Omega_pred[1:n_loc, 1:n_loc] <-
    Omega_pred[1:n_loc, 1:n_loc] + Sigma_eta
  
  # ---- Safety checks ----
  # if (max(abs(Omega_pred - t(Omega_pred))) > 1e-8)
  #   stop("Omega_pred not symmetric")
  
  return(list(
    X_pred = X_pred,
    Omega_pred = Omega_pred
  ))
}

update_step <- function(X_pred, Omega_pred, Y_obs,
                        psi, R) {
  
  n_loc <- nrow(Y_obs)
  state_dim <- length(X_pred)
  m <- state_dim / n_loc - 1
  
  # Innovation
  # Y_pred <- G %*% X_pred
  #O(Nr) -> O(N^2m)
  #############################################
  X_mat <- matrix(X_pred, n_loc, m+1)
  Y_pred <- X_mat %*% psi
  #############################################
  innovation <- Y_obs - Y_pred
  
  # Innovation covariance
  # S_t <- G %*% Omega_pred %*% t(G) + R
  # O(m^2N^3) -> O(m^2N^2)
  #############################################
  
  S_t <- matrix(0, n_loc, n_loc)
  
  for (i in 0:m) {
    for (j in 0:m) {
      row_idx <- (i*n_loc + 1):((i+1)*n_loc)
      col_idx <- (j*n_loc + 1):((j+1)*n_loc)
      
      Omega_ij <- Omega_pred[row_idx, col_idx, drop = FALSE]
      
      S_t <- S_t + psi[i+1] * psi[j+1] * Omega_ij
    }
  }
  
  S_t <- S_t + R
  #############################################
  
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
  # K_t <- Omega_pred %*% t(G) %*% S_inv
  # O(m^2N^3) -> O(m^2N^2)
  #################################################
  Omega_Gt <- matrix(0, state_dim, n_loc)
  
  for (i in 0:m) {
    
    row_idx <- (i*n_loc + 1):((i+1)*n_loc)
    
    block_sum <- matrix(0, n_loc, n_loc)
    
    for (j in 0:m) {
      col_idx <- (j*n_loc + 1):((j+1)*n_loc)
      Omega_ij <- Omega_pred[row_idx, col_idx, drop = FALSE]
      block_sum <- block_sum + psi[j+1] * Omega_ij
    }
    
    Omega_Gt[row_idx, ] <- block_sum
  }
  
  K_t <- Omega_Gt %*% S_inv
  ######################################################
  
  # State update
  X_updated <- X_pred + K_t %*% innovation
  
  # Covariance update
  Omega_updated <- Omega_pred - K_t %*% S_t %*% t(K_t)
  
  # Log-likelihood
  log_det <- as.numeric(determinant(S_t, logarithm = TRUE)$modulus)
  quad_form <- as.numeric(t(innovation) %*% S_inv %*% innovation)
  
  n_loc <- length(Y_obs)
  likeli <- -0.5 * (log_det + quad_form + n_loc * log(2*pi))
  
  # if (any(!is.finite(innovation))) {
  #   stop("Innovation not finite")
  # }
  # 
  # if (any(!is.finite(K_t))) {
  #   stop("Kalman gain not finite")
  # }
  
  return(list(
    X_updated = X_updated,
    Omega_updated = Omega_updated,
    likeli = likeli,
    Y_pred = Y_pred
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
      # state_space$F,
      # state_space$H,
      state_space$Sigma_eta,
      n_loc, state_dim, m
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
      # state_space$G,
      state_space$psi,
      state_space$R
    )
    
    X_t <- update$X_updated
    Omega_t <- update$Omega_updated
    
    likelihood <- likelihood + update$likeli
  }
  
  return(as.numeric(likelihood))
}

neg_loglik <- function(par, data, m, n_side, D) {
  
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
  # state_space <- create_state_space(
  #   n_loc = nrow(data),
  #   m = m,
  #   sigma2_w = theta$sigma2_w,
  #   psi = psi,
  #   Sigma_eta = Sigma_eta
  # )
  
  state_space <- list(
    R = diag(theta$sigma2_w, n_loc),
    psi = psi,
    Sigma_eta = Sigma_eta
  )
  
  # ---- Run Kalman filter ----
  lik <- tryCatch(
    kalman_filter(
      data = data,
      state_space = state_space,
      m = m
    ),
    error = function(e) {
      cat("Kalman error:", conditionMessage(e), "\n")
      return(NA)
    }
  )
  
  # if (!is.finite(lik)) {
  #   message("Non-finite likelihood\n")
  #   return(1e12)
  # }
  
  # message(lik)
  
  # ---- Return negative log-likelihood ----
  return(-as.numeric(lik))
}