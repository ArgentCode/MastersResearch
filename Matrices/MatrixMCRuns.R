mc_one_run <- function(i, true_theta, start_vals, T_len, n_side, m, lower, upper, D) {
  
  
  # message(paste("Beginning Run:", i))
  
  if (i %% 5 == 0) {
    message(paste("Beginning Run:", i))
  }
  
  # Simulate
  mat <- simulate_artfima_spatial(
    T_len = T_len,
    n_side = n_side,
    m = m,
    true_theta = true_theta
  )
  
  # Estimate
  fit <- tryCatch(
    nlminb(
      start = start_vals,
      objective = neg_loglik,
      data = mat,
      m = m,
      lower = lower,
      upper = upper,
      D=D
    ),
    error = function(e) return(NULL)
  )
  
  if (!is.null(fit) && fit$convergence == 0) {
    return(fit$par)
  } else {
    return(rep(NA, length(start_vals)))
  }
}


get_bounds <- function(hat_theta) {
  lower_template <- c(
    d        = -0.49,
    lambda   = 1e-6,
    phi      = -0.99,
    theta    = -0.99,
    sigma2_eta  = 1e-6,
    sigma2_w    = 1e-6,
    rho         = -0.99
  )
  
  upper_template <- c(
    d        = 0.49,
    lambda   = 5,
    phi      = 0.99,
    theta    = 0.99,
    sigma2_eta  = 10,
    sigma2_w    = 10,
    rho         = 0.99
  )
  
  lower <- lower_template[names(hat_theta)]
  upper <- upper_template[names(hat_theta)]
  list(lower = lower, upper = upper)
}