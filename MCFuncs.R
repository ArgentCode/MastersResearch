mc_one_run <- function(i, true_theta, start_vals, T_len, n_side, m, lower, upper) {
  
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
      n_side = n_side,
      lower = lower,
      upper = upper
    ),
    error = function(e) return(NULL)
  )
  
  if (!is.null(fit) && fit$convergence == 0) {
    return(fit$par)
  } else {
    return(rep(NA, length(start_vals)))
  }
}

run_MCMC = function(cluster, true_theta, hat_theta, m, iter, n_side, T, lower = NULL, upper = NULL) {
  
  if (is.null(upper)) {
    bounds = get_bounds(hat_theta)
    lower = bounds$lower
    upper = bounds$upper
  }
  
  chunk_size <- length(cluster)  # or maybe 20
  results_list <- vector("list", iter)
  
  start_time <- Sys.time()
  
  for (start in seq(1, iter, by = chunk_size)) {
    
    end <- min(start + chunk_size - 1, iter)
    
    chunk_res <- parLapply(
      cluster,
      start:end,
      mc_one_run,
      true_theta = true_theta,
      start_vals = hat_theta,
      T_len = T,
      n_side = n_side,
      m = m,
      lower = lower,
      upper = upper
    )
    
    results_list[start:end] <- chunk_res
    
    message(
      sprintf(
        "Completed %d of %d | Elapsed: %.2f mins",
        end,
        iter,
        as.numeric(difftime(Sys.time(), start_time, units = "mins"))
      )
    )
  }
  
  estimates <- do.call(rbind, results_list)
  true_vec <- unlist(true_theta)[names(hat_theta)]
  
  mean_est <- colMeans(estimates, na.rm = TRUE)
  sd_est <- apply(estimates, 2, sd, na.rm = TRUE)
  
  bias <- mean_est - true_vec
  rel_bias <- ifelse(true_vec == 0, NA, bias / true_vec)
  
  mse <- colMeans(
    (estimates -
       matrix(true_vec,
              nrow = iter,
              ncol = length(true_vec),
              byrow = TRUE))^2,
    na.rm = TRUE
  )
  SqrtMSE = sqrt(mse)
  
  results <- rbind(
    True = true_vec,
    Mean = mean_est,
    sd = sd_est,
    RelBias = rel_bias,
    SqrtMSE = SqrtMSE,
    starting = hat_theta
  )
  
  return(round(results,4))
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

run_one_setting <- function(cluster, true_theta, hat_theta, 
                            m, iter, n_side, T_len) {
  
  cat("--------------------------------------------------\n")
  cat("Settings:\n")
  cat("T =", T_len, 
      " | n_side =", n_side,
      " | m =", m,
      " | iter =", iter, "\n\n")
  
  start_time <- Sys.time()
  
  results <- run_MCMC(cluster, true_theta, hat_theta, 
                      m, iter, n_side, T_len)
  
  end_time <- Sys.time()
  
  print(results)
  
  cat("\nRuntime:",
      round(difftime(end_time, start_time, units = "mins"), 2),
      "minutes\n\n")
}