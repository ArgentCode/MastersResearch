library(doParallel)
library(foreach)

mc_one_run <- function(i, true_theta, start_vals, T_len, n_side, m, lower, upper) {
  
  # Simulate
  mat <- simulate_artfima_spatial(
    T_len = T_len,
    n_side = n_side,
    m = m,
    true_theta = true_theta
  )
  
  if (n_side > 1) {
    
    coords <- expand.grid(x = 1:n_side, y = 1:n_side)
    coords <- as.matrix(coords)
    
    D <- as.matrix(dist(coords))
  } else {
    D <- null
  }
  
  # Estimate
  fit <- tryCatch(
    nlminb(
      start = start_vals,
      objective = neg_loglik,
      data = mat,
      m = m,
      n_side = n_side,
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

run_MC <- function(cluster, true_theta, hat_theta, m, iter, n_side, T, lower = NULL, upper = NULL) {
  
  if (is.null(upper)) {
    bounds <- get_bounds(hat_theta)
    lower <- bounds$lower
    upper <- bounds$upper
  }
  
  doParallel::registerDoParallel(cluster)
  
  start_time <- Sys.time()

  estimates <- foreach(
    i = 1:iter,
    .combine = rbind
  ) %dopar% {
    
    tryCatch({
      
      res <- mc_one_run(
        i,
        true_theta = true_theta,
        start_vals = hat_theta,
        T_len = T,
        n_side = n_side,
        m = m,
        lower = lower,
        upper = upper
      )
      
      if (any(!is.finite(res))) {
        stop(paste("Non-finite result on iteration", i))
      }
      
      res
      
    }, error = function(e) {
      
      msg <- paste("Worker error on iteration", i, ":", e$message)
      cat(msg, "\n")
    })
  }
  
  message(
    sprintf(
      "Finished %d simulations | Total time: %.2f mins",
      iter,
      as.numeric(difftime(Sys.time(), start_time, units = "mins"))
    )
  )
  
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
  
  SqrtMSE <- sqrt(mse)
  starting <- unlist(hat_theta)[names(hat_theta)]
  
  results <- rbind(
    True = true_vec,
    Mean = mean_est,
    sd = sd_est,
    RelBias = rel_bias,
    SqrtMSE = SqrtMSE,
    starting = starting
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
  
  results <- run_MC(cluster, true_theta, hat_theta, 
                      m, iter, n_side, T_len)
  
  end_time <- Sys.time()
  print(results)
  
  cat("\nRuntime:",
      round(difftime(end_time, start_time, units = "mins"), 2),
      "minutes\n\n")
}