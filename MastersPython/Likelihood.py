import numpy as np
from State_Space import *
from Kalman import *


# def kalman_loglik(params, Y, coords, m):
#     T, N = Y.shape
    
#     # --- build state space ---
#     ss = StateSpace.build_state_space(params, coords, m)
    
#     dim = ss.F.shape[0]
    
#     # --- initialize ---
#     X = np.zeros(dim)
#     Omega = np.eye(dim) * 1e3  # diffuse
    
#     loglik = 0.0
    
#     for t in range(T):
        
#         # --- predict ---
#         X_pred = ss.F @ X
#         Omega_pred = ss.F @ Omega @ ss.F.T + ss.Q
        
#         # --- update (now returns v, S) ---
#         try:
#             # Innovation
#             Y_pred = ss.G @ X_pred
#             Y_obs = Y[t]
#             v = Y_obs - Y_pred
            
#             S = ss.G @ Omega_pred @ ss.G.T + ss.R
            
#             # Kalman gain
#             K = Omega_pred @ ss.G.T @ np.linalg.inv(S)
            
#             # Update
#             X_new = X_pred + K @ v
#             Omega_new = Omega_pred - K @ ss.G @ Omega_pred
            
#             # --- likelihood contribution ---
#             sign, logdet = np.linalg.slogdet(S)
#             if sign <= 0:
#                 return np.inf
            
#             quad = v.T @ np.linalg.solve(S, v)
            
#             loglik += -0.5 * (logdet + quad + N * np.log(2 * np.pi))
        
#         except np.linalg.LinAlgError:
#             return np.inf
    
#     return loglik

def kalman_loglik(params, Y, coords, m):
    T, N = Y.shape
    
    # --- build state space ---
    ss = StateSpace.build_state_space(params, coords, m)
    
    F = ss.F
    G = ss.G
    Q = ss.Q   # this is C^{HV}
    R = ss.R
    
    dim = F.shape[0]
    
    # --- initialize ---
    X = np.zeros(dim)
    Omega = np.eye(dim) * 1e3  # diffuse
    
    loglik = 0.0
    
    for t in range(T):
        try:
            # --------------------------------------------------
            # Innovation (Ferreira: Y_t - Ŷ_t)
            # --------------------------------------------------
            Y_pred = G @ X
            v = Y[t] - Y_pred
            
            # --------------------------------------------------
            # Delta_t = Var(Y_t - Ŷ_t)
            # --------------------------------------------------
            eps = 1e-6
            Delta = G @ Omega @ G.T + R + eps * np.eye(N)
            
            # --------------------------------------------------
            # Theta_t = Cov(X_{t+1}, innovation)
            # --------------------------------------------------
            Theta = F @ Omega @ G.T
            
            # --------------------------------------------------
            # State update (Eq 11a)
            # --------------------------------------------------
            X = F @ X + Theta @ np.linalg.solve(Delta, v)
            
            # --------------------------------------------------
            # Covariance update (Eq 11b)
            # --------------------------------------------------
            Omega = (
                F @ Omega @ F.T
                + Q
                - Theta @ np.linalg.solve(Delta, Theta.T)
            )
            Omega = 0.5 * (Omega + Omega.T)  # ensure symmetry
            
            # --------------------------------------------------
            # Likelihood
            # --------------------------------------------------
            sign, logdet = np.linalg.slogdet(Delta)
            if sign <= 0:
                return np.inf
            
            quad = v.T @ np.linalg.solve(Delta, v)
            
            loglik += -0.5 * (logdet + quad + N * np.log(2 * np.pi))

            if not np.isfinite(loglik):
                print("Bad params:", params)
        
        except np.linalg.LinAlgError:
            return np.inf
    
    return loglik