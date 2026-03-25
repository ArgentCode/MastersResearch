import numpy as np

#### NO longer in regular use

# kalman.py
def predict_step(ss, X, Omega):
    X_pred = ss.F @ X
    Omega_pred = ss.F @ Omega @ ss.F.T + ss.Q
    return X_pred, Omega_pred


def update_step(state_space, X_pred, Omega_pred, Y_obs):
    G = state_space.G
    R = state_space.R
    
    # Innovation
    Y_pred = G @ X_pred
    v = Y_obs - Y_pred
    
    S = G @ Omega_pred @ G.T + R
    
    # Kalman gain
    K = Omega_pred @ G.T @ np.linalg.inv(S)
    
    # Update
    X_new = X_pred + K @ v
    Omega_new = Omega_pred - K @ G @ Omega_pred
    
    return X_new, Omega_new, v, S