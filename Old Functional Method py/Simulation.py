import numpy as np
from Parameters import *
from Kalman import *
from State_Space import *


def simulate_artfima(params, coords, T, m, burnin=200):
    """
    Simulate spatio-temporal ARTFIMA process.
    
    Returns
    -------
    Z : (T, N) array
    """

    m = m*100
    
    N = coords.shape[0]
    
    # --- ψ weights ---
    psi = psi_artfima(
        m,
        params.d,
        params.lam,
        params.ar,
        params.ma
    )

    # --- spatial correlation (no variance) ---
    R = build_spatial_cov(
        coords,
        sigma2=1.0,
        rho=params.rho,
        model=params.spatial_model
    )

    # --- innovations ---
    total_T = T + burnin
    eta = np.random.multivariate_normal(
        mean=np.zeros(N),
        cov=params.sigma2_eta * R,
        size=total_T
    )
    
    # --- generate process ---
    Z = np.zeros((total_T, N))
    
    for t in range(m, total_T):
        for k in range(m + 1):
            Z[t] += psi[k] * eta[t - k]
    
    # remove burn-in
    return Z[burnin:]