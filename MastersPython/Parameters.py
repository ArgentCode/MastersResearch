import numpy as np
from statsmodels.tsa.arima_process import arma2ma
from scipy.spatial.distance import cdist

# D and lambda components
def psi_frac(m, d, lam):
    psi = np.zeros(m + 1)
    psi[0] = 1.0
    
    for k in range(1, m + 1):
        psi[k] = psi[k - 1] * ((k - 1 + d) / k) * np.exp(-lam)
    
    return(psi)


# ARMA components
def psi_arma(m, ar=None, ma=None):
    if ar is None:
        ar = np.array([])
    if ma is None:
        ma = np.array([])
    
    # statsmodels uses opposite sign convention for AR
    ar = np.asarray(ar)
    ma = np.asarray(ma)
    
    psi_tail = arma2ma(ar=ar, ma=ma, lags=m)
    
    return(np.concatenate(([1.0], psi_tail)))


def psi_artfima(m, d, lam, ar=None, ma=None):
    psi_f = psi_frac(m, d, lam)
    psi_a = psi_arma(m, ar, ma)
    
    psi = np.convolve(psi_a, psi_f)[:m + 1]
    
    return(psi)


def build_spatial_cov(coords, sigma2=1.0, rho=1.0, model=1):
    """
    Build spatial covariance matrix.
    
    Parameters
    ----------
    coords : (N, d) array
        Spatial coordinates
    sigma2 : float
        Marginal variance
    rho : float
        Range parameter
    model : int
        1 = exponential (nu = 1/2)
        2 = Matern (nu = 3/2)
    """
    
    # Pairwise distances
    D = cdist(coords, coords)  # shape (N, N)
    
    if model == 1:
        # Exponential
        Sigma = sigma2 * np.exp(-D / rho)
        
    elif model == 2:
        # Matern ν = 3/2
        scaled_D = D / rho
        Sigma = sigma2 * (1 + scaled_D) * np.exp(-scaled_D)
        
    else:
        raise ValueError("model must be 1 (exp) or 2 (matern 3/2)")
    
    return Sigma