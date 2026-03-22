import numpy as np

from Parameters import *
from Monte_Carlo import *


def main():
    np.random.seed(42)
    
    # --------------------------------------------------
    # SETTINGS
    # --------------------------------------------------
    T = 100
    m = 5
    N = 25
    n_iter = 10
    
    # --------------------------------------------------
    # TRUE PARAMETERS (ARMA(1,1))
    # --------------------------------------------------
    true_params = Parameters(
        d=0.0,                 # FIXED
        lam=0.0,               # FIXED
        ar=[0.45],             # AR(1)
        ma=[0.3],              # MA(1)
        sigma2_eta=0.5,
        rho=0.35,
        spatial_model=1,
        tau2=0.0001               # FIXED
    )

    coords = np.random.rand(N, 2)

    Y = simulate_artfima(true_params, coords, T)
    
    # --------------------------------------------------
    # INITIAL GUESS
    # --------------------------------------------------
    ll_true = kalman_loglik(true_params, Y, coords, m)

    test_params = Parameters(
        d=0.0,
        lam=0.0,
        ar=[0.45],
        ma=[-0.1886],   # your estimate
        sigma2_eta=0.5,
        rho=0.35,
        spatial_model=1,
        tau2=0.0
    )

    ll_est = kalman_loglik(test_params, Y, coords, m)

    print(ll_true, ll_est)

    # psi_true = psi_artfima(m, d=0, lam=0, ar=[0.45], ma=[0.3])
    # psi_flip = psi_artfima(m, d=0, lam=0, ar=[0.45], ma=[-0.1886])

    # print(psi_true)
    # print(psi_flip)

if __name__ == "__main__":
    main()