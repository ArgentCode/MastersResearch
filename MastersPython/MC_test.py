import numpy as np

from McFullCode import *


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
    
    # --------------------------------------------------
    # INITIAL GUESS
    # --------------------------------------------------
    init_params = Parameters(
        d=0.0,                 # FIXED
        lam=0.0,               # FIXED
        ar=[0.2],
        ma=[0.1],
        sigma2_eta=1.0,
        rho=0.6,
        spatial_model=1,
        tau2=0.0001               # FIXED
    )
    
    # --------------------------------------------------
    # WHICH PARAMETERS TO ESTIMATE
    # --------------------------------------------------
    free_params = ["ar", "ma", "sigma2_eta", "rho"]
    
    # --------------------------------------------------
    # SPATIAL LOCATIONS
    # --------------------------------------------------
    coords = np.random.rand(N, 2)
    
    # --------------------------------------------------
    # RUN MONTE CARLO
    # --------------------------------------------------
    results = run_monte_carlo(
        true_params=true_params,
        init_params=init_params,
        coords=coords,
        T=T,
        m=m,
        free_params=free_params,
        n_iter=n_iter,
        output_file="mc_test_1.txt"
    )
    
    print("\n🔥 DONE. Check mc_test_1.txt 🔥")


if __name__ == "__main__":
    main()