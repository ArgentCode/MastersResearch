import numpy as np

# from Simulation import *
# from State_Space import *
# from Kalman import *
# from Parameters import *
# from Likelihood import *
# from Estimation import *
from McFullCode import *

def test_simulation():
    # --- parameters ---
    params = Parameters(
        d=0.3,
        lam=0.1,
        ar=[0.5],
        ma=[0.2],
        sigma2_eta=1.0,
        rho=1.0,
        spatial_model=1,
        tau2=0.5
    )
    
    # --- coordinates ---
    N = 10
    coords = np.random.rand(N, 2)  # random 2D coordinates
    
    # --- simulate process ---
    T = 100
    Z = simulate_artfima(params, coords, T)
    
    assert Z.shape == (T, N), f"Simulated data has wrong shape: {Z.shape}"
    assert not np.any(np.isnan(Z)), "Simulation contains NaNs"
    assert np.all(np.isfinite(Z)), "Simulation contains infs"
    
    print("✅ Simulation passed all checks.")


def test_state_space():
    # --- parameters ---
    params = Parameters(
        d=0.3,
        lam=0.1,
        ar=[0.5],
        ma=[0.2],
        sigma2_eta=1.0,
        rho=1.0,
        spatial_model=1,
        tau2=0.5
    )
    
    # --- coordinates ---
    N = 10
    coords = np.random.rand(N, 2)  # random 2D coordinates
    
    # --- build state space model ---
    m = 5
    ss = StateSpace.build_state_space(params, coords, m)
    
    dim = N * (m + 1)
    
    # --- dimension checks ---
    assert ss.F.shape == (dim, dim), f"F wrong shape: {ss.F.shape}"
    assert ss.Q.shape == (dim, dim), f"Q wrong shape: {ss.Q.shape}"
    assert ss.G.shape == (N, dim), f"G wrong shape: {ss.G.shape}"
    assert ss.R.shape == (N, N), f"R wrong shape: {ss.R.shape}"
    
    # --- symmetry checks (important!) ---
    assert np.allclose(ss.Q, ss.Q.T), "Q not symmetric"
    assert np.allclose(ss.R, ss.R.T), "R not symmetric"
    
    # --- positive semi-definite checks ---
    eig_Q = np.linalg.eigvalsh(ss.Q)
    eig_R = np.linalg.eigvalsh(ss.R)
    
    assert np.all(eig_Q >= -1e-8), "Q not PSD"
    assert np.all(eig_R >= -1e-8), "R not PSD"
    
    print("✅ State space model passed all checks.")


def test_likelihood():
    # --- parameters ---
    params = Parameters(
        d=0.3,
        lam=0.1,
        ar=[0.5],
        ma=[0.2],
        sigma2_eta=1.0,
        rho=1.0,
        spatial_model=1,
        tau2=0.5
    )
    
    # --- build Settings ---
    m = 5
    T = 100
    N = 10
    dim = N * (m + 1)
    coords = np.random.rand(N, 2)  # random 2D coordinates
    
    # --- simulate observation ---
    Y = simulate_artfima(params, coords, T)

    ll = kalman_loglik(params, Y, coords, m)
    ll_true = kalman_loglik(params, Y, coords, m)
    
    assert np.isfinite(ll), "Log-likelihood is not finite"

    n_sim = 5
    wins = 0
        # --- perturbed params ---
    params_bad = Parameters(
        d=0.1,  # wrong
        lam=0.2,
        ar=[0.2],
        ma=[0.1],
        sigma2_eta=2.0,
        rho=0.5,
        spatial_model=1,
        tau2=0.5
    )

    for i in range(n_sim):
        Y = simulate_artfima(params, coords, T)
        
        ll_bad = kalman_loglik(params_bad, Y, coords, m)
        
        if ll_true > ll_bad:
            wins += 1

    assert wins >= 3, "Likelihood does not prefer true parameters often enough"
    print("✅ Log-likelihood calculation passed all checks.")

def test_Estimation():
    np.random.seed(0)
    
    # --- true parameters ---
    params = Parameters(
        d=0,
        lam=0,
        ar=[0.7],
        ma=[0.2],
        sigma2_eta=1.0,
        rho=1.0,
        spatial_model=1,
        tau2=0
    )
    
    # --- coordinates ---
    N = 25
    coords = np.random.rand(N, 2)
    
    # --- simulate data ---
    T = 100
    Y = simulate_artfima(params, coords, T)
    
    # --- estimation settings ---
    m = 5
    init_params = Parameters(
        d=0,
        lam=0,
        ar=[0.5],
        ma=[0.5],
        sigma2_eta=1.5,
        rho=0.8,
        spatial_model=1,
        tau2=0
    )
    
    free_params = ["ar", "ma", "sigma2_eta", "rho"]
    
    # --- estimate ---
    est_params, result = estimate_params(
        Y,
        coords,
        m,
        base_params=init_params,
        free_params=free_params
    )
    
    # --------------------------------------------------
    # ✅ 1. Optimizer success
    # --------------------------------------------------
    assert result.success, f"Optimization failed: {result.message}"
    
    # --------------------------------------------------
    # ✅ 2. Likelihood improvement (VERY important)
    # --------------------------------------------------
    ll_init = kalman_loglik(init_params, Y, coords, m)
    ll_est  = kalman_loglik(est_params, Y, coords, m)
    
    assert ll_est >= ll_init - 1e-8, "Likelihood did not improve"
    
    # --------------------------------------------------
    # ✅ 3. Fixed parameters stayed fixed
    # --------------------------------------------------
    assert est_params.d == init_params.d, "d changed but should be fixed"
    assert est_params.lam == init_params.lam, "lam changed but should be fixed"
    assert est_params.tau2 == init_params.tau2, "tau2 changed but should be fixed"
    
    # --------------------------------------------------
    # ✅ 4. Parameter bounds / validity
    # --------------------------------------------------
    assert est_params.sigma2_eta > 0, "sigma2_eta not positive"
    assert est_params.rho > 0, "rho not positive"
    assert -1 < est_params.ar[0] < 1, "AR out of bounds"
    assert -1 < est_params.ma[0] < 1, "MA out of bounds"
    
    # --------------------------------------------------
    # ✅ 5. Finite values
    # --------------------------------------------------
    for val in vars(est_params).values():
        if isinstance(val, (float, int)):
            assert np.isfinite(val), "Non-finite parameter estimate"
    
    # --------------------------------------------------
    # ✅ 6. Improvement toward truth (soft check)
    # --------------------------------------------------
    def dist(p1, p2):
        return (
            abs(p1.ar[0] - p2.ar[0]) +
            abs(p1.ma[0] - p2.ma[0]) +
            abs(p1.sigma2_eta - p2.sigma2_eta) +
            abs(p1.rho - p2.rho)
        )
    
    dist_init = dist(init_params, params)
    dist_est  = dist(est_params, params)
    
    assert dist_est <= dist_init + 1e-6, "Estimates not closer to truth"
    
    print("✅ Parameter estimation passed all checks.")

if __name__ == "__main__":
    test_simulation()
    test_state_space()
    test_likelihood()
    test_Estimation()