import numpy as np
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from scipy.optimize import minimize
from statsmodels.tsa.arima_process import arma2ma

##############################
# Parameters
##############################

# D and lambda components
def psi_frac(m, d, lam):
    psi = np.zeros(m + 1)
    psi[0] = 1.0

    for k in range(1, m + 1):
        psi[k] = psi[k - 1] * ((k - 1 + d) / k) * np.exp(-lam)

    return(psi)


def psi_arma(m, ar=None, ma=None):
    if ar is None:
        ar = np.array([])
    if ma is None:
        ma = np.array([])

    ar = np.asarray(ar)
    ma = np.asarray(ma)

    # --- correct polynomial form ---
    ar_poly = np.r_[1.0, -ar]   # (1 - φB)
    ma_poly = np.r_[1.0, ma]    # (1 + θB)

    psi_tail = arma2ma(ar=ar_poly, ma=ma_poly, lags=m)

    psi = np.empty(m + 1)
    psi[0] = 1.0
    psi[1:] = psi_tail[:m]

    return psi


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

##############################
# Simulation
##############################

# def simulate_arma_spatial(params, coords, T, burnin=200):
#     """
#     Simulate ARMA(1,1) with spatially correlated innovations
#     """
#     N = coords.shape[0]

#     # --- spatial covariance ---
#     Sigma = build_spatial_cov(
#         coords,
#         sigma2=params.sigma2_eta,
#         rho=params.rho,
#         model=params.spatial_model
#     )

#     # --- innovations ---
#     eta = np.random.multivariate_normal(
#         mean=np.zeros(N),
#         cov=Sigma,
#         size=T + burnin
#     )

#     # --- process ---
#     eps = np.zeros((T + burnin, N))

#     phi = params.ar[0]
#     theta = params.ma[0]

#     for t in range(1, T + burnin):
#         eps[t] = (
#             phi * eps[t-1]
#             + theta * eta[t-1]
#             + eta[t]
#         )

#     return eps[burnin:]
  
def simulate_artfima_spatial(params, coords, T, m=50, burnin=200):
    """
    Fully consistent simulation using ψ representation
    (works for both ARMA and ARTFIMA)
    """
    N = coords.shape[0]

    # --- spatial covariance ---
    Sigma = build_spatial_cov(
        coords,
        sigma2=params.sigma2_eta,
        rho=params.rho,
        model=params.spatial_model
    )

    # --- ψ weights ---
    psi = psi_artfima(
        m,
        params.d,
        params.lam,
        params.ar,
        params.ma
    )

    total_T = T + burnin + m

    # --- innovations ---
    eta = np.random.multivariate_normal(
        mean=np.zeros(N),
        cov=Sigma,
        size=total_T
    )

    # --- process ---
    Z = np.zeros((total_T, N))

    for t in range(m, total_T):
        # vectorized dot product instead of loop
        Z[t] = psi @ eta[t - np.arange(m + 1)]

    return Z[burnin:]

##############################
# State_Space
##############################

@dataclass
class Parameters:
    d: float
    lam: float
    ar: list
    ma: list
    sigma2_eta: float
    rho: float
    spatial_model: int
    tau2: float = 0.0

    def __post_init__(self):
        self.ar = np.asarray(self.ar)
        self.ma = np.asarray(self.ma)
        
        assert self.sigma2_eta > 0, "sigma2_eta must be positive"
        assert self.rho > 0, "rho must be positive"
        assert self.spatial_model in [1, 2], "model must be 1 or 2"

    def to_vector(self):
        return np.array([
            self.d,
            self.lam,
            *self.ar,
            *self.ma,
            self.sigma2_eta,
            self.rho,
            self.tau2
        ])


@dataclass
class StateSpace:
    F: np.ndarray
    Q: np.ndarray
    G: np.ndarray
    R: np.ndarray

    @staticmethod
    def build_state_space(params, coords, m):
        N = coords.shape[0]
        
        # --------------------------------------------------
        # ψ weights (MA representation)
        # --------------------------------------------------
        #psi = psi_artfima(m, params.d, params.lam, params.ar, params.ma)
        psi = psi_artfima(
            m,
            params.d,
            params.lam,
            params.ar,
            params.ma
        )
        
        # --------------------------------------------------
        # Temporal transition (shift innovations)
        # X_t = [η_t, η_{t-1}, ..., η_{t-m}]
        # --------------------------------------------------
        F = np.zeros((m+1, m+1))
        F[1:, :-1] = np.eye(m)   # shift down
        
        # --------------------------------------------------
        # Lift to spatial
        # --------------------------------------------------
        F_full = np.kron(F, np.eye(N))
        
        # --------------------------------------------------
        # Spatial covariance (correlation only)
        # --------------------------------------------------
        Sigma_S = build_spatial_cov(
            coords,
            sigma2=1.0,
            rho=params.rho,
            model=params.spatial_model
        )
        
        # --------------------------------------------------
        # Process noise (η_t enters first block)
        # --------------------------------------------------
        Q_base = np.zeros((m+1, m+1))
        Q_base[0, 0] = params.sigma2_eta
        
        Q = np.kron(Q_base, Sigma_S)
        
        # --------------------------------------------------
        # Observation matrix
        # Z_t = sum ψ_k η_{t-k}
        # --------------------------------------------------
        G_blocks = []
        for k in range(m + 1):
            G_blocks.append(psi[k] * np.eye(N))
        
        G = np.hstack(G_blocks)   # shape (N, N*(m+1))
        
        # --------------------------------------------------
        # Measurement noise
        # --------------------------------------------------
        R = params.tau2 * np.eye(N)
        
        return StateSpace(F_full, Q, G, R)
    
##############################
# Kalman
##############################
    
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
    # Omega = np.eye(dim) * 1e3  # diffuse
    # Spatial covariance (FULL, with sigma2)
    Sigma_S = build_spatial_cov(
        coords,
        sigma2=params.sigma2_eta,
        rho=params.rho,
        model=params.spatial_model
    )
    
    # State covariance: block diagonal
    Omega = np.kron(np.eye(m+1), Sigma_S)
    
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

##############################
# Estimation
##############################

PARAM_ORDER = [
    "d",
    "lam",
    "ar",
    "ma",
    "sigma2_eta",
    "rho",
    "tau2"
]


def params_to_vector(params, free_params):
    vec = []
    
    for name in PARAM_ORDER:
        if name not in free_params:
            continue
        
        val = getattr(params, name)
        
        if name in ["ar", "ma"]:
            vec.extend(val)
        else:
            vec.append(val)
    
    return np.array(vec)

def vector_to_params(theta, base_params, free_params):
    """
    base_params = full parameter object (provides fixed values)
    """
    
    params_dict = vars(base_params).copy()
    
    idx = 0
    
    for name in PARAM_ORDER:
        if name not in free_params:
            continue
        
        if name in ["ar", "ma"]:
            k = len(params_dict[name])
            params_dict[name] = list(theta[idx:idx+k])
            idx += k
        else:
            params_dict[name] = theta[idx]
            idx += 1
    
    return Parameters(**params_dict)


def build_bounds(base_params, free_params):
    bounds = []
    
    for name in PARAM_ORDER:
        if name not in free_params:
            continue
        
        if name == "d":
            bounds.append((1e-3, 0.49))
        elif name == "lam":
            bounds.append((1e-4, 10))
        elif name == "sigma2_eta":
            bounds.append((1e-6, 10))
        elif name == "rho":
            bounds.append((1e-4, 10))
        elif name == "tau2":
            bounds.append((1e-6, 10))
        elif name in ["ar", "ma"]:
            k = len(getattr(base_params, name))
            bounds.extend([(-0.99, 0.99)] * k)
    
    return bounds


def objective(theta, Y, coords, m, base_params, free_params):
    try:
        params = vector_to_params(theta, base_params, free_params)
        ll = kalman_loglik(params, Y, coords, m)
        return -ll
    except Exception:
        return np.inf
    

def estimate_params(Y, coords, m, base_params, free_params):
    
    theta0 = params_to_vector(base_params, free_params)
    bounds = build_bounds(base_params, free_params)
    
    result = minimize(
        objective,
        theta0,
        args=(Y, coords, m, base_params, free_params),
        method="L-BFGS-B",
        bounds=bounds
    )
    
    est_params = vector_to_params(result.x, base_params, free_params)
    
    return est_params, result

##############################
# Monte_Carlo
##############################


def params_to_array(params):
    return np.array([
        params.d,
        params.lam,
        *params.ar,
        *params.ma,
        params.sigma2_eta,
        params.rho,
        params.tau2
    ])


def run_monte_carlo(
    true_params,
    init_params,
    coords,
    T,
    m,
    free_params,
    n_iter,
    output_file="mc_results.txt"
):
    
    np.random.seed(0)
    
    theta_true = params_to_array(true_params)
    n_params = len(theta_true)
    
    estimates = np.zeros((n_iter, n_params))
    
    # --------------------------------------------------
    # MONTE CARLO LOOP
    # --------------------------------------------------
    for i in range(n_iter):
        if (i + 1) % 5 == 0:
          print(f"Iteration {i+1}/{n_iter}")

        Y = simulate_artfima_spatial(true_params, coords, T, m)
          
        try:
            est_params, result = estimate_params(
                Y,
                coords,
                m,
                base_params=init_params,
                free_params=free_params
            )
            
            if not result.success:
                print(f"  ⚠️ Optimizer failed at iter {i}")
                estimates[i, :] = np.nan
                continue
            
            estimates[i, :] = params_to_array(est_params)
        
        except Exception as e:
            print(f"  ⚠️ Error at iter {i}: {e}")
            estimates[i, :] = np.nan
    
    # --------------------------------------------------
    # CLEAN RESULTS
    # --------------------------------------------------
    valid = ~np.isnan(estimates).any(axis=1)
    estimates = estimates[valid]
    
    iter_eff = estimates.shape[0]
    
    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------
    mean_est = np.mean(estimates, axis=0)
    sd_est   = np.std(estimates, axis=0)
    
    rel_bias = mean_est / theta_true - 1
    mse      = np.mean((estimates - theta_true)**2, axis=0)
    
    # --------------------------------------------------
    # PARAM NAMES
    # --------------------------------------------------
    names = (
        ["d", "lam"] +
        [f"ar{i+1}" for i in range(len(true_params.ar))] +
        [f"ma{i+1}" for i in range(len(true_params.ma))] +
        ["sigma2_eta", "rho", "tau2"]
    )

    # --------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------
    # --------------------------------------------------
    # Select only estimated (free) parameters
    # --------------------------------------------------
    def is_selected(name, free_params):
        for fp in free_params:
            if fp in ["ar", "ma"]:
                if name.startswith(fp):   # catches ar1, ar2, etc.
                    return True
            else:
                if name == fp:
                    return True
        return False

    param_indices = [i for i, name in enumerate(names) if is_selected(name, free_params)]
    # param_indices = [i for i, name in enumerate(names) if name in free_params]
    selected_names = [names[i] for i in param_indices]

    # Pretty labels (optional but nice)
    def pretty_name(name):
        if name == "sigma2_eta":
            return "sigma2"
        elif name == "rho":
            return "rho"
        elif name == "d":
            return "d"
        elif name == "lam":
            return "lambda"
        elif name == "tau2":
            return "tau2"
        elif "ar" in name:
            return "phi"
        elif "ma" in name:
            return "theta"
        
        return name

    col_labels = [pretty_name(n) for n in selected_names]

    # --------------------------------------------------
    # Extract values
    # --------------------------------------------------
    def pick(arr):
        return [arr[i] for i in param_indices]

    true_vals = pick(theta_true)
    mean_vals = pick(mean_est)
    sd_vals   = pick(sd_est)
    rb_vals   = pick(rel_bias)
    mse_vals  = pick(mse)
    rmse_vals = [np.sqrt(x) for x in mse_vals]

    init_vals = pick(params_to_array(init_params))

    # --------------------------------------------------
    # Print table
    # --------------------------------------------------

    with open(output_file, "w") as f:

        # --------------------------------------------------
        # Header
        # --------------------------------------------------
        f.write("--------------------------------------------------\n")
        f.write("Monte Carlo Results\n")
        f.write("--------------------------------------------------\n\n")

        f.write("Settings:\n")
        f.write(
            f"T = {T} | N = {coords.shape[0]} | m = {m} | "
            f"iter = {n_iter} | valid = {iter_eff}\n"
        )
        f.write(f"Free Params: {free_params}\n\n")

        # --------------------------------------------------
        # Column width (auto sizing)
        # --------------------------------------------------
        col_width = 12

        # Header
        header = [""] + col_labels
        header_line = "".join(f"{h:<{col_width}}" for h in header)
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")

        # --------------------------------------------------
        # Safe formatter (handles NaN, etc.)
        # --------------------------------------------------
        def fmt(x):
            if np.isnan(x):
                return "nan"
            return f"{x:.4f}"

        # --------------------------------------------------
        # Rows
        # --------------------------------------------------
        def write_row(label, values):
            row = f"{label:<{col_width}}" + "".join(
                f"{fmt(v):<{col_width}}" for v in values
            )
            f.write(row + "\n")

        write_row("True", true_vals)
        write_row("Mean", mean_vals)
        write_row("SD", sd_vals)
        write_row("RelBias", rb_vals)
        write_row("RMSE", rmse_vals)
        write_row("Initial", init_vals)

        f.write("-" * len(header_line) + "\n")
    
    print(f"✅ Monte Carlo complete. Results saved to {output_file}")
    
    return {
        "mean": mean_est,
        "sd": sd_est,
        "rel_bias": rel_bias,
        "mse": mse,
        "estimates": estimates
    }
