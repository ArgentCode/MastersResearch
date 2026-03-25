from Simulation import *
from Estimation import *
import numpy as np


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
        print(f"Iteration {i+1}/{n_iter}")
        
        Y = simulate_artfima(true_params, coords, T)
        
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
    def is_free(name):
        # Handle AR/MA parameters (ar1, ar2, ... or ma1, ma2, ...)
        if name.startswith("ar"):
            return "ar" in free_params
        elif name.startswith("ma"):
            return "ma" in free_params
        else:
            return name in free_params
    
    param_indices = [i for i, name in enumerate(names) if is_free(name)]
    selected_names = [names[i] for i in param_indices]

    # Pretty labels (optional but nice)
    def pretty_name(name):
        if "ar" in name:
            return "phi"
        elif "ma" in name:
            return "theta"
        elif name == "sigma2_eta":
            return "sigma2"
        elif name == "rho":
            return "rho"
        elif name == "d":
            return "d"
        elif name == "lam":
            return "lambda"
        elif name == "tau2":
            return "tau2"
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
    
        f.write("--------------------------------------------------\n")
        f.write("Monte Carlo Results\n")
        f.write("--------------------------------------------------\n\n")
        f.write("Settings:\n")
        f.write(
            f"T = {T} | N = {coords.shape[0]} | m = {m} | "
            f"iter = {n_iter} | valid = {iter_eff}\n"
        )
        f.write(f"Free Params: {free_params}\n\n")

        # Header
        header = [""] + col_labels
        f.write("".join(f"{h:<12}" for h in header))
        f.write("\n")

        # Rows
        def write_row(label, values):
            f.write(f"{label:<12}" + "".join(f"{v:<12.4f}" for v in values))
            f.write("\n")

        write_row("True", true_vals)
        write_row("Mean", mean_vals)
        write_row("SD", sd_vals)
        write_row("RelBias", rb_vals)
        write_row("RMSE", rmse_vals)
        write_row("Initial", init_vals)
    
    print(f"✅ Monte Carlo complete. Results saved to {output_file}")
    
    return {
        "mean": mean_est,
        "sd": sd_est,
        "rel_bias": rel_bias,
        "mse": mse,
        "estimates": estimates
    }
