from scipy.optimize import minimize
from Parameters import *
from Likelihood import *

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