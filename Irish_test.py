import numpy as np
import pandas as pd
from McFullCode import *

import re

def dms_to_decimal(coord):
    # Extract direction (N, S, E, W)
    direction = coord[-1]
    
    # Extract numbers (degrees, minutes, optional seconds)
    nums = list(map(float, re.findall(r"\d+\.?\d*", coord)))
    
    if len(nums) == 2:
        deg, minutes = nums
        seconds = 0
    elif len(nums) == 3:
        deg, minutes, seconds = nums
    else:
        raise ValueError(f"Unexpected format: {coord}")
    
    decimal = deg + minutes / 60 + seconds / 3600
    
    # Apply sign
    if direction in ['S', 'W']:
        decimal *= -1
        
    return decimal

# Convert lat/lon to approximate km before passing to build_spatial_cov
def latlon_to_km(coords):
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])
    R_earth = 6371.0
    x = R_earth * lon_rad * np.cos(lat_rad.mean())
    y = R_earth * lat_rad
    return np.column_stack([x, y])

def harmonic_design_matrix(doy, K=3):
    X = [np.ones_like(doy)]
    
    for k in range(1, K+1):
        X.append(np.cos(2 * np.pi * k * doy / 365))
        X.append(np.sin(2 * np.pi * k * doy / 365))
    
    return np.column_stack(X)

def compute_CR_metrics(Y_test, forecasts, forecast_vars, station_codes):
    """
    Compute CR1, CR2, CR3 for each station.
    
    Parameters
    ----------
    Y_test         : (K, N) — observed test data
    forecasts      : (K, N) — predicted means
    forecast_vars  : (K, N) — predictive variances (Delta diagonal)
    station_codes  : list of N station name strings
    
    Returns
    -------
    dict with keys 'CR1', 'CR2', 'CR3', each (N,) array
    """
    K, N = Y_test.shape

    errors  = Y_test - forecasts          # (K, N)
    errors2 = errors ** 2                 # (K, N)

    sum_errors  = np.sum(errors,  axis=0)   # (N,)
    sum_errors2 = np.sum(errors2, axis=0)   # (N,)
    sum_vars    = np.sum(forecast_vars, axis=0)   # (N,)

    CR1 = sum_errors  / np.sqrt(sum_vars)
    CR2 = np.sqrt(sum_errors2 / sum_vars)
    CR3 = np.sqrt(sum_errors2 / K)

    # --- print table ---
    col_w = 10
    header = f"{'Station':<{col_w}}{'CR1':>{col_w}}{'CR2':>{col_w}}{'CR3':>{col_w}}"
    print("\nOut-of-sample forecast metrics:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, code in enumerate(station_codes):
        print(f"{code:<{col_w}}{CR1[i]:>{col_w}.4f}{CR2[i]:>{col_w}.4f}{CR3[i]:>{col_w}.4f}")
    print("-" * len(header))
    print(f"{'Mean':<{col_w}}{CR1.mean():>{col_w}.4f}{CR2.mean():>{col_w}.4f}{CR3.mean():>{col_w}.4f}")

    return {"CR1": CR1, "CR2": CR2, "CR3": CR3}


def main():
    np.random.seed(5473)

    # Methods to test against:
    '''
    AR(1) phi = 0.5, sigma2 = 0.5, rho = 0.005
    ARMA(1,1) phi = 0.5, theta = 0.25, sigma2 = 0.5, rho = 0.005
    FN(d) with d = 0.15, sigma2 = 0.5, rho = 0.005
    ARFIMA(1,d,0) with d=0.25, phi = 0.5, sigma2 = 0.5, rho = 0.005
    ARFIMA(2,d,0) with d=0.25, phi1 = 0.5, phi2 = -0.5, sigma2 = 0.5, rho = 0.005
    ARTFIMA(1,d,lambda,0) with d=0.25, lambda = 0.001, phi = 0.5, sigma2 = 0.5, rho = 0.005
    '''

    # --------------------------------------------------
    # Model Set up
    # --------------------------------------------------
    m = 20
    init_params = Parameters(
        d=0.0,                 # FIXED
        lam=0.0,               # FIXED
        ar=[0.75],
        ma=[0],
        sigma2_eta=0.7,
        rho=150,
        spatial_model=1,
        tau2=0               # FIXED
    )
    free_params = ["ar", "sigma2_eta", "rho"]
    
    # --------------------------------------------------
    # Temporal data
    # --------------------------------------------------
    df = pd.read_csv("irish_wind_t.csv", skipinitialspace=True)
    df = df.head(100)
    # Rows = time points, columns = stations
    # First row is header with station codes e.g. "RPT","VAL",...
    station_codes = list(df.columns)  # ['RPT', 'VAL', 'ROS', ...]

    df = df.apply(pd.to_numeric, errors='coerce')
    T, N = df.shape
    print(f"T={T} time points, N={N} stations")

    sqrt_data = np.sqrt(df.values)  # shape (T, N)
    nan_idx = np.where(np.isnan(sqrt_data))
    print(f"Found {len(nan_idx[0])} NaN values in sqrt data.")

    # --------------------------------------------------
    # Seasonal adjustment
    # --------------------------------------------------
    daily_mean = np.nanmean(sqrt_data, axis=0)  # mean over time, shape (N,) -- just for reference
    dates = pd.date_range(start="1961-01-01", periods=T, freq="D")
    day_of_year = dates.dayofyear.values

    # Fit harmonics to each station separately, then remove
    X_full = harmonic_design_matrix(day_of_year, K=3)  # (T, 7)

    deseasonalized = np.empty_like(sqrt_data)
    for n in range(N):
        y_n = sqrt_data[:, n]
        valid = ~np.isnan(y_n)
        beta_n = np.linalg.lstsq(X_full[valid], y_n[valid], rcond=None)[0]
        seasonal_n = X_full @ beta_n
        deseasonalized[:, n] = y_n - seasonal_n

    print("Deseasonalized shape:", deseasonalized.shape)  # should be (T, N)

    # --------------------------------------------------
    # Fill NaNs (neighbor interpolation, then zero fallback)
    # --------------------------------------------------
    nan_idx = np.where(np.isnan(deseasonalized))
    print(f"Found {len(nan_idx[0])} NaN values in deseasonalized data.")
    for i, j in zip(*nan_idx):
        left  = deseasonalized[i-1, j] if i - 1 >= 0 else np.nan      # previous time step
        right = deseasonalized[i+1, j] if i + 1 < T  else np.nan      # next time step
        fill  = np.nanmean([left, right])
        deseasonalized[i, j] = fill if np.isfinite(fill) else 0.0

    pd.DataFrame(deseasonalized, columns=station_codes).to_csv("irish_wind_deseasonalized.csv", index=False)

    # --------------------------------------------------
    # Spatial locations — aligned to station_codes order
    # --------------------------------------------------
    coords_df = pd.read_csv("irish_wind_locs.csv", quotechar='"')
    coords_df["lat"] = coords_df["Latitude"].apply(dms_to_decimal)
    coords_df["lon"] = coords_df["Longitude"].apply(dms_to_decimal)

    # Build a lookup: code -> (lat, lon)
    loc_lookup = dict(zip(coords_df["Code"], zip(coords_df["lat"], coords_df["lon"])))

    # Align to station order from wind data
    missing = [c for c in station_codes if c not in loc_lookup]
    if missing:
        print(f"WARNING: no location found for stations: {missing}")

    coords = np.array([loc_lookup[c] for c in station_codes])  # (N, 2)
    coords_km = latlon_to_km(coords)

    nan_idx = np.where(np.isnan(coords_km))
    print(f"Found {len(nan_idx[0])} NaN values in coords data.")
    print("coords_km shape:", coords_km.shape)  # should be (N, 2)
    # print(coords_km)


    # -------------------------------------------------
    # Testing
    # -------------------------------------------------
    testing = False
    if testing:
        # Test the objective at your initial params
        params = init_params  # whatever you have set

        print("=== PSI weights ===")
        psi = psi_artfima(m, params.d, params.lam, params.ar, params.ma)
        print("psi:", psi)
        print("any nan/inf in psi:", np.any(~np.isfinite(psi)))

        print("\n=== Spatial covariance ===")
        Sigma_S = build_spatial_cov(coords_km, sigma2=params.sigma2_eta, rho=params.rho, model=params.spatial_model)
        print("Sigma_S min/max:", Sigma_S.min(), Sigma_S.max())
        print("any nan/inf:", np.any(~np.isfinite(Sigma_S)))
        print("condition number:", np.linalg.cond(Sigma_S))

        print("\n=== State space ===")
        ss = StateSpace.build_state_space(params, coords_km, m)
        print("F shape:", ss.F.shape)
        print("G shape:", ss.G.shape)
        print("Q shape:", ss.Q.shape)
        print("any nan/inf in Q:", np.any(~np.isfinite(ss.Q)))

        print("\n=== One-step Kalman (t=0) ===")
        N = coords_km.shape[0]
        Sigma_S_full = build_spatial_cov(coords_km, sigma2=params.sigma2_eta, rho=params.rho, model=params.spatial_model)
        Omega = np.kron(np.eye(m+1), Sigma_S_full)
        X = np.zeros(ss.F.shape[0])
        Delta = ss.G @ Omega @ ss.G.T + ss.R + 1e-6 * np.eye(N)
        print("Delta min/max:", Delta.min(), Delta.max())
        print("Delta condition number:", np.linalg.cond(Delta))
        sign, logdet = np.linalg.slogdet(Delta)
        print("slogdet sign:", sign, "logdet:", logdet)

        print("\n=== Full loglik at initial params ===")
        ll = kalman_loglik(params, deseasonalized, coords_km, m)
        print("loglik:", ll)
    
    # --------------------------------------------------
    # Train/test split
    # --------------------------------------------------
    Z = 7
    Y_train = deseasonalized[:-Z]   # (T-7, N)
    Y_test  = deseasonalized[-Z:]   # (7, N)

    # --- fit on training data ---
    optimized_params, result = estimate_params(
        Y=Y_train,
        m=m,
        coords=coords_km,
        base_params=init_params,
        free_params=free_params
    )

    if result.success:
        print("\nOptimization succeeded.")


    # --- forecast ---
    forecasts, forecast_vars = kalman_forecast(
        params=optimized_params,
        Y_train=Y_train,
        coords=coords_km,
        m=m,
        Z=Z
    )

    # --- evaluate ---
    rmse = np.sqrt(np.mean((forecasts - Y_test) ** 2, axis=0))
    print("\nPer-station RMSE (deseasonalized sqrt scale):")
    for code, r in zip(station_codes, rmse):
        print(f"  {code}: {r:.4f}")
    print(f"Overall RMSE: {np.sqrt(np.mean((forecasts - Y_test)**2)):.4f}")

    metrics = compute_CR_metrics(Y_test, forecasts, forecast_vars, station_codes)

if __name__ == "__main__":
    main()