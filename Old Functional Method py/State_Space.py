from dataclasses import dataclass
import numpy as np
from Parameters import *

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


# @dataclass
# class StateSpace:
#     F: np.ndarray
#     Q: np.ndarray
#     G: np.ndarray
#     R: np.ndarray
#     # C_HV: np.ndarray

#     @staticmethod
#     def build_state_space(params, coords, m):
#         N = coords.shape[0]
        
#         psi = psi_artfima(m, params.d, params.lam, params.ar, params.ma)
        
#         # --- Temporal F ---
#         F = np.zeros((m+1, m+1))
#         F[0, :] = psi[:m+1]
#         F[1:, :-1] = np.eye(m)
        
#         # --- Lift ---
#         F_full = np.kron(F, np.eye(N))
        
#         # --- Spatial covariance ---
#         Sigma_S = build_spatial_cov(
#             coords,
#             sigma2=1.0,                 
#             rho=params.rho,
#             model=params.spatial_model
#         )

#         # --- Process noise ---
#         Q_base = np.zeros((m+1, m+1))
#         Q_base[0, 0] = params.sigma2_eta  

#         Q = np.kron(Q_base, Sigma_S)
        
#         # --- Observation ---
#         G = np.zeros((N, N*(m+1)))
#         G[:, :N] = np.eye(N)
        
#         R = params.tau2 * np.eye(N)
        
#         return StateSpace(F_full, Q, G, R)
    
#     from dataclasses import dataclass
# import numpy as np


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
        psi = psi_artfima(m, params.d, params.lam, params.ar, params.ma)
        
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