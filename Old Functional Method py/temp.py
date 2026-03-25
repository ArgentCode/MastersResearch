import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess, arma2ma

# D and lambda components
def psi_frac(m, d, lam):
    psi = np.zeros(m + 1)
    psi[0] = 1.0

    for k in range(1, m + 1):
        psi[k] = psi[k - 1] * ((k - 1 + d) / k) * np.exp(-lam)

    return(psi)


# ARMA component
def psi_arma(m, ar=None, ma=None):
    """
    MA(∞) coefficients for ARMA(1,1) exactly as in Ferreira
    """
    if ar is None or len(ar) == 0:
        phi = 0.0
    else:
        phi = ar[0]

    if ma is None or len(ma) == 0:
        theta = 0.0
    else:
        theta = ma[0]

    psi = np.zeros(m + 1)
    psi[0] = 1.0

    for j in range(1, m + 1):
        psi[j] = (phi - theta) * (phi ** (j - 1))

    return psi


def psi_artfima(m, d, lam, ar=None, ma=None):
    psi_f = psi_frac(m, d, lam)
    psi_a = psi_arma(m, ar, ma)

    psi = np.convolve(psi_a, psi_f)[:m + 1]

    return(psi)
  
def psi_artfima_recursive(m, d, lam, ar=None, ma=None):
    """
    Recursive ψ_j construction for ARFIMA(p,d,q)
    Matches Ferreira Eq. (page 5)
    """
    if ar is None:
        ar = []
    if ma is None:
        ma = []

    p = len(ar)
    q = len(ma)

    # --- fractional weights π_j(d) ---
    pi = np.zeros(m + 1)
    pi[0] = 1.0
    for j in range(1, m + 1):
        pi[j] = pi[j-1] * ((j - 1 + d) / j) * np.exp(-lam)

    # --- ψ weights ---
    psi = np.zeros(m + 1)
    psi[0] = 1.0

    for j in range(1, m + 1):
        val = pi[j]

        # AR part
        for i in range(1, min(j, p) + 1):
            val += ar[i-1] * psi[j - i]

        # MA part (Ferreira sign convention!)
        for i in range(1, min(j, q) + 1):
            val += ma[i-1] * pi[j - i]

        psi[j] = val

    return psi

def main():
    m = 10
    d = 0
    lam = 0
    ar = [0.5]
    ma = [0.2]

    psi_conv = psi_artfima(m, d, lam, ar, ma)
    psi_rec = psi_artfima_recursive(m, d, lam, ar, ma)
    psi_ma = [1, ] + list(arma2ma(ar=ar, ma=ma, lags=m))

    print("ψ (convolution):", psi_conv)
    print("ψ (recursive):   ", psi_rec)
    print("ψ (MA):           ", psi_ma)

if __name__ == "__main__":
    main()