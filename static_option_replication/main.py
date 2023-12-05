"""
Chapter 26.17 of "Options, Futures, and Other Derivatives", e11 by John Hull.

The following code implements construction of replication portfolio of a
barrier option as described in chapter 26.17.
"""

import numpy as np

from scipy.stats import norm


def calc_bs_call(S_0: float,
                 K: float,
                 T: float,
                 sigma: float,
                 r: float,
                 q: float) -> float:
    """
    Value European equity call option using Black-Scholes formula.

    Michal Mackanic
    05/12/2023

    Parameters
    ----------
    S_0 : float
        Spot equity price.
    K : float
        Strike price.
    T : float
        Option maturity in years.
    sigma : float
        Annualized equity volatility.
    r : float
        Annualized continuous risk-free rate.
    q : float
        Annualized continuous dividend yield.

    Returns
    -------
    float
        Option value.

    Example
    -------
    npv =\
        calc_bs_call(S_0=50.0,
                     K=50.0,
                     T=1.0,
                     sigma=0.30,
                     r=0.05,
                     q=0.02)
    """
    if (T == 0):
        return np.max([S_0 - K, 0])
    else:

        d_1 = (np.log(S_0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d_2 = d_1 - sigma * np.sqrt(T)

        npv =\
            S_0 * np.exp(-q * T) * norm.cdf(d_1) -\
            K * np.exp(-r * T) * norm.cdf(d_2)

        return npv


def main() -> None:
    """
    Implement example valuation of replication portfolio of chapter 26.17.

    Michal Mackanic
    05/12/2023

    Parameters
    ----------

    Returns
    -------
    float
        Value of the replication portfolio.

    """
    # specify input parameters
    S_0 = 50.0
    S_boundary = 60.0
    sigma = 0.30
    r = 0.10
    q = 0.00
    opt_maturity = 9 / 12
    N = 100  # number of steps applied to horizontal boundary

    # define times along horizontal boundary
    Ts = np.linspace(opt_maturity, 0, N + 1)

    # define options that will be used to construct replication portfolio
    ptf_opts = [{"S_0": 50.0,
                 "K": 50.0,
                 "T": opt_maturity}]

    for T in Ts:
        ptf_opts.append({"S_0": 50.0, "K": 60.0, "T": T})
    ptf_opts = ptf_opts[0: -1]

    # vector holding positions in individual options of the replication
    # portfolio
    ptf_pos = np.zeros(len(ptf_opts))
    ptf_pos[0] = 1.0

    # determine positions in individual options of the replication portfolio
    for T in Ts[1:]:

        # calculate NPV of the individual options for individual time points
        # along horizontal boundary
        ptf_npvs = []
        for opt in ptf_opts:
            opt_residual_maturity = opt["T"] - T
            if (opt_residual_maturity > 0.0):
                npv =\
                    calc_bs_call(S_0=S_boundary,
                                 K=opt["K"],
                                 T=opt_residual_maturity,
                                 sigma=sigma,
                                 r=r,
                                 q=q)
            else:
                npv = 0.0

            ptf_npvs.append(npv)

        # calculate positions in individual options of the replication
        # portfolio
        idx = [i for i, x in enumerate(ptf_npvs) if x == 0]
        if (idx == []):
            idx = len(ptf_npvs) - 1
        else:
            idx = idx[0] - 1
        ptf_pos[idx] =\
            -np.sum(ptf_npvs[0: idx] * ptf_pos[0: idx]) / ptf_npvs[idx]

    # calculate value of the replication portfolio at time 0
    ptf_npv = 0.0
    for opt_idx in range(len(ptf_npvs)):
        npv =\
            calc_bs_call(S_0=S_0,
                         K=ptf_opts[opt_idx]["K"],
                         T=ptf_opts[opt_idx]["T"],
                         sigma=sigma,
                         r=r,
                         q=q)
        ptf_npv += npv * ptf_pos[opt_idx]

    return ptf_npv


npv = main()
