
import numpy as np
import pandas as pd

from scipy.stats import norm


def bs_call(S, K, r, q, T, sigma):
    """
    Michal Mackanic
    19/11/2023 v1.0

    Description
    -----------
    Calculate value of a simple European equity option using Black-Scholes
    model.

    Parameters
    ----------
    S : float
        spot price of the underlying equity
    K : float
        strike price
    r : float
        risk-free interest rate (annual in continuous compounding)
    q : float
        dividend yield (annual in continuous compounding)
    T : float
        time to option maturity in years
    sigma : float
        annual equity volatility

    Returns:
        opt : float
            option value

    Example
    -------
    S = 100
    K = 90
    r = 0.05
    q = 0.07
    T = 5.00
    sigma = 0.30
    opt = bs_call(S=S, K=K, r=r, q=q, T=T, sigma=sigma)
    """

    d1 = (np.log(S/K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    opt = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return opt


def main(paths_no, quantile, seed=None):
    """
    Michal Mackanic
    19/11/2023 v1.0

    Description
    -----------
    This function illustrates Cornish-Fisher expansion on an example of
    European call option. For details see chapter 11 of "Options, Futures,
    and other Derivatives, e11" by John Hull.

    source: http://www-2.rotman.utoronto.ca/~hull/TechnicalNotes/TechnicalNote10.pdf

    Parameters
    ----------
        paths_no : int
            number of simulations used to calculated moments entering
            Cornish-Fisher expansion
        q : float
            equity price quantile for which option price change is to be
            estimated
        seed : int
            seed of random number generator

    Returns
    -------
        opt_bs : float
            current option value using Black-Scholes formula
        opt_mc : float
            current option value using Monte-Carlo simulation of equity prices
        opt_qntl : float
            option value for quantile q using simulated equity prices
        opt_norm : float
            option value for quantile estimated based on assumption of normal
            distribution
        opt_cf : float
            option value for quantile q estimated based on Cornish-Fisher
            expansion

    Example
    -------
    # compare opt_qntl vs. opt_norm and opt_qntl vs. opt_cf
    [opt_bs, opt_mc, opt_qntl, opt_norm, opt_cf] = main(paths_no=1000000, quantile=0.99, seed=None)
    """

    # option parameters
    print("Setting up option parameters...")
    S_0 = 100
    K = 90
    r = 0.05
    q = 0.07
    T = 1 / 252  # the approximation works only in very short time horizons; the longer time horizons it fails spectacularly
    sigma = 0.20

    # fix random seed
    print("Fixing random seed generator...")
    if (seed is not None):
        np.random.seed(seed=seed)

    # generate random variables
    print("Generate random variables...")
    dt = T
    steps_no = int(T / dt)
    u = norm.rvs(size=[paths_no, steps_no])

    # simulate stock prices
    print("Simulate equity prices")
    dS_aux1 = np.array([[0] * paths_no], dtype=np.float64).T
    dS_aux2 = (r - q - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * u
    dS = np.append(dS_aux1, dS_aux2, axis=1)
    dS = np.cumsum(dS, axis=1)
    S_T = S_0 * np.exp(dS)[:, 1]

    # calculate the current option values
    print("Calculating the current option value using Black-Scholes formula...")
    opt_bs = bs_call(S_0, K, r, q, T, sigma)

    # calculate option values based on simulated equity prices
    print("Calculating the current option value using Monte-Carlo...")
    opt_payoffs = S_T - K
    opt_payoffs = np.array([max(opt_payoff, 0.0) for opt_payoff in opt_payoffs], float)
    opt_mcs = np.exp(-r * T) * opt_payoffs

    # calculated option value using simulated equity prices
    opt_mc = np.mean(opt_mcs)

    # calculate option value for the assumed quantile using simulated equity
    # prices
    print("Calculating quantile option price using Monte-Carlo...")
    opt_qntl = np.quantile(opt_mcs, quantile)

    # calculate option value for the assumed quantile using assumption of
    # normal distribution
    print("Calculating quantile option price using normal approximation...")
    stdev = np.std(opt_mcs - opt_bs)
    opt_norm = opt_bs + stdev * norm.ppf(quantile)

    # calculate option value for the assumed quantile using Cornish-Fisher
    # expansion
    print("Calculating quantile option price using Cornish-Fisher expansion...")
    d_opt = opt_mcs - opt_bs
    moment_1 = np.mean(d_opt)
    moment_2 = np.std(d_opt) ** 2
    moment_3 = 1 / (moment_2 ** (3 / 2)) * np.mean((d_opt - moment_1) ** 3)

    z_q = norm.ppf(quantile)
    w_q = z_q + 1 / 6 * (z_q ** 2 - 1) * moment_3

    opt_cf = opt_bs + moment_1 + w_q * moment_2 ** (1 / 2)

    # return results
    return [opt_bs, opt_mc, opt_qntl, opt_norm, opt_cf]


