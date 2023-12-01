"""
Valuation of synthetic CDO.

See chapter 25.10, example 25.2 of book "Options, Futures and Other
Derivatives", e11 by John Hull.
"""

import math
import numpy as np
from scipy.stats import norm

import my_gaussian_quadrature


def calc_pit_pd(hr: float, t: float, rho: float, F: float) -> float:
    """
    Calculate PIT PD using Vasicek formula.

    Parameters
    ----------
    hr : float
        Hazard rate.
    t : float
        Time to default in years.
    rho : float
        Correlation rho in Vasicek model.
    F : float
        Systematic risk factor in Vasicek model.

    Returns
    -------
    float
        Stressed PIT PD.

    """
    ttc_pd = (1.0 - np.exp(-hr * t))
    return norm.cdf((norm.ppf(ttc_pd) - np.sqrt(rho) * F) / np.sqrt(1.0 - rho))


def calc_ptf_pd(n: int, k: int, pit_pd: float) -> float:
    """
    Calculate portfolio default rate for a given PIT PD and number of defaults.

    Parameters
    ----------
    n : int
        Number of counterparties within underlying portfolio.
    k : int
        Number of defaults within portfolio.
    pit_pd : float
        PIT PD for a single counterparty.

    Returns
    -------
    float
        Portfolio PIT PD for a given number of defaults.

    """
    ptf_pd = math.factorial(n) / (math.factorial(n - k) * math.factorial(k))
    ptf_pd *= pit_pd ** k * (1 - pit_pd) ** (n - k)

    return ptf_pd


def calc_E(n: float,
           alpha_L: float,
           alpha_H: float,
           rho: float,
           R: float,
           F: float,
           hr: float,
           dt: float,
           j: float) -> float:
    """
    Calculate expected tranche principal.

    Parameters
    ----------
    n : float
        Number of counterparties within underlying portfolio.
    alpha_L : float
        Tranche attachment point.
    alpha_H : float
        Tranche dettachment point.
    rho : float
        Correlation rho in Vasicek model.
    R : float
        Recovery rate.
    F : float
        Systematic risk factor in Vasicek model.
    hr : float
        Hazard rate.
    dt : float
        Time step for "insurance" payments based on CDS spread.
    j : float
        "Insurance" payment number, e.g. j = 5 means 5th payment.

    Returns
    -------
    float
        Expected tranche principal for j-th "insurance" payment.

    """
    # %% attachment and detachment points expressed in number of defaults
    n_L = alpha_L * n / (1 - R)
    n_H = alpha_H * n / (1 - R)

    # %% calculate time of j-th "insurance" payment
    t = j * dt

    # %% calculate PIT PD
    pit_pd = calc_pit_pd(hr=hr, t=t, rho=rho, F=F)

    # %% calculate expected tranch principal
    E = 0.0
    for k in range(int(np.ceil(n_L))):
        E += calc_ptf_pd(n=n, k=k, pit_pd=pit_pd)

    for k in range(int(np.ceil(n_L)), int(np.ceil(n_H))):
        E += calc_ptf_pd(n=n, k=k, pit_pd=pit_pd) *\
            (alpha_H - k * (1 - R) / n) / (alpha_H - alpha_L)

    return E


def calc_A(m: float,
           T: float,
           n: float,
           alpha_L: float,
           alpha_H: float,
           rho: float,
           R: float,
           hr: float,
           F: float,
           r: float) -> float:
    """
    Calculate PV expected payment.

    Parameters
    ----------
    m : float
        Number of "insurance" payments during CDO lifetime of T years.
    T : float
        CDO lifetime in years.
    n : float
        Number of counterparties within underlying portfolio.
    alpha_L : float
        Tranche attachment point.
    alpha_H : float
        Tranche dettachment point.
    rho : float
        Correlation rho in Vasicek model.
    R : float
        Recovery rate.
    hr : float
        Hazard rate.
    F : float
        Systematic risk factor in Vasicek model.
    dt : float
        Time step for "insurance" payments based on CDS spread.
    j : float
        "Insurance" payment number, e.g. j = 5 means 5th payment.
    r : float
        Risk-free rate.

    Returns
    -------
    float
        PV expected payment.

    """
    dt = T / m

    A = 0.0
    for j in range(1, m + 1):

        t = j * dt
        df = np.exp(-r * t)
        E = calc_E(n=n,
                   alpha_L=alpha_L,
                   alpha_H=alpha_H,
                   rho=rho,
                   R=R,
                   F=F,
                   hr=hr,
                   dt=dt,
                   j=j)
        A += dt * E * df

    return A


def calc_B(m: float,
           T: float,
           n: float,
           alpha_L: float,
           alpha_H: float,
           rho: float,
           R: float,
           hr: float,
           F: float,
           r: float) -> float:
    """
    Calculate PV accrual payment.

    Parameters
    ----------
    m : float
        Number of "insurance" payments during CDO lifetime of T years.
    T : float
        CDO lifetime in years.
    n : float
        Number of counterparties within underlying portfolio.
    alpha_L : float
        Tranche attachment point.
    alpha_H : float
        Tranche dettachment point.
    rho : float
        Correlation rho in Vasicek model.
    R : float
        Recovery rate.
    hr : float
        Hazard rate.
    F : float
        Systematic risk factor in Vasicek model.
    r : float
        Risk-free rate.

    Returns
    -------
    float
        PV accrual payment.

    """
    dt = T / m

    E_prev = calc_E(n=n,
                    alpha_L=alpha_L,
                    alpha_H=alpha_H,
                    rho=rho,
                    R=R,
                    F=F,
                    hr=hr,
                    dt=dt,
                    j=0)

    B = 0.0
    for j in range(1, m + 1):

        t = j * dt
        df = np.exp(-r * (t - dt / 2))
        E_curt = calc_E(n=n,
                        alpha_L=alpha_L,
                        alpha_H=alpha_H,
                        rho=rho,
                        R=R,
                        F=F,
                        hr=hr,
                        dt=dt,
                        j=j)
        B += 0.5 * dt * (E_prev - E_curt) * df
        E_prev = E_curt

    return B


def calc_C(m: float,
           T: float,
           n: float,
           alpha_L: float,
           alpha_H: float,
           rho: float,
           R: float,
           hr: float,
           F: float,
           r: float) -> float:
    """
    Calculate PV expected payoff.

    Parameters
    ----------
    m : float
        Number of "insurance" payments during CDO lifetime of T years.
    T : float
        CDO lifetime in years.
    n : float
        Number of counterparties within underlying portfolio.
    alpha_L : float
        Tranche attachment point.
    alpha_H : float
        Tranche dettachment point.
    rho : float
        Correlation rho in Vasicek model.
    R : float
        Recovery rate.
    hr : float
        Hazard rate.
    F : float
        Systematic risk factor in Vasicek model.
    r : float
        Risk-free rate.

    Returns
    -------
    float
        PV expected payoff.

    """
    dt = T / m

    E_prev = calc_E(n=n,
                    alpha_L=alpha_L,
                    alpha_H=alpha_H,
                    rho=rho,
                    R=R,
                    F=F,
                    hr=hr,
                    dt=dt,
                    j=0)

    C = 0.0
    for j in range(1, m + 1):

        t = j * dt
        df = np.exp(-r * (t - dt / 2))
        E_curt = calc_E(n=n,
                        alpha_L=alpha_L,
                        alpha_H=alpha_H,
                        rho=rho,
                        R=R,
                        F=F,
                        hr=hr,
                        dt=dt,
                        j=j)
        C += (E_prev - E_curt) * df
        E_prev = E_curt

    return C


def calc_sprd(m: float,
              T: float,
              n: float,
              alpha_L: float,
              alpha_H: float,
              rho: float,
              R: float,
              hr: float,
              r: float) -> float:
    """
    Calculate CDO spread.

    Parameters
    ----------
    m : float
        Number of "insurance" payments during CDO lifetime of T years.
    T : float
        CDO lifetime in years.
    n : float
        Number of counterparties within underlying portfolio.
    alpha_L : float
        Tranche attachment point.
    alpha_H : float
        Tranche dettachment point.
    rho : float
        Correlation rho in Vasicek model.
    R : float
        Recovery rate.
    hr : float
        Hazard rate.
    r : float
        Risk-free rate.

    Returns
    -------
    float
        CDO spread.

    """
    # %% create auxiliary single variable functions f(x)
    A_func = (lambda x: calc_A(m=m,
                               T=T,
                               n=n,
                               alpha_L=alpha_L,
                               alpha_H=alpha_H,
                               rho=rho,
                               R=R,
                               hr=hr,
                               F=x,
                               r=r))

    B_func = (lambda x: calc_B(m=m,
                               T=T,
                               n=n,
                               alpha_L=alpha_L,
                               alpha_H=alpha_H,
                               rho=rho,
                               R=R,
                               hr=hr,
                               F=x,
                               r=r))

    C_func = (lambda x: calc_C(m=m,
                               T=T,
                               n=n,
                               alpha_L=alpha_L,
                               alpha_H=alpha_H,
                               rho=rho,
                               R=R,
                               hr=hr,
                               F=x,
                               r=r))

    # %% apply Gaussian quadrature on f(x) functions
    A = my_gaussian_quadrature.calc(A_func)
    B = my_gaussian_quadrature.calc(B_func)
    C = my_gaussian_quadrature.calc(C_func)

    # %% calculate CDO spread
    s = C / (A + B)

    # %% return results
    return s, A, B, C


# %% run
s, A, B, C =\
    calc_sprd(m=5*4,
              T=5,
              n=125,
              alpha_L=0.03,
              alpha_H=0.06,
              rho=0.15,
              R=0.40,
              hr=0.0083,
              r=0.035)
