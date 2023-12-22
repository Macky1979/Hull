"""'Fitting the variance-gamma model to financial data' by Eugene Seneta."""

import math
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from scipy.special import gamma as gamma_func  # gamma function
from scipy.stats import gamma as gamma_dist  # gamma distribution
from scipy.stats import norm as norm_dist
import scipy.integrate as integrate

from scipy.stats import skew
from scipy.stats import kurtosis

import warnings
warnings.filterwarnings("error")


def bessel3(nu, omega) -> float:
    """
    Implement modified Bessel function of the third kind.

    Michal Mackanic
    22/12/2023 v1.0

    Description
    -----------
    Implement modified Bessel function of the third kind following defition
    of "Fitting the variance-gamma model to financial data" by Eugene Seneta.

    Parameters
    ----------
    nu : float
        Parameter of modified Bessel function of the third kind.
    omega : TYPE
        Value for which the function is to be evaluated.

    Returns
    -------
    float
        Value of Bessel function of the third kind.

    Example
    -------
    nu = -0.30
    omega = 2.0
    bessel3(nu=nu, omega=omega)

    """
    f = lambda x : np.exp(-omega * math.cosh(x)) * math.cosh(nu * x)
    return integrate.quad(f, 0, 30)[0]


def get_log_returns(vals: np.array) -> np.array:
    """
    Calculate log-returns from a price series.

    Michal Mackanic
    22/12/2023 v1.0

    Parameters
    ----------
    vals : np.array
        Price series.

    Returns
    -------
    xs : np.array
        log-returns.

    Example
    -------
    vals = np.array([1.00, 1.01, 1.00, 0.99, 1.02, 1.04, 1.05], float)
    returns = get_log_returns(vals=vals)

    """
    # calculate log-returns
    xs = np.log(vals[0:-1] / vals[1:])

    # return log-returns
    return xs


def check_param_rngs(theta: float, sigma: float, vega: float) -> bool:
    """
    Check ranges of variance-gamma parameters.

    Michal Mackanic
    22/12/2023 v1.0

    Description
    -----------
    Check ranges of variance-gamma parameters. The name of parameters and
    their meaning reflects paper "Fitting the variance-gamma model to
    financial data" by Eugene Seneta.

    Parameters
    ----------
    theta : float
        Theta parameter.
    sigma : float
        Sigma parameter.
    vega : float
        Vega parameter.

    Returns
    -------
    bool
        True - Parameters are within reasonable ranges.
        False - Parameters are not within reasonable ranges.

    Example
    -------
    check_param_rngs(theta= 0.0, sigma = 0.25, vega = 1.25)

    """
    # check parameter ranges (optimized for S&P 500)
    if ((theta <= -2.0) or (theta > 2.0)):
        return False
    elif ((sigma <= 1e-5) or (sigma > 2.0)):
        return False
    elif (vega <= 1e-2) or (vega > 10):
        return False
    else:
        return True


def get_moments(vals: np.array) -> list[float]:
    """
    Calculate the first four moments.

    Michal Mackanic
    22/12/2023 v1.0

    Parameters
    ----------
    vals : np.array
        Values of a random variable for which the four moments are to be
        calculated.

    Returns
    -------
    list[float]
        Values of the four moments.

    Example
    -------
    log_returns =\
        np.array([ 0.003916850,  0.004087070,  0.00793661, -0.003913780,
                  -0.000569117, -0.005423170,  0.00585654,  0.003776990,
                  -0.000946684,  0.000979647, -0.00195614,  0.000596756,
                   0.004052870, -0.002022990,  0.00736313,  0.001281280,
                   0.001189640,  0.001595810,  0.01889530, -0.000836091,
                   0.015495700, -0.008116760,  0.00100443,  0.002836120,
                   0.001751430,  0.009349870,  0.01868300,  0.010451200,
                   0.006454090,  0.011938500, -0.00481187, -0.011903100,
                  -0.014443400,  0.007239490, -0.00168696, -0.012665200,
                  -0.008519010, -0.013490400, -0.00000001,  0.010538600,
                  -0.005031480, -0.006265950,  0.00428383,  0.005194460,
                   0.006284030,  0.011745600, -0.00130487,  0.008077090,
                  -0.013839400,  0.000079287, -0.00271317,  0.005875780,
                   0.000229292, -0.014844100,  0.00401502, -0.002298250,
                  -0.016536900, -0.009439200, -0.00215342,  0.000721036,
                  -0.012234100,  0.008394550,  0.00124085, -0.005712160,
                   0.006701010,  0.001425580, -0.00321647, -0.006995990,
                  -0.004203010,  0.001797540, -0.00159822,  0.003825800,
                   0.014404100,  0.006245060,  0.00669552, -0.013549400,
                   0.010984300, -0.002781280,  0.00685562, -0.000148740,
                  -0.007742760, -0.007584130, -0.01161810,  0.005733910,
                  -0.001070200,  0.000250656, -0.00706364, -0.004227190,
                   0.008983580, -0.005314090, -0.00255123, -0.013936200,
                  -0.002668650,  0.001467640,  0.00982941, -0.006445430,
                  -0.000155460,  0.002810760,  0.00402597,  0.000324102])

    [m1, m2, m3, m4] = get_moments(log_returns)

    """
    # calculate moments
    m1 = np.mean(vals)  # mean
    m2 = np.mean([(val - m1) ** 2 for val in vals])  # variance
    m3 = np.mean([(val - m1) ** 3 for val in vals])
    m4 = np.mean([(val - m1) ** 4 for val in vals])

    # return moments
    return [m1, m2, m3, m4]


def f_m1(c: float,
         theta: float = 0.0) -> float:
    """
    Calculate the first moment.

    Michal Mackanic
    22/12/2023 v1.0

    Description
    -----------
    Calculate the first moment from variance-gamma parameters as defined in
    paper "Fitting the variance-gamma model to financial data" by Eugene
    Seneta.

    Parameters
    ----------
    c : float
        Parameter c.
    theta : float, optional
        Parameter theta.
        The default is 0.0.

    Returns
    -------
    float
        Value of the first moment.

    """
    return c + theta


def f_m2(sigma: float,
         vega: float,
         theta: float = 0.0) -> float:
    """
    Calculate the second moment.

    Michal Mackanic
    22/12/2023

    Description
    -----------
    Calculate the second moment from variance-gamma parameters as defined in
    paper "Fitting the variance-gamma model to financial data" by Eugene
    Seneta.

    Parameters
    ----------
    sigma : float
        Sigma parameter.
    vega : float
        Vega parameter.
    theta : float, optional
        Theta parameter.
        The default is 0.0.

    Returns
    -------
    float
        Value of the second moment.

    """
    return sigma ** 2 + theta ** 2 * vega


def f_m3(sigma: float,
         vega: float,
         theta: float = 0.0) -> float:
    """
    Calculate the third moment.

    Michal Mackanic
    22/12/2023

    Description
    -----------
    Calculate the third moment from variance-gamma parameters as defined in
    paper "Fitting the variance-gamma model to financial data" by Eugene
    Seneta.

    Parameters
    ----------
    sigma : float
        Sigma parameter.
    vega : float
        Vega parameter.
    theta : float, optional
        Theta parameter.
        The default is 0.0.

    Returns
    -------
    float
        Value of the third moment.

    """
    return 2 * theta ** 3 * vega ** 2 + 3 * sigma ** 2 * theta * vega

def f_m4(sigma: float,
         vega: float,
         theta: float = 0.0) -> float:
    """
    Calculate the fourth moment.

    Michal Mackanic
    22/12/2023

    Description
    -----------
    Calculate the fourth moment from variance-gamma parameters as defined in
    paper "Fitting the variance-gamma model to financial data" by Eugene
    Seneta.

    Parameters
    ----------
    sigma : float
        Sigma parameter.
    vega : float
        Vega parameter.
    theta : float, optional
        Theta parameter.
        The default is 0.0.

    Returns
    -------
    float
        Value of the fourth moment.

    """
    m4 =\
        3 * sigma ** 4 * vega + 12 * sigma ** 2 * theta ** 2 * vega ** 2 +\
        6 * theta ** 4 * vega ** 3 + 3 * sigma ** 4 +\
        6 * sigma ** 2 * theta ** 2 * vega + 3 * theta ** 4 * vega ** 2

    return m4


def calib_approx(xs: np.array) -> dict[str, float]:
    """
    Approximate calibration of variance-gamma distribution.

    Michal Mackanic
    22/12/2023 v1.0

    Description
    -----------
    Approximate calibration of variance-gamma distribution based on paper
    "Fitting the variance-gamma model to financial data" by Eugene Seneta.

    Parameters
    ----------
    xs : np.array
        Values (e.g. stock log-returns) to which the variance-gamma
        distribution is to be fitted.

    Returns
    -------
    dict[str, float]
        Dictionary containing parameter names and their estimated values.

    Example
    -------
    log_returns =\
        np.array([ 0.003916850,  0.004087070,  0.00793661, -0.003913780,
                  -0.000569117, -0.005423170,  0.00585654,  0.003776990,
                  -0.000946684,  0.000979647, -0.00195614,  0.000596756,
                   0.004052870, -0.002022990,  0.00736313,  0.001281280,
                   0.001189640,  0.001595810,  0.01889530, -0.000836091,
                   0.015495700, -0.008116760,  0.00100443,  0.002836120,
                   0.001751430,  0.009349870,  0.01868300,  0.010451200,
                   0.006454090,  0.011938500, -0.00481187, -0.011903100,
                  -0.014443400,  0.007239490, -0.00168696, -0.012665200,
                  -0.008519010, -0.013490400, -0.00000001,  0.010538600,
                  -0.005031480, -0.006265950,  0.00428383,  0.005194460,
                   0.006284030,  0.011745600, -0.00130487,  0.008077090,
                  -0.013839400,  0.000079287, -0.00271317,  0.005875780,
                   0.000229292, -0.014844100,  0.00401502, -0.002298250,
                  -0.016536900, -0.009439200, -0.00215342,  0.000721036,
                  -0.012234100,  0.008394550,  0.00124085, -0.005712160,
                   0.006701010,  0.001425580, -0.00321647, -0.006995990,
                  -0.004203010,  0.001797540, -0.00159822,  0.003825800,
                   0.014404100,  0.006245060,  0.00669552, -0.013549400,
                   0.010984300, -0.002781280,  0.00685562, -0.000148740,
                  -0.007742760, -0.007584130, -0.01161810,  0.005733910,
                  -0.001070200,  0.000250656, -0.00706364, -0.004227190,
                   0.008983580, -0.005314090, -0.00255123, -0.013936200,
                  -0.002668650,  0.001467640,  0.00982941, -0.006445430,
                  -0.000155460,  0.002810760,  0.00402597,  0.000324102])

    res = calib_approx(log_returns)

    """
    # assume that higher order theta is small
    sigma = np.std(xs)
    vega = kurtosis(xs) / 3 - 1
    theta = skew(xs) * sigma / (3 * vega)
    c = np.mean(xs) - theta

    # store calibration results
    res_approx = {}
    res_approx["c"] = c
    res_approx["theta"] = theta
    res_approx["sigma"] = sigma
    res_approx["vega"] = vega

    # return calibration results
    return res_approx


def calib_moments(m1: float,
                  m2: float,
                  m3: float,
                  m4: float,
                  x0: list[float]) -> dict:
    """
    Moment calibration of variance-gamma distribution.

    Michal Mackanic
    22/12/2023 v1.0

    Description
    -----------
    Calibration of variance-gamma distribution based on moments as described
    in paper "Fitting the variance-gamma model to financial data" by Eugene
    Seneta.

    Parameters
    ----------
    m1: float
        The first moment.
    m2: float
        The second moment.
    m3: float
        The third moment.
    m4: float
        The four moment.
    x0: list[float]
        Initial parameter estimate from which the optimization should start.
        list[float] of size 3 - We assume that theta is zero and thus
            the variance-gamma distribution is symmetric.
        list[float] of size 4 - We assume that theta is not necessarily zero
            and thus the variance-gamma distributio can be non-symetric.

    Returns
    -------
    dict[str, float]
        Dictionary containing parameter names and their estimated values.

    Example
    -------
    m1 = 0.00019224434
    m2 = 5.713632523818538e-05
    m3 = -1.8996709595888222e-08
    m4 = 9.205033282310362e-09

    # assuming theta == 0.0
    x0 = [0.00, 0.02, 0.50]  # c, sigma, vega
    res_symmetric = calib_moments(m1=m1, m2=m2, m3=m3, m4=m4, x0=x0)

    # assuming theta != 0.0
    x0 = [0.00, 0.02, 0.50, 0.0]  # c, sigma, vega, theta
    res_nonsymmetric = calib_moments(m1=m1, m2=m2, m3=m3, m4=m4, x0=x0)

    """
    # general function to be minimized
    def f_min(c: float,
              sigma: float,
              vega: float,
              theta: float,
              x0: list[float],
              m1: float,
              m2: float,
              m3: float,
              m4: float) -> float:

        # parameters in range
        if check_param_rngs(theta=theta,
                            sigma=sigma,
                            vega=vega):

            # calculate squre of differences
            aux = 0.0
            aux += (f_m1(c=c, theta=theta) - m1) ** 2
            aux += (f_m2(sigma=sigma, vega=vega, theta=theta) - m2) ** 2
            aux += (f_m3(sigma=sigma, vega=vega, theta=theta) - m3) ** 2
            aux += (f_m4(sigma=sigma, vega=vega, theta=theta) - m4) ** 2

            # return evaluated function
            return aux

        # parameters outside of range
        else:
            return 1e10

    # function to minimize assuming theta = 0.0
    if (len(x0) == 3):
        f = lambda x: f_min(c=x[0],
                            sigma=x[1],
                            vega=x[2],
                            theta=0.0,
                            x0=x0,
                            m1=m1,
                            m2=m2,
                            m3=m3,
                            m4=m4)
    # function to minimize assuming theta >= 0
    elif (len(x0) == 4):
        f = lambda x: f_min(c=x[0],
                            sigma=x[1],
                            vega=x[2],
                            theta=x[3],
                            x0=x0,
                            m1=m1,
                            m2=m2,
                            m3=m3,
                            m4=m4)
    else:
        raise ValueError("Unsupported initial estimated point!")

    # find optimum
    options = {"disp": True, "maxiter": 10000}
    res = opt.minimize(f,
                       x0=x0,
                       method="Nelder-Mead",
                       tol=1e-10,
                       options=options)

    # collect results from optimization
    res_opt = {}
    res_opt["c"] = res.x[0]
    res_opt["sigma"] = res.x[1]
    res_opt["vega"] = res.x[2]

    if (len(res.x) == 3):
        res_opt["theta"] = 0.0
    else:
        res_opt["theta"] = res.x[3]

    # return results
    return res_opt


def calib_ml(zs: np.array, x0: list[float]) -> dict:
    """
    Maximum likelihood calibration of variance-gamma distribution.

    Michal Mackanic
    22/12/2023 v1.0

    Description
    -----------
    Maximum likelihood calibration of variance-gamma distribution as described
    in paper "Fitting the variance-gamma model to financial data" by Eugene
    Seneta.

    Parameters
    ----------
    zs: np.array()
        Values of a random variables after subtracting mean to be used
        as an input to maximum likelihood method.
    x0: list[float]
        Initial parameter estimate from which the optimization should start.
        list[float] of size 2 - We assume that theta is zero and thus
            the variance-gamma distribution is symmetric.
        list[float] of size 3 - We assume that theta is not necessarily zero
            and thus the variance-gamma distributio can be non-symetric.

    Returns
    -------
    dict[str, float]
        Dictionary containing parameter names and their estimated values.

    Example
    -------
    log_returns =\
        np.array([ 0.003916850,  0.004087070,  0.00793661, -0.003913780,
                  -0.000569117, -0.005423170,  0.00585654,  0.003776990,
                  -0.000946684,  0.000979647, -0.00195614,  0.000596756,
                   0.004052870, -0.002022990,  0.00736313,  0.001281280,
                   0.001189640,  0.001595810,  0.01889530, -0.000836091,
                   0.015495700, -0.008116760,  0.00100443,  0.002836120,
                   0.001751430,  0.009349870,  0.01868300,  0.010451200,
                   0.006454090,  0.011938500, -0.00481187, -0.011903100,
                  -0.014443400,  0.007239490, -0.00168696, -0.012665200,
                  -0.008519010, -0.013490400, -0.00000001,  0.010538600,
                  -0.005031480, -0.006265950,  0.00428383,  0.005194460,
                   0.006284030,  0.011745600, -0.00130487,  0.008077090,
                  -0.013839400,  0.000079287, -0.00271317,  0.005875780,
                   0.000229292, -0.014844100,  0.00401502, -0.002298250,
                  -0.016536900, -0.009439200, -0.00215342,  0.000721036,
                  -0.012234100,  0.008394550,  0.00124085, -0.005712160,
                   0.006701010,  0.001425580, -0.00321647, -0.006995990,
                  -0.004203010,  0.001797540, -0.00159822,  0.003825800,
                   0.014404100,  0.006245060,  0.00669552, -0.013549400,
                   0.010984300, -0.002781280,  0.00685562, -0.000148740,
                  -0.007742760, -0.007584130, -0.01161810,  0.005733910,
                  -0.001070200,  0.000250656, -0.00706364, -0.004227190,
                   0.008983580, -0.005314090, -0.00255123, -0.013936200,
                  -0.002668650,  0.001467640,  0.00982941, -0.006445430,
                  -0.000155460,  0.002810760,  0.00402597,  0.000324102])

    # assuming theta == 0.0
    x0 = [0.02, 0.50]  # sigma, vega
    res_symmetric = calib_moments(m1=m1, m2=m2, m3=m3, m4=m4, x0=x0)

    # assuming theta != 0.0
    x0 = [0.02, 0.50, 0.0]  # sigma, vega, theta
    res_nonsymmetric = calib_moments(m1=m1, m2=m2, m3=m3, m4=m4, x0=x0)

    """
    # log variance-gamma distribution used in maximum likelihood approach
    def pdf_ml(zs: np.array,
               theta: float,
               sigma: float,
               vega: float) -> np.array:

        pdf = 0.0
        pdf += np.log(2)
        pdf += (theta * (zs + theta) / sigma ** 2)

        pdf -= np.log(sigma * np.sqrt(2 * np.pi) * vega ** (1 / vega))
        pdf -= np.log(gamma_func(1 / vega))

        aux = 0.0
        aux += np.log(np.abs(zs + theta))
        aux -= np.log(np.sqrt(2 * sigma ** 2 / vega + theta ** 2))
        aux *= (1 / vega - 0.5)
        pdf += aux

        nu = 1 / vega - 0.50
        omegas = 0.0
        omegas += np.abs(zs + theta)
        omegas *= np.sqrt(2 * sigma ** 2 / vega + theta ** 2)
        omegas /= sigma ** 2

        pdf += np.log([bessel3(nu=nu, omega=omega) for omega in omegas])

        return pdf

    # general function to be minimized
    def f_min(zs: list[float],
              sigma: float,
              vega: float,
              theta: float) -> float:

        # parameters in range
        if check_param_rngs(theta=theta,
                            sigma=sigma,
                            vega=vega):

            # calculate maximum likelihood
            ml = pdf_ml(zs=zs, sigma=sigma, vega=vega, theta=theta)
            ml = -np.sum(ml)

            # return maximum likelihood
            return ml

        # parameters outside of range
        else:
            return 1e10

    # function to minimize assuming theta = 0.0
    if (len(x0) == 2):

        f = lambda x: f_min(zs=zs,
                            sigma=x[0],
                            vega=x[1],
                            theta=0.0)

    # function to minimize assuming theta != 0
    elif (len(x0) == 3):

        f = lambda x: f_min(zs=zs,
                            sigma=x[0],
                            vega=x[1],
                            theta=x[2])

    else:
        raise ValueError("Unsupported initial estimated point!")

    # find optimum
    options = {"disp": True, "maxiter": 10000}
    res = opt.minimize(f,
                       x0=x0,
                       method="Nelder-Mead",
                       tol=1e-10,
                       options=options)

    # collect results from optimization
    res_ml = {}
    res_ml["sigma"] = res.x[0]
    res_ml["vega"] = res.x[1]
    if (len(x0) == 2):
        res_ml["theta"] = 0.0
    else:
        res_ml["theta"] = res.x[2]

    # return results
    return res_ml


def pdf(x: np.array,
        c: float,
        theta: float,
        sigma: float,
        vega: float) -> np.array:
    """
    PDF function of variance-gamma distribution.

    Michal Mackanic
    22/12/2023 v1.0

    Description
    -----------
    PDF funciton of variance-gamma distribution as defined in paper
    "Fitting the variance-gamma model to financial data" by Eugene Seneta.
    Please note that implementation of CDF function is rather problematic as
    PDF function is quite steep; see graph produced by function main(). For
    the same reason it is very difficult to implement inverse function of
    variance-gamma distribution.

    Parameters
    ----------
    x : np.array
        Value for PDF to be valued at.
    c : float
        Parameter c.
    theta : float
        Parameter theta.
    sigma : float
        Parameter sigma.
    vega : float
        Parameter vega.

    Returns
    -------
    pdf : TYPE
        DESCRIPTION.

    """
    pdf = 2 * np.exp(theta * (x - c) / sigma ** 2)
    pdf /= sigma * np.sqrt(2 * np.pi)
    pdf /= vega ** (1 / vega) * gamma_func(1 / vega)

    aux = np.abs(x - c) / np.sqrt(2 * sigma ** 2 / vega + theta ** 2)
    pdf *= aux ** (1 / vega - 0.5)

    omegas = np.abs(x - c) * np.sqrt(2 * sigma ** 2 / vega + theta ** 2)
    omegas /= sigma ** 2
    pdf *= np.array([bessel3(nu=1/vega-0.50,
                             omega=omega) for omega in omegas], float)

    return pdf


def main(file_nm: str) -> dict[str, dict[str, float]]:
    """
    Demonstrate variance-gamma functions implemented above.

    Michal Mackanic
    22/12/2023 v1.0

    Parameters
    ----------
    file_nm : str
        File with S&P 500 price history.

    Returns
    -------
    res : dict[str, dict[str, float]]
        Dictionary with calibration results.

    Example
    -------
    res = main(file_nm="data//S&P500_historical_data.csv")

    """
    # read historical values of S&P 500 index
    print("Loading S&P 500 history from a file...")
    sp500 = pd.read_csv(file_nm, low_memory=False)
    sp500 = np.array(sp500["Close"])  # keep close price only

    # get log-returns
    print("Calculating log-returns...")
    xs = get_log_returns(sp500)

    # remove mean
    print("Removing mean from log-returns...")
    zs = xs - np.mean(xs)

    # approximative calibration based on assumption that higher orders of
    # theta are negligible
    print("Approximate calibration...")
    res_approx = calib_approx(xs=xs)

    # determine moments
    print("Calculationg the first four moments...")
    [m1, m2, m3, m4] = get_moments(xs)

    # initial parameter estimates for moments calibration
    x0 = [0.00, 0.02, 0.50]  # c, sigma, vega

    # calibrate variance gamma distribution on moments assuming theta = 0.0
    print("Calibrating variance gamma using moments and assuming " +
          "symmetric distribution...")
    res_mo = calib_moments(m1=m1,
                           m2=m2,
                           m3=m3,
                           m4=m4,
                           x0=x0)

    # calibrate variance gamma distribution on moments assuming theta != 0.0
    # using paramaters from the previous step as initial estimates
    print("Calibrating variance gamma using moments and assuming " +
          "asymmetric distribution...")
    x0 = [res_mo["c"], res_mo["sigma"], res_mo["vega"], res_mo["theta"]]
    res_mo = calib_moments(m1=m1,
                           m2=m2,
                           m3=m3,
                           m4=m4,
                           x0=x0)

    # calibrate variance gamma distribution using maximum likelihood method
    # and parameters from the previous step as initial estimates

    print("Calibrationg variance gamma using maximum likelihood...")
    x0 = [0.01, 1.0, 0.0]  # sigma, vega, theta
    res_ml = calib_ml(zs=zs, x0=x0)
    res_ml["c"] = np.mean(xs) - res_ml["theta"]

    # return results
    print("Collecting calibration results...")

    res = {}
    res["moments"] = {}
    res["moments"]["m1"] = m1
    res["moments"]["m2"] = m2
    res["moments"]["m3"] = m3
    res["moments"]["m4"] = m4

    res["calib_approx"] = res_approx
    res["calib_mo"] = res_mo
    res["calib_ml"] = res_ml

    # compare CDF functions based on the three calibration approaches
    print("Plotting PDF functions...")

    x = np.linspace(-0.01, 0.01, 100)

    pdf_approx =\
        pdf(x=x,
            c=res["calib_approx"]["c"],
            theta=res["calib_approx"]["theta"],
            sigma=res["calib_approx"]["sigma"],
            vega=res["calib_approx"]["vega"])

    pdf_mo =\
        pdf(x=x,
            c=res["calib_mo"]["c"],
            theta=res["calib_mo"]["theta"],
            sigma=res["calib_mo"]["sigma"],
            vega=res["calib_mo"]["vega"])

    pdf_ml =\
        pdf(x=x,
            c=res["calib_ml"]["c"],
            theta=res["calib_ml"]["theta"],
            sigma=res["calib_ml"]["sigma"],
            vega=res["calib_ml"]["vega"])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, pdf_approx, color="tab:red", label="approximate")
    ax.plot(x, pdf_mo, color="tab:blue", label="moment matching")
    ax.plot(x, pdf_ml, color="tab:orange", label="maximum likelihood")
    ax.set_title("PDF based on maximum approximate calibration vs. " +
                 "likelihood vs. moment matching")
    plt.legend(loc="best")
    plt.show()

    # return calibraiton results
    return res


def simul(S_0: float,
          simuls_no: int,
          c: float,
          theta: float,
          sigma: float,
          vega: float) -> list[np.array, np.array, float, float, float, float]:
    """
    Simulate random draws from variance-gamma distribution.

    Michal Mackanic
    22/12/2023 v1.0

    Description
    -----------
    Simulate random draws from variance-gamma distribution over a unit time
    step. The "unit" time step corresponds to a time step used during the
    calibration process. To simulate random draws, methodology of paper
    "Fitting the variance-gamma model to financial data" by Eugene
    Seneta was used.

    Parameters
    ----------
    S_0 : float
        Spot price of the underlying asset.
    simuls_no : int
        Number of
    c : float
        Parameter c.
    theta : float
        Parameter theta.
    sigma : float
        Parameter sigma.
    vega : float
        Parameter vega.

    Returns
    -------
    list[np.array, np.array, float, float, float, float]
        S : np.array
            Simulated prices.
        dS : np.array
            Simulated log-returns.
        m1 : float
            The first moment of the simulated log-returns.
        m2 : float
            The second moment of the simulated log-returns.
        m3 : float
            The third moment of the simulated log-returns.
        m4 : float
            The fourth moment of the simulated log-returns.

    Example
    -------
    [S, dS, m1, m2, m3, m4] =\
        simul(S_0=1.0,
              simuls_no=1000000,
              c=0.0010631369007665743,
              theta=-0.0006855360204174363,
              sigma=0.011166376979989283,
              vega=4.400859336304571)

    """
    # simulate delta time as random gamma variable
    dt = gamma_dist.rvs(a=1/vega, loc=0, scale=vega, size=simuls_no)

    # simulate log change of the underlying asset in unit time step
    # (corresponds to calibration time step)
    dS = c + theta * dt + sigma * np.sqrt(dt) * norm_dist.rvs(size=simuls_no)

    # calculate simulated prices at the end of the the unit time step
    S = S_0 * np.exp(dS)

    plt.hist(x=dS, bins=1000)
    plt.show()

    counts, bins = np.histogram(dS, bins=1000)
    plt.stairs(counts, bins)

    [m1, m2, m3, m4] = get_moments(dS)

    return [S, dS, m1, m2, m3, m4]
