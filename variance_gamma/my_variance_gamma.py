import math
import numpy as np
import pandas as pd
import scipy.optimize as opt

from scipy.special import gamma
import scipy.integrate as integrate

import warnings
warnings.filterwarnings("error")


def bessel3(nu, omega):

    f = lambda x: np.exp(-omega * math.cosh(x)) * math.cosh(nu * x)
    return integrate.quad(f, 0, 30 / nu)[0]


def get_log_returns(vals: np.array) -> np.array:

    # calculate log returns
    xs = np.log(vals[0:-1] / vals[1:])

    # return log returns
    return xs


def check_param_rngs(theta: float, sigma: float, vega: float) -> bool:

        # check parameter ranges (optimized for S&P 500)
        if ((theta <= -2.0) or (theta > 2.0)):
            return False
        elif ((sigma <= 1e-5) or (sigma > 2.0)):
            return False
        elif (vega <= 1e-2):
            return False
        else:
            return True


def get_moments(vals: np.array) -> list[float]:

    # calculate moments
    m1 = np.mean(vals)  # mean
    m2 = np.mean([(val - m1) ** 2 for val in vals])  # variance
    m3 = np.mean([(val - m1) ** 3 for val in vals])
    m4 = np.mean([(val - m1) ** 4 for val in vals])

    # return moments
    return [m1, m2, m3, m4]

# derive 1st moment from parameters of variance-gamma distribution
def f_m1(c: float,
         theta: float = 0.0) -> float:
    return c + theta

# derive 2nd moment from parameters of variance-gamma distribution
def f_m2(sigma: float,
         vega: float,
         theta: float = 0.0) -> float:
    return sigma ** 2 + theta ** 2 * vega

# derive 3rd moment from parameters of variance-gamma distribution
def f_m3(sigma: float,
         vega: float,
         theta: float = 0.0) -> float:
    return 2 * theta ** 3 * vega ** 2 + 3 * sigma ** 2 * theta * vega

# derive 4th moment from parameters of variance-gamma distribution
def f_m4(sigma: float,
         vega: float,
         theta: float = 0.0) -> float:

    m4 =\
        3 * sigma ** 4 * vega + 12 * sigma ** 2 * theta ** 2 * vega ** 2 +\
        6 * theta ** 4 * vega ** 3 + 3 * sigma ** 4 +\
        6 * sigma ** 2 * theta ** 2 * vega + 3 * theta ** 4 * vega ** 2

    return m4

def calib_moments(m1: float,
                  m2: float,
                  m3: float,
                  m4: float,
                  x0: list[float]) -> dict:

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
        f = lambda x : f_min(c=x[0],
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
        f = lambda x : f_min(c=x[0],
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


def pdf_ml(z: np.array,
           theta: float,
           sigma: float,
           vega: float) -> np.array:

    pdf = 0.0
    pdf += np.log(2)
    pdf += (theta * (z + theta) / sigma ** 2)

    pdf -= np.log(sigma * np.sqrt(2 * np.pi) * vega ** (1 / vega))
    pdf -= np.log(gamma(1 / vega))

    aux = 0.0
    aux += np.log(np.abs(z + theta))
    aux -= np.log(np.sqrt(2 * sigma ** 2 / vega + theta ** 2))
    aux *= (1 / vega - 0.5)
    pdf += aux

    nu = 1 / vega - 0.50
    omegas = 0.0
    omegas += np.abs(z + theta) * np.sqrt(2 * sigma ** 2 / vega + theta ** 2)
    omegas /= sigma ** 2

    try:
        pdf += np.log([bessel3(nu=nu, omega=omega) for omega in omegas])
    except:
        print("bingo!")

    return pdf


def calib_ml(zs: np.array, x0: list[float]) -> dict:

    def f_min(z: np.array,
              sigma: float,
              vega: float,
              theta: float) -> float:

        # parameters in range
        if check_param_rngs(theta=theta,
                            sigma=sigma,
                            vega=vega):

            # calculate maximum likelihood
            ml = pdf_ml(z=z, theta=theta, sigma=sigma, vega=vega)
            ml = -np.sum(ml)

            # return maximum likelihood
            return ml

        # parameters outside of range
        else:
            return 1e10

    # function to minimize assuming theta = 0.0
    if (len(x0) == 2):
        f = lambda x : f_min(z=zs, sigma=x[0], vega=x[1], theta=0.0)
    # function to minimize assuming theta >= 0
    elif (len(x0) == 3):
        f = lambda x : f_min(z=zs, sigma=x[0], vega=x[1], theta=x[2])
    else:
        raise ValueError("Unsupported initial estimated point!")

    # optimize parameters
    options = {"disp": True, "maxiter": 1000}
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


def main(file_nm: str) -> np.array:

    # read historical values of S&P 500 index
    print("Loading S&P 500 history from a file...")
    sp500 = pd.read_csv(file_nm, low_memory=False)
    sp500 = np.array(sp500["Close"])  # keep close price only

    # get log returns
    print("Calculating log returns...")
    xs = get_log_returns(sp500)

    # remove mean
    print("Removing mean from log returns...")
    zs = xs - np.mean(xs)

    # determine moments
    print("Calculationg the first four moments...")
    [m1, m2, m3, m4] = get_moments(xs)

    # initial parameter estimates for moments calibration
    x0 = [0.00, 0.02, 0.50] # c, sigma, vega

    # calibrate variance gamma distribution on moments assuming theta = 0.0
    print("Calibrating variance gamma using moments and assuming " +
          "symmetric distribution...")
    res_mo = calib_moments(m1=m1,
                           m2=m2,
                           m3=m3,
                           m4=m4,
                           x0=x0)

    """
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
    """

    # initial parameter estimates for maximum likelihood calibration assuming
    # theta = 0.0
    x0 = [0.02, 0.50] # sigma, vega, theta

    # calibrate variance gamma distribution using maximum likelihood method
    # and parameters from the previous step as initial estimates

    print("Calibrationg variance gamma using maximum likelihood...")
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

    res["calib_moments"] = res_mo
    res["calib_ml"] = res_ml

    return res


def pdf(x: np.array,
        c: float,
        theta: float,
        sigma: float,
        vega: float) -> np.array:

    pdf = 2 * np.exp(theta * (x - c) / sigma ** 2)
    pdf /= sigma * np.sqrt(2 * np.pi) * vega ** (1 / vega) * gamma(1 / vega)

    aux = np.abs(x - c) / np.sqrt(2 * sigma ** 2 / vega + theta ** 2)
    pdf *=  aux ** (1 / vega - 0.5)

    omegas = np.abs(x - c) * np.sqrt(2 * sigma ** 2 / vega + theta ** 2)
    omegas /= sigma ** 2
    pdf *= np.array([bessel3(nu=1/vega-0.50,
                             omega=omega) for omega in omegas], float)

    return pdf


def cdf(x: np.array,
        c: float,
        theta: float,
        sigma: float,
        vega: float) -> np.array:

    f = lambda x: pdf(x=np.array([x], float),
                      c=c,
                      theta=theta,
                      sigma=sigma,
                      vega=vega)[0]

    return np.array([integrate.quad(f, -5, x_)[0] for x_ in x], float)


def inv(q: float,
        c: float,
        theta: float,
        sigma: float,
        vega: float) -> float:

    f = lambda x: np.abs(cdf(x=x,
                             c=c,
                             theta=theta,
                             sigma=sigma,
                             vega=vega) - q)

    options = {"disp": False, "maxiter": 100}
    res = opt.minimize(f,
                       x0=0.50,
                       method="Nelder-Mead",
                       tol=1e-10,
                       options=options)

    return res.x[0]


res = main(file_nm="data//S&P500_historical_data.csv")

"""
import time
x = np.linspace(-1, 1, 100)
t = time.time()
aux = cdf(x=x, c=2.585e-4, theta=-1.510e-4, sigma=0.08, vega=0.422)
elapsed = time.time() - t
print(elapsed)
"""