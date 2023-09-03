import numpy as np
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader

# import Black-Scholes formulas
file_nm = '/home/macky/Documents/Hull/black_scholes/main.py'
bs = SourceFileLoader('main', file_nm).load_module()

def main(S_0, Ks, r, q, sigmas, T):
    """
    Michal Mackanic 21/08/2023 v1.0

    This function calculates and plots implied probabilities in line with
    appendix of chapter 20 of "Options, Futures, and Other Derivatives", e11
    by John Hull.

    input:
        S_0: float
            spot price of the undelying stock
        Ks: list of floats
            list of strike prices; length of Ks must corresponds to length of
            sigmas
        r: float
            annual continuous risk-free interest rate
        q: float
            annual continuous dividend yield
        sigmas: list of floats
            list of implied volatities corresponding to individual strikes
            specified through variable Ks
        T: float
            option maturity in years
    output:
        S_T: list of floats
            price of underlying stock at option maturity for which implied
            probability was calculated
        g_T: list of floats
            implied probability assigned to individual elements of S_T list

    example:
        S_0 = 10
        Ks = [6, 7, 8, 9, 10, 11, 12, 13, 14]
        r = 0.03
        q = 0.00
        sigmas = [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22]
        T = 3 / 12
        [S_T, g_T] = main(S_0=S_0, Ks=Ks, r=r, q=q, sigmas=sigmas, T=T)
    """

    #### add interpolated strike price and volatility

    # lists holding original and interpolated values
    Ks_adj = []
    Ks_adj.append(Ks[0])
    sigmas_adj = []
    sigmas_adj.append(sigmas[0])

    # go through input strike prices and implied volatilities and
    # interpolate values
    for idx in range(len(Ks) - 1):

        # interpolate strike price K
        K_1 = Ks_adj[-1]
        K_2 = Ks[idx + 1]
        K_interp = (K_1 + K_2) / 2
        Ks_adj.append(K_interp)
        Ks_adj.append(K_2)

        # interpolate implied volatility
        sigma_1 = sigmas_adj[-1]
        sigma_2 = sigmas[idx + 1]
        sigma_interp = (sigma_1 + sigma_2) / 2
        sigmas_adj.append(sigma_interp)
        sigmas_adj.append(sigma_2)

    #### calculate option values
    calls = []
    for idx in range(len(Ks_adj)):

        # calculate greeks for call option for a particular spot price
        call = bs.BlackScholes(tp='call',
                              S_0=S_0,
                              K=Ks_adj[idx],
                              r=r,
                              q=q,
                              sigma=sigmas_adj[idx],
                              T=T,
                              greeks=False)
        calls.append(call.f)

    #### calculate implied probability distribution
    g_T = []
    S_T = []
    for idx in range(1, len(Ks_adj) - 1, 2):

        # extract call prices
        c1 = calls[idx - 1]
        c2 = calls[idx]
        c3 = calls[idx + 1]

        # extract S_T
        K = Ks_adj[idx]
        S_T.append(K)

        # determine delta
        delta = Ks_adj[idx] - Ks_adj[idx - 1]

        # determine implied probability
        g = np.exp(r * T) * (c1 + c3 - 2 * c2) / (delta ** 2)
        g_T.append(g)

    #### plot implied distribution
    plt.bar(S_T, g_T, width=1.0)
    plt.title('Implied probability distribution')
    plt.xlabel('$S_T$')
    plt.ylabel('probability density')
    plt.show()

    #### return prices and their corresponding implied probabilities
    return S_T, g_T