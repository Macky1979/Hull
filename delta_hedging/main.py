import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader

# import Black-Scholes formulas
file_nm = '/home/macky/Documents/Hull/black_scholes/main.py'
bs = SourceFileLoader('main', file_nm).load_module()

def main(tp, S, K, r, sigma, T=1):
    """
    Michal Mackanic 08/08/2023 v1.0

    This function illustrates dynamic hedging as explained in chapter 19.4
    of "Options, Futures, and Other Derivatives", e11 by John Hull

    input:
        tp: float
            option type - 'call' / 'put'
        S: list of floats
            realized stock prices
        K: float
            strike
        r: float
            continuous annual risk-free rate
        sigma: float
            annual standard deviation entering Black-Scholes formula
        T: float
                option maturity in years
    output:
        costs: float
            total cost of dynamic hedging
        tbl: pd.DataFrame
            table with comprehensive information on delta hedging

    example 1:
        #table 19.2
        tp = 'call'
        S = [49.00, 48.12, 47.37, 50.25, 51.75, 53.12, 53.00,
             51.87, 51.38, 53.00, 49.88, 48.50, 49.88, 50.37,
             52.13, 51.88, 52.87, 54.87, 54.62, 55.87, 57.25]
        K = 50
        r = 0.05
        sigma = 0.20
        T = 20/52
        info= main(tp=tp, S=S, K=K, r=r, sigma=sigma, T=T)
        info = main(tp=tp, S=S, K=K, r=r, sigma=sigma, T=T)
        opt = bs.BlackScholes(tp=tp,
                              S_0=S[0],
                              K=K,
                              r=r,
                              q=0.0,  # we assume stocks with zero dividend yield
                              sigma=sigma,
                              T=T)
        print('discounted cost of dynamic hedging: ' + '{:5.3f}'.format(info[0]))
        print('option price: ' + '{:5.3f}'.format(opt.f))

    example 2:
        #table 19.3
        tp = 'call'
        S = [49.00, 49.75, 52.00, 50.00, 48.38, 48.25, 48.75,
             49.63, 48.25, 48.25, 51.12, 51.50, 49.88, 49.88,
             48.75, 47.50, 48.00, 46.25, 48.13, 46.63, 48.12]
        K = 50
        r = 0.05
        sigma = 0.20
        T = 20/52
        info = main(tp=tp, S=S, K=K, r=r, sigma=sigma, T=T)
        opt = bs.BlackScholes(tp=tp,
                              S_0=S[0],
                              K=K,
                              r=r,
                              q=0.0,  # we assume stocks with zero dividend yield
                              sigma=sigma,
                              T=T)
        print('discounted cost of dynamic hedging: ' + '{:5.3f}'.format(info[0]))
        print('option price: ' + '{:5.3f}'.format(opt.f))
    """

    # get number of steps
    steps_no = len(S)

    # calculate time step length
    dt = T / (len(S) - 1)

    # calculate times for individual time steps
    t =  np.cumsum([0] + [dt] * (steps_no - 1))

    # list of dictionaries holding comprehensive information on dynamic hedging
    tbl = []

    # go step by step
    for step_idx in range(steps_no):

        # evaluate option
        opt = bs.BlackScholes(tp=tp,
                              S_0=S[step_idx],
                              K=K,
                              r=r,
                              q=0.0,  # we assume stocks with zero dividend yield
                              sigma=sigma,
                              T=T-t[step_idx])

        # store relevant step information
        step = {}
        step['t'] = t[step_idx]
        step['dt'] = dt
        step['S'] = S[step_idx]
        step['delta'] = opt.delta
 
        step['shares purchased'] = opt.delta
        if (len(tbl) > 0):
            step['shares purchased'] -= tbl[-1]['delta']

        step['cost of shares purchased'] = step['S'] * step['shares purchased']

        step['cost of financing'] = step['cost of shares purchased'] * np.exp(r * (T - step['t']))

        step['cumulative cost of financing'] = step['cost of financing']
        if (len(tbl) > 0):
            step['cumulative cost of financing'] += tbl[-1]['cumulative cost of financing']

        tbl.append(step)

    # final table
    tbl = pd.DataFrame(tbl)

    # call payoff at maturity
    if (tp == 'call'):
        # option is exercised
        if (K < step['S']):
            cost = K
        # option is not exercised
        else:
            cost = 0.0
    # put payoff at maturity
    else:
        # option is exercised
        if (K > step['S']):
            cost = -step['S']
        # option is not exercised
        else:
            cost = 0.0

    # add cumulative cost of financing as of option maturity to get total cost
    # of dynamic hedging
    cost -= step['cumulative cost of financing']

    # discount total cost
    cost *= np.exp(-r * T)

    # return total cost of delta hedging and table with comprehensive information
    return [cost, tbl]