import numpy as np

from scipy.stats import norm


class BlackScholes:
    """
    Michal Mackanic 08/08/2023 v1.0

    This class represents European option valued via Black-Scholes model using
    S_0.

    __init__(self, tp, S_0, K, r, q, sigma, T):
        initiate object and calculate option value using Black-Scholes formula;
        see chapter 17 of "Options, Futures, and Other Derivates" 11e by John
        Hull
        variables (set 1):
            tp: float
                option type - 'call' / 'put'
            S_0: float
                stock price at time t = 0
            K: float
                strike
            r: float
                continuous annual risk-free rate
            q: float
                continuous annual dividend yield
            sigma: float
                annual standard deviation entering Black-Scholes formula
            T: float
                option maturity in years
        variables (set 2):
            tp: float
                option type - 'call' / 'put'
            F_0: float
                expected forward price as seen at time t = 0
            K: float
                strike
            r: float
                continuous annual risk-free rate
            sigma: float
                annual standard deviation entering Black-Scholes formula
            T: float
                option maturity in years

    example 1:
        # Black-Scholes formula based on S_0
        tp='call'
        S_0=100
        K=80
        r=0.05
        q=0.01
        sigma=0.20
        T=1.00
        opt = BlackScholes(tp=tp, S_0=S_0, K=K, r=r, q=q, sigma=sigma, T=T)
        print('option price: ' + '{:10.3f}'.format(opt.f))

    example 2:
        # Black-Scholes formula based on F_0
        tp='call'
        S_0=100
        K=80
        r=0.05
        q=0.01
        sigma=0.20
        T=1.00
        F_0 = S_0 * np.exp((r - q) * T)
        opt = BlackScholes(tp=tp, F_0=F_0, K=K, r=r, sigma=sigma, T=T)
        print('option price: ' + '{:10.3f}'.format(opt.f))


    example 3:
        # symmetry of FX options
        # put option
        tp='put'
        fx=1.10
        K=1.05
        r=0.05
        r_f=0.03
        sigma=0.20
        T=1.00
        put = BlackScholes(tp=tp, S_0=fx, K=K, r=r, q=r_f, sigma=sigma, T=T)
        print('put option price:  ' + '{:10.3f}'.format(put.f))
        # call option
        tp='call'
        fx=1.00
        K=1.10/1.05
        r=0.03
        r_f=0.05
        sigma=0.20
        T=1.00
        call = BlackScholes(tp=tp, S_0=fx, K=K, r=r, q=r_f, sigma=sigma, T=T)
        print('call option price: ' + '{:10.3f}'.format(call.f * K))
    """

    def __init__(self, **kwargs):

        # store variables
        parameters = {}
        for key, value in kwargs.items():
            parameters[key] = value
        self.parameters = parameters

        # check that all parameters were specified
        S_0_param_nms = np.sort(['tp', 'S_0', 'K', 'r', 'q', 'sigma', 'T'])
        F_0_param_nms = np.sort(['tp', 'F_0', 'K', 'r', 'sigma', 'T'])
        obj_param_nms = np.sort(list(self.parameters.keys()))

        if (len(obj_param_nms) == len(S_0_param_nms)):
            S_0_param_match = (obj_param_nms == S_0_param_nms).all()
        else:
            S_0_param_match = False

        if (len(obj_param_nms) == len(F_0_param_nms)):
            F_0_param_match = (obj_param_nms == F_0_param_nms).all()
        else:
            F_0_param_match = False

        if (S_0_param_match):
            self.version = 'S_0'
        elif (F_0_param_match):
            self.version = 'F_0'
        else:
            raise ValueError('Incorrect parameters!')

        # check option type
        if (self.parameters['tp'] not in ['call', 'put']):
            raise ValueError (self.tp + ' is not a supported option type!')

        # Black-Scholes formula based on S_0
        if (self.version == 'S_0'):

            # extract parameters
            tp = self.parameters['tp']
            S_0 = self.parameters['S_0']
            K = self.parameters['K']
            r = self.parameters['r']
            q = self.parameters['q']
            sigma = self.parameters['sigma']
            T = self.parameters['T']

            # calculate d1 and d2
            d1 = (np.log(S_0 / K) + (r - q + sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            self.parameters['d1'] = d1
            self.parameters['d2'] = d2

            # calculate option value
            if (tp == 'call'):
                self.f = S_0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                self.f = K * np.exp(-r * T) * norm.cdf(-d2) - S_0 * np.exp(-q * T) * norm.cdf(-d1)

        # Black-Scholes formula based on F_0 = S_0 * exp((r - q) * T); if we know F_0 we do not
        # have to estimate dividend yield q
        else:

            # extract parameters
            tp = self.parameters['tp']
            F_0 = self.parameters['F_0']
            K = self.parameters['K']
            r = self.parameters['r']
            sigma = self.parameters['sigma']
            T = self.parameters['T']

            # calculate d1 and d2
            d1 = (np.log(F_0 / K) + (sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = (np.log(F_0 / K) - (sigma ** 2) * T) / (sigma * np.sqrt(T))
            self.parameters['d1'] = d1
            self.parameters['d2'] = d2

            # calculate option value
            if (tp == 'call'):
                self.f = F_0 * np.exp(-r * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                self.f = K * np.exp(-r * T) * norm.cdf(-d2) - F_0 * np.exp(-r * T) * norm.cdf(-d1)