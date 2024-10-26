"""Black-Scholes model."""

# pylint: disable=C0103

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
            greeks: bool
                True - calculate Greeks
                False - do not calculate Greeks
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
            greeks: bool
                True - calculate Greeks
                False - do not calculate Greeks
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
        tp = 'call'
        greeks = True
        S_0 = 100
        K = 80
        r = 0.05
        q = 0.01
        sigma = 0.20
        T = 1.00
        max_msg_len = 20

        opt = BlackScholes(tp=tp, greeks=greeks, S_0=S_0, K=K, r=r, q = q, sigma=sigma, T=T)
        opt.calc()

        msg = 'price: '
        msg_len = int(max_msg_len - len(msg))
        num = '{:.3f}'.format(opt.f)
        msg_len -= len(num)
        msg += ' ' * msg_len + num
        print(msg)

        for key, value in opt.greeks.items():
            msg = key + ': '
            msg_len = int(max_msg_len - len(msg))
            num = '{:.3f}'.format(opt.greeks[key])
            msg_len -= len(num)
            msg += ' ' * msg_len + num
            print(msg)

    example 2:
        # Black-Scholes formula based on F_0
        tp = 'call'
        greeks = False
        S_0 = 100
        K = 80
        r = 0.05
        q = 0.01
        sigma = 0.20
        T = 1.00
        F_0 = S_0 * np.exp((r - q) * T)
        opt = BlackScholes(tp=tp, greeks=greeks, F_0=F_0, K=K, r=r, sigma=sigma, T=T)
        opt.calc()
        print('option price: ' + '{:10.3f}'.format(opt.f))

    example 3:
        # symmetry of FX options
        # put option
        tp = 'put'
        greeks = False
        fx = 1.10
        K = 1.05
        r = 0.05
        r_f = 0.03
        sigma = 0.20
        T = 1.00
        put = BlackScholes(tp=tp, greeks=greeks, S_0=fx, K=K, r=r, q=r_f, sigma=sigma, T=T)
        put.calc()
        print('put option price:  ' + '{:10.3f}'.format(put.f))
        # call option
        tp = 'call'
        greeks = False
        fx = 1.00
        K = 1.10/1.05
        r = 0.03
        r_f = 0.05
        sigma = 0.20
        T = 1.00
        call = BlackScholes(tp=tp, greeks=greeks, S_0=fx, K=K, r=r, q=r_f, sigma=sigma, T=T)
        call.calc()
        print('call option price: ' + '{:10.3f}'.format(call.f * K))
    """

    def __init__(self,
                 **kwargs: dict[str, float | bool]) -> None:

        # store variables
        parameters = {}
        for key, value in kwargs.items():
            parameters[key] = value
        self.parameters = parameters

        # check that all parameters were specified
        S_0_param_nms = np.sort(['tp', 'S_0', 'K', 'r', 'q', 'sigma', 'T'])
        F_0_param_nms = np.sort(['tp', 'F_0', 'K', 'r', 'sigma', 'T'])
        obj_param_nms = np.sort(list(self.parameters.keys()))

        if 'greeks' in obj_param_nms:
            obj_param_nms = np.delete(obj_param_nms, np.where(obj_param_nms == 'greeks'))

        if len(obj_param_nms) == len(S_0_param_nms):
            S_0_param_match = (obj_param_nms == S_0_param_nms).all()
        else:
            S_0_param_match = False

        if len(obj_param_nms) == len(F_0_param_nms):
            F_0_param_match = (obj_param_nms == F_0_param_nms).all()
        else:
            F_0_param_match = False

        if S_0_param_match:
            self.version = 'S_0'
        elif F_0_param_match:
            self.version = 'F_0'
        else:
            raise ValueError('Incorrect parameters!')

        # check option type
        if self.parameters['tp'] not in ['call', 'put']:
            raise ValueError (self.parameters['tp'] + ' is not a supported option type!')

        # set up Greeks calculation
        self.greeks = {}
        if not hasattr(self.parameters, 'greeks'):
            self.parameters['greeks'] = False

        # option value
        self.f = None

    def calc(self):

        """Calculate option value."""

        # Black-Scholes formula based on S_0
        if self.version == 'S_0':
            self.calc_S0()

        # Black-Scholes formula based on F_0 = S_0 * exp((r - q) * T); if we know F_0 we do not
        # have to estimate dividend yield q
        else:
            self.calc_F0()


    def calc_S0(self):
        """Black-Scholes formula based on S_0."""

        # extract parameters
        tp = self.parameters['tp']
        S_0 = self.parameters['S_0']
        K = self.parameters['K']
        r = self.parameters['r']
        q = self.parameters['q']
        sigma = self.parameters['sigma']
        T = self.parameters['T']
        greeks = self.parameters['greeks']

        # calculate d1 and d2
        d1 = (np.log(S_0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        self.parameters['d1'] = d1
        self.parameters['d2'] = d2

        # calculate call value
        if tp == 'call':

            self.f =\
                S_0 * np.exp(-q * T) * norm.cdf(d1) -\
                K * np.exp(-r * T) * norm.cdf(d2)

            # calculate Greeks
            if greeks:

                self.greeks['delta'] =\
                    np.exp(-q * T) * norm.cdf(d1)

                self.greeks['gamma'] =\
                    np.exp(-q * T) * norm.pdf(d1) / (S_0 * sigma * np.sqrt(T))

                self.greeks['theta'] =\
                    -S_0 * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T)) +\
                    q * S_0 * norm.pdf(d1) * np.exp(-q * T) -\
                    r * K * np.exp(-r * T) * norm.cdf(d2)

                self.greeks['vega'] =\
                    S_0 * np.sqrt(T) * norm.pdf(d1) * np.exp(-q * T)

                self.greeks['rho_r'] =\
                    K * T * np.exp(-r * T) * norm.cdf(d2)

                self.greeks['rho_q'] =\
                    -T * np.exp(-q * T) * S_0 * norm.cdf(d1)

        # calculate put value
        else:

            self.f =\
                K * np.exp(-r * T) * norm.cdf(-d2) -\
                S_0 * np.exp(-q * T) * norm.cdf(-d1)

            # calculate Greeks
            if greeks:
                self.greeks = {}

                self.greeks['delta'] =\
                    np.exp(-q * T) * (norm.cdf(d1) - 1.0)

                self.greeks['gamma'] =\
                    np.exp(-q * T) * norm.pdf(d1) / (S_0 * sigma * np.sqrt(T))

                self.greeks['theta'] =\
                    -S_0 * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T)) -\
                    q * S_0 * norm.pdf(-d1) * np.exp(-q * T) +\
                    r * K * np.exp(-r * T) * norm.cdf(-d2)

                self.greeks['vega'] =\
                    S_0 * np.sqrt(T) * norm.pdf(d1) * np.exp(-q * T)

                self.greeks['rho_r'] =\
                    -K * T * np.exp(-r * T) * norm.cdf(-d2)

                self.greeks['rho_q'] =\
                    T * np.exp(-q * T) * S_0 * norm.cdf(-d1)

    def calc_F0(self):
        """Black-Scholes formula based on F_0."""

        # extract parameters
        tp = self.parameters['tp']
        F_0 = self.parameters['F_0']
        K = self.parameters['K']
        r = self.parameters['r']
        sigma = self.parameters['sigma']
        T = self.parameters['T']
        greeks = self.parameters['greeks']

        # calculate d1 and d2
        d1 = (np.log(F_0 / K) + (sigma ** 2) * T / 2) / (sigma * np.sqrt(T))
        d2 = (np.log(F_0 / K) - (sigma ** 2) * T / 2) / (sigma * np.sqrt(T))
        self.parameters['d1'] = d1
        self.parameters['d2'] = d2

        # calculate call value
        if tp == 'call':

            self.f =\
                F_0 * np.exp(-r * T) * norm.cdf(d1) -\
                K * np.exp(-r * T) * norm.cdf(d2)

            # calculate delta
            if greeks:
                self.greeks = {}
                self.greeks['delta'] = norm.cdf(d1)

        # calculate put value
        else:

            self.f =\
                K * np.exp(-r * T) * norm.cdf(-d2) -\
                F_0 * np.exp(-r * T) * norm.cdf(-d1)

            # calculate delta
            if greeks:
                self.greeks = {}
            self.greeks['delta'] = norm.cdf(d1) - 1.0

    def get_S_0_ladder(self, ladder_points: list[float]) -> list[list[float]]:
        """Return S_0 ladder."""

        # get base option value
        self.calc()
        S_0 = self.parameters['S_0']
        base_npv = self.f

        # calculate option value for individual ladder points
        stress_npv = []
        for ladder_point in ladder_points:
            self.parameters['S_0'] = ladder_point
            self.calc()
            stress_npv.append(self.f - base_npv)

        # set BlackScholes back to its original form
        self.parameters['S_0'] = S_0
        self.calc()

        # return ladder
        return [ladder_points, stress_npv]

    def get_sigma_ladder(self, ladder_points: list[float]) -> list[list[float]]:
        """Return sigma ladder."""

        # get base option value
        self.calc()
        S_0 = self.parameters['S_0']
        base_npv = self.f

        # calculate option value for individual ladder points
        stress_npv = []
        for ladder_point in ladder_points:
            self.parameters['S_0'] = ladder_point
            self.calc()
            stress_npv.append(self.f - base_npv)

        # set BlackScholes back to its original form
        self.parameters['S_0'] = S_0
        self.calc()

        # return ladder
        return [ladder_points, stress_npv]

    def get_r_ladder(self, ladder_points: list[float]) -> list[list[float]]:
        """Return r ladder."""

        # get base option value
        self.calc()
        r = self.parameters['r']
        base_npv = self.f

        # calculate option value for individual ladder points
        stress_npv = []
        for ladder_point in ladder_points:
            self.parameters['r'] = ladder_point
            self.calc()
            stress_npv.append(self.f - base_npv)

        # set BlackScholes back to its original form
        self.parameters['r'] = sigma
        self.calc()

        # return ladder
        return [ladder_points, stress_npv]