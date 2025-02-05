"""Black-Scholes model."""
# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: redefined-outer-name

import numpy as np
from scipy.stats import norm


class BlackScholes:
    """
    European option pricing model based on Black-Scholes formula.

    Description
    -----------
    Calculate option value and Greeks based on Black-Scholes formula; see chapter
    17 of "Options, Futures, and Other Derivatives, 11e" by John C. Hull for more
    details.

    Parameters
    ----------
    opt_tp : str
        Option type, "call" or "put"
    S0 : float
        Spot price of the underlying. You can either specify S0 or F0
        (not both).
    F0 : float
        Forward price of the underlying. You can either specify S0 or F0
        (not both).
    K : float
        Strike price
    r : float
        Risk-free rate
    q : float
        Dividend yield
    sigma : float
        Volatility
    T : float
        Time to maturity

    Example 1
    ---------
    # Black-Scholes formula based on S0
    opt_tp = "call"
    S0 = 100
    K = 80
    r = 0.05
    q = 0.01
    sigma = 0.20
    T = 1.00

    opt = BlackScholes(opt_tp=opt_tp, S0=S0, K=K, r=r, q=q, sigma=sigma, T=T)
    opt.calc()
    print(f"option price: {opt.f:.3f}")
    opt.calc_greeks()
    for key, value in opt.greeks.items():
        print(f"option {key}: {value:.3f}")

    Example 2
    ---------
    # Symmetry of FX options

    # Put option
    opt_tp = "put"
    fx = 1.10
    K = 1.05
    r = 0.05
    r_f = 0.03
    sigma = 0.20
    T = 1.00
    put = BlackScholes(opt_tp=opt_tp, S0=fx, K=K, r=r, q=r_f, sigma=sigma, T=T)
    put.calc()
    print(f"put option price: {put.f:.3f}")

    # Call option
    opt_tp = "call"
    fx = 1.00
    K = 1.10/1.05
    r = 0.03
    r_f = 0.05
    sigma = 0.20
    T = 1.00
    call = BlackScholes(opt_tp=opt_tp, greeks=greeks, S0=fx, K=K, r=r, q=r_f, sigma=sigma, T=T)
    call.calc()
    print(f"call option price: {put.f:.3f}")
    """

    def __init__(self,
                 **kwargs: dict[str, float | bool]) -> None:
        """
        Initialize Black-Scholes option object.

        Description
        -----------
        Initialize Black-Scholes option object.

        Raises
        ------
        ValueError
            incorrect parameters
        ValueError
            icorrect option type

        Example
        -------
        """
        # Store variables
        self.parameters = dict(kwargs)

        # Check that all parameters were specified
        S0_param_nms = sorted(set(("opt_tp", "S0", "K", "r", "q", "sigma", "T")))
        F0_param_nms = sorted(set(("opt_tp", "F0", "K", "r", "sigma", "T")))
        obj_param_nms = sorted(set(self.parameters.keys()))

        if S0_param_nms == obj_param_nms:
            self.version = "S0"
        elif F0_param_nms == obj_param_nms:
            self.version = "F0"
        else:
            raise ValueError("incorrect parameters")

        # Check option type
        if self.parameters["opt_tp"] not in ["call", "put"]:
            raise ValueError(self.parameters["opt_tp"] + " is not a supported option type")

        # Set up Greeks calculation
        self.greeks = {}

        # Option value
        self.f = None

    def calc(self):
        """
        Calculate option value based on Black-Scholes formula.

        Description
        -----------
        Calculate option value based on Black-Scholes formula. The formula could
        be based on spot value S0 or forward value F0 = S0 * exp((r - q) * T).

        Example
        -------
        # Put FX option
        opt_tp = "put"
        fx = 1.10
        K = 1.05
        r = 0.05
        r_f = 0.03
        sigma = 0.20
        T = 1.00
        put = BlackScholes(opt_tp=opt_tp, S0=fx, K=K, r=r, q=r_f, sigma=sigma, T=T)
        put.calc()
        print(f"option price: {opt.f:.3f}")
        """
        # Black-Scholes formula based on S0
        if self.version == "S0":
            self.calc_S0()

        # Black-Scholes formula based on F0 = S0 * exp((r - q) * T); if we know F0 we do not
        # have to estimate dividend yield q
        else:
            self.calc_F0()

    def calc_S0(self):
        """
        Calculate option value based on Black-Scholes formula using spot value.

        Description
        -----------
        Calculate option value based on Black-Scholes formula using spot value
        # of the underlying S0.

        Example
        -------
        # Black-Scholes formula based on S0
        opt_tp = "call"
        S0 = 100
        K = 80
        r = 0.05
        q = 0.01
        sigma = 0.20
        T = 1.00
        opt = BlackScholes(opt_tp=opt_tp, S0=S0, K=K, r=r, q=q, sigma=sigma, T=T)
        opt.calc()
        print(f"option price: {opt.f:.3f}")
        """
        # Extract parameters
        opt_tp = self.parameters["opt_tp"]
        S0 = self.parameters["S0"]
        K = self.parameters["K"]
        r = self.parameters["r"]
        q = self.parameters["q"]
        sigma = self.parameters["sigma"]
        T = self.parameters["T"]

        # Calculate d1 and d2
        d1 = (np.log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        self.parameters["d1"] = d1
        self.parameters["d2"] = d2

        # Calculate call value
        if opt_tp == "call":

            self.f =\
                S0 * np.exp(-q * T) * norm.cdf(d1) -\
                K * np.exp(-r * T) * norm.cdf(d2)

        # Calculate put value
        else:
            self.f =\
                K * np.exp(-r * T) * norm.cdf(-d2) -\
                S0 * np.exp(-q * T) * norm.cdf(-d1)

    def calc_F0(self):
        """
        Calculate option value based on Black-Scholes formula using forward price.

        Description
        -----------
        Calculate option value based on Black-Scholes formula using forward
        price of the underlying defined as F0 = S0 * exp((r - q) * T).

        Example
        -------
        # Black-Scholes formula based on F0
        opt_tp = "call"
        S0 = 100
        K = 80
        r = 0.05
        q = 0.01
        sigma = 0.20
        T = 1.00
        F0 = S0 * np.exp((r - q) * T)
        opt = BlackScholes(opt_tp=opt_tp, F0=F0, K=K, r=r, sigma=sigma, T=T)
        opt.calc()
        print(f"option price: {opt.f:.3f}")
        """
        # Extract parameters
        opt_tp = self.parameters["opt_tp"]
        F0 = self.parameters["F0"]
        K = self.parameters["K"]
        r = self.parameters["r"]
        sigma = self.parameters["sigma"]
        T = self.parameters["T"]

        # Calculate d1 and d2
        d1 = (np.log(F0 / K) + (sigma**2) * T / 2) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        self.parameters["d1"] = d1
        self.parameters["d2"] = d2

        # Calculate call value
        if opt_tp == "call":

            self.f =\
                np.exp(-r * T) * (F0 * norm.cdf(d1) - K * norm.cdf(d2))

        # Calculate put value
        else:
            self.f =\
                np.exp(-r * T) * (K * norm.cdf(-d2) - F0 * norm.cdf(-d1))

    def calc_first_derivative(self,
                              param_nm: str,
                              step: float = 1e-5) -> float:
        """
        Calculate the first derivative of the option value with respect to
        a parameter.

        Description
        -----------
        Calculate the first derivative of the option value with respect to
        a parameter using the central difference method. The method could be
        used to calculate Greeks delta, theta, vega, and rho.

        Link: https://math.umd.edu/~dlevy/classes/amsc466/lecture-notes/differentiation-chap.pdf

        Parameters
        ----------
        param_nm : str
            Name of the parameter for which the derivative is calculated.
        step : float, optional
            Shift applied to the original parameter value, by default 1e-5.

        Returns
        -------
        float
            First derivative of the option value with respect to the parameter.

        Example
        -------
        # Calculate delta of European call option
        opt_tp = "call"
        S0 = 100
        K = 80
        r = 0.05
        q = 0.01
        sigma = 0.20
        T = 1.00
        opt = BlackScholes(opt_tp=opt_tp, S0=S0, K=K, r=r, q=q, sigma=sigma, T=T)
        delta = opt.calc_first_derivative(param_nm="S0")
        print(f"option delta: {delta:.3f}")
        """
        # Store the current value of the parameter
        param_val = self.parameters[param_nm]

        # Shift the parameter value by a small step a re-calculate the option
        # value
        self.parameters[param_nm] = param_val + step
        self.calc()
        f_up = self.f
        self.parameters[param_nm] = param_val - step
        self.calc()
        f_down = self.f

        # Return the parameter to its original value and re-calculate the option
        # value
        self.parameters[param_nm] = param_val
        self.calc()

        # Calculate the first derivative
        return (f_up - f_down) / (2 * step)

    def calc_second_derivative(self,
                               param_nm: str,
                               step: float = 1e-5) -> float:
        """
        Calculate the second derivative of the option value with respect to
        a parameter.

        Description
        -----------
        Calculate the second derivative of the option value with respect to
        a parameter using the central difference method. The method could be
        used to calculate Greeks gamma and volga.

        Link: https://math.umd.edu/~dlevy/classes/amsc466/lecture-notes/differentiation-chap.pdf

        Parameters
        ----------
        param_nm : str
            Name of the parameter for which the derivative is calculated.
        step : float, optional
            Shift applied to the original parameter value, by default 1e-5.

        Returns
        -------
        float
            Second derivative of the option value with respect to the parameter.

        Example
        -------
        # Calculate gamma of European call option
        opt_tp = "call"
        S0 = 100
        K = 80
        r = 0.05
        q = 0.01
        sigma = 0.20
        T = 1.00
        opt = BlackScholes(opt_tp=opt_tp, S0=S0, K=K, r=r, q=q, sigma=sigma, T=T)
        gamma = opt.calc_second_derivative(param_nm="S0")
        print(f"option gamma: {gamma:.3f}")
        """
        # Store the current value of the parameter
        param_val = self.parameters[param_nm]

        # Shift the parameter value by a small step a re-calculate the option
        # value
        self.parameters[param_nm] = param_val + step
        self.calc()
        f_up = self.f
        self.parameters[param_nm] = param_val - step
        self.calc()
        f_down = self.f

        # Return the parameter to its original value and re-calculate the option
        # value
        self.parameters[param_nm] = param_val
        self.calc()
        f_mid = self.f

        # Calculate the second derivative
        return (f_up + f_down - 2 * f_mid) / step**2

    def calc_cross_derivative(self,
                              param_nm_1: str,
                              param_nm_2: str,
                              step: float = 1e-5) -> float:
        """
        Calculate the cross derivative of the option value with respect to
        two parameters.

        Description
        -----------
        Calculate the cross derivative of the option value with respect to
        two parameters. The method could be used to calculate Greeks vanna.

        Link: https://math.stackexchange.com/questions/2931510/cross-derivatives-using-finite-differences.

        Parameters
        ----------
        param_nm_1 : str
            Name of the first parameter for which the derivative is calculated.
        param_nm_2 : str
            Name of the second parameter for which the derivative is calculated.
        step : float, optional
            Shift applied to the original parameters value, by default 1e-5.

        Returns
        -------
        float
            Cross derivative of the option value with respect to the parameter.

        Example
        -------
        # Calculate vanna of European call option
        opt_tp = "call"
        S0 = 100
        K = 80
        r = 0.05
        q = 0.01
        sigma = 0.20
        T = 1.00
        opt = BlackScholes(opt_tp=opt_tp, S0=S0, K=K, r=r, q=q, sigma=sigma, T=T)
        vanna = opt.calc_cross_derivative(param_nm_1="sigma", param_nm_2="S0")
        print(f"option vanna: {vanna:.3f}")
        """
        # Store the current values of the parameters
        param_val_1 = self.parameters[param_nm_1]
        param_val_2 = self.parameters[param_nm_2]

        # Shift the parameters values by a small step a re-calculate the
        # option value
        self.parameters[param_nm_1] = param_val_1 + step
        self.parameters[param_nm_2] = param_val_2 + step
        self.calc()
        f_up_up = self.f

        self.parameters[param_nm_1] = param_val_1 + step
        self.parameters[param_nm_2] = param_val_2 - step
        self.calc()
        f_up_down = self.f

        self.parameters[param_nm_1] = param_val_1 - step
        self.parameters[param_nm_2] = param_val_2 + step
        self.calc()
        f_down_up = self.f

        self.parameters[param_nm_1] = param_val_1 - step
        self.parameters[param_nm_2] = param_val_2 - step
        self.calc()
        f_down_down = self.f

        # Return the parameters to their original values and re-calculate the
        # option value
        self.parameters[param_nm_1] = param_val_1
        self.parameters[param_nm_2] = param_val_2
        self.calc()

        # Calculate the cross derivative
        return (f_up_up - f_up_down - f_down_up + f_down_down) / (2 * step)**2

    def calc_greeks(self,
                    step: float = 1e-5) -> None:
        """
        Calculate Greeks.

        Description
        -----------
        Calculate Greeks delta, gamma, rho_r, theta, vega, volga, vanna, and rho_q.

        Parameters
        ----------
        step : float, optional
            _description_, by default 1e-5

        Example
        -------
        # Calculate Greeks of European call option
        opt_tp = "call"
        S0 = 100
        K = 80
        r = 0.05
        q = 0.01
        sigma = 0.20
        T = 1.00
        opt = BlackScholes(opt_tp=opt_tp, S0=S0, K=K, r=r, q=q, sigma=sigma, T=T)
        opt.calc_greeks()
        for key, value in opt.greeks.items():
            print(f"option {key}: {value:.3f}")
        """

        if self.version == "S0":

            # Calculate delta
            self.greeks["delta"] =\
                self.calc_first_derivative(param_nm="S0",
                                           step=step)

            # Calculate gamma
            self.greeks["gamma"] =\
                self.calc_second_derivative(param_nm="S0",
                                            step=step)

            # Calculate rho with respect to q
            self.greeks["rho_q"] =\
                self.calc_first_derivative(param_nm="q",
                                           step=step)

        else:

            # Calculate delta
            self.greeks["delta"] =\
                self.calc_first_derivative(param_nm="F0",
                                           step=step)

            # Calculate gamma
            self.greeks["gamma"] =\
                self.calc_second_derivative(param_nm="F0",
                                            step=step)

        # Calculate rho with respect to r
        self.greeks["rho_r"] =\
            self.calc_first_derivative(param_nm="r",
                                       step=step)

        # Calculate theta
        self.greeks["theta"] =\
            self.calc_first_derivative(param_nm="T",
                                       step=step)

        # Calculate vega
        self.greeks["vega"] =\
            self.calc_first_derivative(param_nm="sigma",
                                       step=step)

        # Calculate volga
        self.greeks["volga"] =\
            self.calc_second_derivative(param_nm="sigma",
                                        step=step)

        # Calculate vanna
        self.greeks["vanna"] =\
            self.calc_cross_derivative(param_nm_1="sigma",
                                       param_nm_2="S0",
                                       step=step)
