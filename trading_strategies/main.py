import numpy as np
import copy as cp
import matplotlib.pyplot as plt


class Strategy:
    """
    Michal Mackanic 01/07/2023 v1.0

    Calculate and plot payoff of various option trading strategies.

    __init__(instr_defs, underlying_stock_price_rng):
        initiate Instrument class - initiate individual instruments forming
        the trading strategy and calculate their payoffs
        variables:
            instr_defs: dict of lists
                value for each key in the dictionary represents defintion of a
                single instrument forming the trading strategy
                instr_defs[key][0]: str
                    instrument type; allowed values are 'stock', 'call' and
                    'put'
                instr_defs[key][1]: str
                    position type; allowed values are 'short' and 'long'
                instr_defs[key][2]: float
                     specify option strike
                instr_defs[key][3]: float
                    purchase price - S_0 for stock or premium for option
                instr_defs[key][4]: np.array of floats
                    price range of the underlying stock, which is used to
                    evaluate trading strategy
            underlying_stock_price_rng: np.array of floats
                array of underlying stock prices for which the trading strategy
                is evaluated
    plot_payoff():
        plot payoff of the trading strategy and its individual instruments

    example 1:
        # writting a covered call
        instr_defs =\
            {'stock A': ['stock', 'long', None, 45.0],
             'European call on stock A': ['call', 'short', 50.0, 3.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 2:
        # protective put
        instr_defs =\
            {'stock A': ['stock', 'long', None, 45.0],
             'European put on stock A': ['put', 'long', 50.0, 3.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 3:
        # bull spread from call options
        instr_defs =\
            {'European call with K_1': ['call', 'long', 30, 4.0],
             'European call with K_2': ['call', 'short', 50.0, 1.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 4:
        # bull spread from put options
        instr_defs =\
            {'European put with K_1': ['put', 'long', 30, 1.0],
             'European put with K_2': ['put', 'short', 50.0, 4.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 5:
        # bear spread from put options
        instr_defs =\
            {'European put with K_1': ['put', 'long', 50.0, 4.0],
             'European put with K_2': ['put', 'short', 30.0, 1.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 6:
        # bear spread from call options
        instr_defs =\
            {'European call with K_1': ['call', 'long', 50, 1.0],
             'European call with K_2': ['call', 'short', 30.0, 4.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 7:
        # box spread
        instr_defs =\
            {'European call with K_1': ['call', 'long', 30, 4.0],
             'European call with K_2': ['call', 'short', 50.0, 1.0],
             'European put with K_1': ['put', 'long', 50.0, 4.0],
             'European put with K_2': ['put', 'short', 30.0, 1.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 8:
        # butterfly spread from call options
        instr_defs =\
            {'European call with K_1': ['call', 'long', 30.0, 5.5],
             'European call with K_2a': ['call', 'short', 50.0, 3.0],
             'European call with K_2b': ['call', 'short', 50.0, 3.0],
             'European call with K_3': ['call', 'long', 70.0, 1.5]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 9:
        # butterfly spread from put options
        instr_defs =\
            {'European put with K_1': ['put', 'long', 30.0, 1.5],
             'European put with K_2a': ['put', 'short', 50.0, 3.0],
             'European put with K_2b': ['put', 'short', 50.0, 3.0],
             'European put with K_3': ['put', 'long', 70.0, 5.5]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 10:
        # bottom straddle
        instr_defs =\
            {'European call': ['call', 'long', 50.0, 5.0],
             'European put': ['put', 'long', 50.0, 5.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 11:
        # top straddle
        instr_defs =\
            {'European call': ['call', 'short', 50.0, 5.0],
             'European put': ['put', 'short', 50.0, 5.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 12:
        # strip
        instr_defs =\
            {'European call': ['call', 'long', 50.0, 5.0],
             'European put a': ['put', 'long', 50.0, 5.0],
             'European put b': ['put', 'long', 50.0, 5.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 13:
        # strap
        instr_defs =\
            {'European call a': ['call', 'long', 50.0, 5.0],
             'European call b': ['call', 'long', 50.0, 5.0],
             'European put': ['put', 'long', 50.0, 5.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 14:
        # bottom strangle
        instr_defs =\
            {'European call': ['call', 'long', 70.0, 5.0],
             'European put': ['put', 'long', 30.0, 5.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()

    example 14:
        # top strangle
        instr_defs =\
            {'European call': ['call', 'short', 70.0, 5.0],
             'European put': ['put', 'short', 30.0, 5.0]}
        underlying_stock_price_rng = np.linspace(0.00, 100.00, 1000)
        strategy = Strategy(instr_defs=instr_defs, underlying_stock_price_rng=underlying_stock_price_rng)
        strategy.plot_payoff()
    """

    def __init__(self,
                 instr_defs,
                 underlying_stock_price_rng=np.linspace(0.0, 0.1, 1000)):

        # store price range of the underlying stock for which payoff is calculated
        self.underlying_stock_price_rng = cp.deepcopy(underlying_stock_price_rng)

        # initiate inditivual instruments forming the trading strategy and
        # determine their payoff
        self.instruments = {}
        for key in instr_defs.keys():
            instr_defs[key].append(cp.deepcopy(self.underlying_stock_price_rng))
            self.instruments[key] = Instrument(instr_defs[key])
            self.instruments[key].calc_payoff()

        # calculate payoff of the trading strategy
        self.tot_payoff = None
        for key in self.instruments.keys():
            if (self.tot_payoff is None):
                self.tot_payoff = cp.deepcopy(self.instruments[key].payoff)
            else:
                self.tot_payoff += cp.deepcopy(self.instruments[key].payoff)

    def plot_payoff(self):

        # plot total strategy payoff
        plt.errorbar(self.underlying_stock_price_rng,
                     self.tot_payoff,
                     linestyle='solid',
                     linewidth=2.0,
                     color='red',
                     label='strategy payoff')

        # plot payoff of individual instruments forming the strategy
        for key in self.instruments.keys():
            plt.errorbar(self.underlying_stock_price_rng,
                         self.instruments[key].payoff,
                         linestyle='dotted',
                         linewidth=1.0,
                         label=key)

        # place legend
        plt.legend(loc='best', fontsize='x-small')

        # label axis
        plt.xlabel('$S_T$ (underlying stock price at maturity)')
        plt.ylabel('payoff')

        # show plot
        plt.show()


class Instrument:
    """
    Michal Mackanic 01/07/2023 v1.0

    Define a stock / European option and project its payoff at maturity. The
    class is the basic building block of Strategy class.

    __init__(instr_def):
        initiate Instrument class
        variables:
            instr_def: list
                instr_def[0]: str
                    instrument type; allowed values are 'stock', 'call' and
                    'put'
                instr_def[1]: str
                    position type; allowed values are 'short' and 'long'
                instr_def[2]: float
                     specify option strike
                instr_def[3]: float
                    purchase price - S_0 for stock or premium for option
                instr_def[4]: np.array of floats
                    price range of the underlying stock, which is used to
                    evaluate trading strategy
    calc_payoff():
        calculate instrument payoff at maturity for assumed price range
        of the underlying stock

    example 1:
        instrument = Instrument(['stock', 'short', None, 50, np.linspace(0.00, 100.00, 100)])
        instrument.calc_payoff()
        print(instrument.payoff)

    example 2:
        instrument = Instrument(['call', 'long', 50, 5, np.linspace(0.00, 100.00, 100)])
        instrument.calc_payoff()
        print(instrument.payoff)
    """

    def __init__(self, instr_def):

        # check instrument type
        if (instr_def[0] not in ['stock', 'call', 'put']):
            ValueError("'" + instr_def[0] + "' is not a supported " +\
            "instrument type! Allowed values are 'stock', 'call' and 'put'.")

        # check short / long position
        if (instr_def[1] not in ['short', 'long']):
            ValueError("'" + instr_def[1] + "' is not a supported " +\
            "position type! Allowed values are 'short' and 'long'.")

        # store instrument type
        self.instr_tp = instr_def[0]

        # store position type
        self.position_tp = instr_def[1]

        # store strike
        if (self.instr_tp == 'stock'):
            self.K = None
        else:
            self.K = instr_def[2]

        # specify purchase price of stock or option premium at time t = 0
        self.purchase_price = instr_def[3]

        # range to be used for underlying stock
        self.underlying_stock_price_rng = instr_def[4]

    def calc_payoff(self):

        # stock payoff
        if (self.instr_tp == 'stock'):
            S_0 = np.array([self.purchase_price] * len(self.underlying_stock_price_rng))
            self.payoff = cp.deepcopy(self.underlying_stock_price_rng - S_0)

        # call payoff
        elif (self.instr_tp == 'call'):
            K = np.array([self.K] * len(self.underlying_stock_price_rng))
            opt_premium = np.array([self.purchase_price] * len(self.underlying_stock_price_rng))
            self.payoff = cp.deepcopy(np.maximum(self.underlying_stock_price_rng - K, 0.0) - opt_premium)

        # put payoff
        elif (self.instr_tp == 'put'):
            K = np.array([self.K] * len(self.underlying_stock_price_rng))
            opt_premium = np.array([self.purchase_price] * len(self.underlying_stock_price_rng))
            self.payoff = cp.deepcopy(np.maximum(K - self.underlying_stock_price_rng, 0.0) - opt_premium)

        # correction for short position
        if (self.position_tp == 'short'):
            self.payoff *= -1.0