import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm


class BrownianMotion:
    """
    Michal Mackanic 07/08/2023 v1.0

    This class represents a node on a binomial tree.

    __init__(self, S_0, mu, sigma, T, steps_no, paths_no=1):
        store variables describing Monte-Carlo simulation of Brownian motion
        variables:
            S_0: float
                stock price at time t = 0
            mu: float
                drift in Brownian motion
            sigma: float
                standard deviation in Brownian motion
            T: float
                simulation period in years
            steps_no: int
                number of steps within Monte-Carlo simulation
            paths_no: int
                number of generated paths
    generate(self, seed=None):
        generate paths of Monte-Carlo
        variables:
            seed: int
                seed for random variables
    plot(self):
        plot up to 100 generated paths
        variables:
    calc_exp_return(self):
        calculate expected return using geometric and aritmetic average

    example:
        S_0 = 100
        mu = 0.05
        sigma = 0.20
        T = 1
        steps_no = T * 12
        paths_no = 1000000 # only the first 100 paths are ploted
        mc = BrownianMotion(S_0=S_0, mu=mu, sigma=sigma, T=T, steps_no=steps_no, paths_no=paths_no)
        mc.generate(seed=None)
        mc.plot()
        mc.calc_exp_return()
    """

    def __init__(self, S_0, mu, sigma, T, steps_no, paths_no=1):

        # store variables
        self.S_0 = S_0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.steps_no = steps_no
        self.dt = self.T / self.steps_no
        self.paths_no = paths_no

        # calculate times in the simulation path
        self.t = np.cumsum([0] + [self.dt] * self.steps_no)

    def generate(self, seed=None):

        # fix seed if required
        if (seed is not None):
            np.random.seed(seed=seed)

        # generate listw of random variables drawn from N(0, 1)
        u = norm.rvs(size=[self.paths_no, self.steps_no])

        # calculate stock prices
        dS_aux1 = np.array([[0] * self.paths_no], dtype=np.float64).T
        dS_aux2 = self.mu * self.dt + self.sigma * np.sqrt(self.dt) * u
        dS = np.append(dS_aux1, dS_aux2, axis=1)
        dS += 1
        dS = np.cumprod(dS, axis=1)
        dS *= self.S_0
        self.S_t = dS

    def plot(self):

        # plot stock price path
        for path_idx in range(min(self.paths_no, 100)):
            plt.plot(self.t,
                     self.S_t[path_idx],
                     linestyle='solid',
                     linewidth=0.5,
                     color='blue')

        # place legend
        plt.title('Simulated paths of Brownian motion')

        # label axis
        plt.xlabel('$t$')
        plt.ylabel('$S_t$')

        # show plot
        plt.show()


    def calc_exp_return(self):

        # calculate arithmetic average
        self.arithmetic_avg = self.mu  # exact value of arithmetic average
        self.arithmetic_avg_est = np.mean(self.S_t[:, 1:] / self.S_t[:, 0:-1] - 1) / self.dt  # estimate of arithmetic expected return based on averaging of individual steps

        # calculate geometric average
        self.geometric_avg = self.mu - self.sigma ** 2 / 2  # exact value of geometic expected return
        self.geometric_avg_est_1 = np.mean(np.log(self.S_t[:, -1] / self.S_0) * (1 / self.T))  # estimated geometric expected return based on comparison of initial and final stock prices
        self.geometric_avg_est_2 = np.mean(np.log(self.S_t[:, 1:]) - np.log(self.S_t[:, 0:-1])) / self.dt  # estimated geometric expected return based on averaging individual steps

        # print results
        print('ARITHMETIC VS. GEOMETRIC AVERAGE')
        print('true arithmetic average:         ' + '{:10.8f}'.format(self.arithmetic_avg))
        print('estimated arithmetic average:    ' + '{:10.8f}'.format(self.arithmetic_avg_est))
        print('true geometric average:          ' + '{:10.8f}'.format(self.geometric_avg))
        print('1st estimated geometric average: ' + '{:10.8f}'.format(self.geometric_avg_est_1))
        print('2nd estimated geometric average: ' + '{:10.8f}'.format(self.geometric_avg_est_2))