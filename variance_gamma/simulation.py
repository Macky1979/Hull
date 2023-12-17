import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import gamma

simuls_no = 100000

T = 0.50
vega = 0.50
theta = 0.10
sigma = 0.20
r = 0.0
q = 0.0
S_0 = 100.0

omega = (T / vega) * np.log(1 - theta * vega)

g = gamma.rvs(a=T/vega, loc=0, scale=vega, size=simuls_no)
theta_g = g * theta * np.sqrt(g) * sigma * norm.rvs(size=simuls_no)
S_T = S_0 * np.exp((r - q) * T + theta_g + omega)

'''
plt.hist(x=S_T, bins=1000)
plt.show()
'''
counts, bins = np.histogram(S_T, bins=1000)
plt.stairs(counts, bins)