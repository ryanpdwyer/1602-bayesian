from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from pymc3 import Model, Normal, HalfNormal, find_MAP, NUTS, sample

np.random.seed(123)

alpha, sigma = 1, 1
beta = [1, 2.5]

size = 100

X1 = np.random.randn(size)
X2 = np.random.randn(size)*0.2

Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

# fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))

# axes[0].scatter(X1, Y)
# axes[1].scatter(X2, Y)
# axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');

# plt.show()

basic_model = Model()

with basic_model:

    alpha = Normal('alpha', mu=0, sd=10)
    # what is shape?
    beta = Normal('beta', mu=0, sd=10, shape=2)
    sigma = HalfNormal('sigma', sd=1)
    mu = alpha + beta[0]*X1 + beta[1]*X2
    # Observed stochastic
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

with basic_model:
    start = find_MAP(fmin=optimize.fmin_powell)

    trace = sample(2000, start=start)

print(start)
print("alpha : {}".format(alpha))
print("beta : {}".format(beta))
print("sigma : {}".format(sigma))

