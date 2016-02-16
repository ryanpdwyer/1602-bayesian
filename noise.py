#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import optimize
import sigutils
from pymc3 import Model, Normal, HalfNormal, find_MAP, NUTS, sample


# Assume:
# y ~ H*x + ε
# H = low pass filter with gain k, response time τ
# Modeling options: We *know* k, τ, infer x.
# We have reasonably strong priors on k, τ, perhaps a raw impulse response,
# plus some experimental data. We have *some* idea of ε, but not much.

fs = 1e3
nyq = fs/2
T = 0.5
dt = 1/fs
N = int(round(T*fs))+1

t = (np.arange(N) - (N-1)/2)*dt

tau_filt = 0.01
f_filt = 1/(2*np.pi*tau_filt)
tau = 0.05
eps = 15

y_i_test = np.where(t > 0, 0.1*np.exp(-t/tau_filt), 0)
y_i_test[250] = 0.05



x = np.where(t > 0, (1-np.exp(-t/tau)), 0)


ba = signal.butter(1, f_filt/nyq)

impulse = np.zeros_like(t)
impulse[N//2] = 1000

y_i = signal.lfilter(*ba, x=impulse)
y_i_n = y_i  + np.random.randn(N)*eps
y_x = signal.lfilter(*ba, x=x)
y_x_n = y_x + np.random.randn(N)*eps

basic_model = Model()

impulse_response = lambda t, tau: (np.sign(t)+1)*0.5*np.exp(-t/tau)/tau

with basic_model:
    tau_response = HalfNormal('tau_response', sd=5*tau_filt)
    sigma = HalfNormal('sigma', sd=5*eps)
    mu = (np.sign(t)+1)*0.5*(np.exp(-t/tau_response)/tau_response)
    # Observed stochastic
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=y_i_n)


with basic_model:
    start = find_MAP(start={'tau_response_log': np.log(tau_filt), 'sigma_log': np.log(eps)},
                     fmin=optimize.fmin_powell)

    trace = sample(1000, start=start, tune=250, njobs=4)

print(start)
taus = trace.get_values(tau_response, burn=500)
tau_m = np.mean(taus)
sigmas = trace.get_values(sigma, burn=500)
sigma_m = np.mean(sigmas)
print(u"tau_response : {} ± {}".format(tau_m, np.std(taus, ddof=1)))
print("sigma : {} ± {}".format(sigma_m, np.std(sigmas, ddof=1)))


# fig, (ax1, ax2) = sigutils.magtime_z(*ba, fs=fs)
# fig.show()
# plt.close()

print("done")

plt.plot(t, y_i_n)
plt.plot(t, y_i, 'y-', linewidth=1)
plt.plot(t, impulse_response(t, tau_m), 'm')
plt.fill_between(t, impulse_response(t, tau_m) - sigma_m, impulse_response(t, tau_m)+sigma_m,
                  color='m', alpha=0.4)
plt.xlim(-0.03, 0.15)
plt.show()
