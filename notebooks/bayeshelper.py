from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import norm
from ipywidgets import interactive
import numpy as np

def linear_model(alpha, beta, sigma, N=50, pt=1):
    xlim = (0, 10)
    x = np.linspace(xlim[0], xlim[1], N)
    xi = x[pt]
    mu = alpha + beta * x
    y_pts = np.linspace(mu[pt] - 5*sigma, mu[pt] + 5*sigma, 200)
    y_probs = norm.pdf(y_pts, scale=sigma, loc=mu[pt])
    y = np.random.randn(N)*sigma + mu
    mb = np.polyfit(x, y, 1)
    log_like = np.sum(np.log(norm.pdf((y - mu) / sigma)))
    gs = gridspec.GridSpec(nrows=1, ncols=4)
    fig = plt.figure()
    ax = fig.add_subplot(gs[:-1])
    ax2 = fig.add_subplot(gs[-1])
    fig.subplots_adjust(wspace=0.05)
    ax.plot(x, y, 'bo', markeredgewidth=0)
    ax.plot(x, mu, 'b-')
    ax.plot(x, np.polyval(mb, x), 'g-')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax2.set_xlabel("Probability")
    ax.set_xlim(np.mean(xlim) -0.52*(xlim[1]-xlim[0]), np.mean(xlim) + 0.52*(xlim[1]-xlim[0]))
    ylim = ax.get_ylim()
    xlim_ = ax.get_xlim()
    loglike_sign = '' if log_like >= 0 else '-'
    ax.text(xlim_[0]+(xlim_[1] - xlim_[0])*0.05, ylim[0]+(ylim[1] - ylim[0])*0.8, "$\log L = {}{:.1f}$\n$y_i \\sim N(mx_i + b, \\sigma)$".format(loglike_sign, abs(log_like)),
           size=14)
    ax2.set_ylim(*ylim)
    ax2.set_yticklabels([''])
    ax2.plot(y_probs, y_pts)
    ax2.scatter(norm.pdf(y[pt], scale=sigma, loc=mu[pt]), y[pt], c='', edgecolors='b')
    ax.vlines(x[pt], min(mu[pt], y[pt]),  max(mu[pt], y[pt]), )
    ax2.axhline(mu[pt], color='0.5', linestyle='--')
    ax2.axhline(y[pt], color='0.5', linestyle='--')
    ax2.set_xlim(0, norm.pdf(0)*1.1)
    ax2.set_xticklabels([''])
    return fig, ax