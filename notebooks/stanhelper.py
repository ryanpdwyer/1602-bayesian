from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from matplotlib.offsetbox import AnchoredText


def HDI_from_MCMC(posterior_samples, credible_mass):
    # Computes highest density interval from a sample of representative values,
    # estimated as the shortest credible interval
    # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0]*nCIs
    for i in range(nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
        HDImin = sorted_points[ciWidth.index(min(ciWidth))]
        HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
    return(HDImin, HDImax)

def Pfi(d, samp, i):
    fc = d['mu_fc'] + samp['dfc'][i]
    Q = samp['Q'][i]
    kc = samp['kc'][i]
    return Pf(d['f'], 
              calc_P_x0(fc*u.Hz, Q, kc*u('N/m'), d['T']*u.K).to('nm^2/Hz').magnitude,
              fc, Q, samp['Pdet'][i]*d['scale']
             )


def plot_all_traces(samp):
    N = len(samp)
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(7.5, 7.5))
    for i, (xname, x) in enumerate(samp.items()):
        for j, (yname, y) in enumerate(samp.items()):
            ax = axes[i][j]
            if i != j:
                ax.plot(x, y, '.', markersize=4, alpha=0.1)
                ax.set_yticklabels([''])
                ax.set_xticklabels([''])
                namey = AnchoredText(yname,2,  frameon=False)
                namex = AnchoredText(xname,4,  frameon=False)
                ax.add_artist(namey)
                ax.add_artist(namex)
            else:
                sns.kdeplot(x, ax=ax)
                ax.set_yticklabels([''])
                ax.set_xticklabels([''])
                name = AnchoredText(xname,2,  frameon=False)
                ax.add_artist(name)
    fig.tight_layout()
    return fig, ax

def fh2data(fh, fmin, fmax, fc, kc, Q, Pdet=None, T=298,
            sigma_fc=5, sigma_kc=5, sigma_Q=10000, sigma_Pdet=1e-6,
            sigma_k=10):
    f_all = fh['x'][:]
    m = (f_all > fmin) & (f_all < fmax)
    f = f_all[m]
    psd_err = (fh['y_std'][:] / fh['y'].attrs['n_avg']**0.5)[m]
    psd = fh['y'][:][m]
    M, _ = fh['PSD_subset'][:].shape

    N = f.size
    
    # Scale data
    psd_scale = psd.mean()
    psd_scaled = psd / psd_scale
    
    if Pdet is None:
        mu_Pdet = np.percentile(psd_scaled, 25)
    else:
        mu_Pdet = Pdet / psd_scale
        
    
    return {'fmin': fmin,
     'fmax': fmax,
     'N': N,
     'M': M,       
     'y': psd_scaled,
     'y_err': psd_err / psd_scale,
     'f': f,
     'scale': psd_scale,
     'mu_fc': fc,
     'mu_kc': kc,
     'mu_Q': Q,
     'mu_Pdet': mu_Pdet,  # scaled
     'sigma_fc': sigma_fc,
     'sigma_kc': sigma_kc,
     'sigma_Q': sigma_Q,
     'sigma_Pdet': sigma_Pdet / psd_scale,
     'sigma_k': sigma_k,
     'T': T,
     }

def fh2data_all(fh, fmin, fmax, fc, kc, Q, Pdet=None, T=298,
            sigma_fc=5, sigma_kc=5, sigma_Q=10000, sigma_Pdet=1e-6,
            sigma_k=10):
    f_all = fh['f_subset'][:]
    m = (f_all > fmin) & (f_all < fmax)
    f = f_all[m]
    psd = fh['PSD_subset'][:][:, m]

    M, N = psd.shape
    
    # Scale data
    psd_scale = psd.mean()
    psd_scaled = psd / psd_scale
    
    
    
    
    if Pdet is None:
        mu_Pdet = np.percentile(psd_scaled, 25)
    else:
        mu_Pdet = Pdet / psd_scale
        
    
    return {'fmin': fmin,
     'fmax': fmax,
     'N': N,
     'y': psd_scaled,
     'f': f,
     'scale': psd_scale,
     'mu_fc': fc,
     'mu_kc': kc,
     'mu_Q': Q,
     'mu_Pdet': mu_Pdet,  # scaled
     'sigma_fc': sigma_fc,
     'sigma_kc': sigma_kc,
     'sigma_Q': sigma_Q,
     'sigma_Pdet': sigma_Pdet / psd_scale,
     'sigma_k': sigma_k,
     'T': T,
     'M': M
     }

def initial(d, k0=0.6, Pdet=None):
    if Pdet is None:
        Pdet = d['sigma_Pdet']*0.1
    return lambda: {'dfc': 0, 'kc': d['mu_fc'],
                    'Q': d['mu_Q'], 'Pdet': Pdet, 'k': k0}


naive_stan = """
data {
  int<lower=0> N;
  vector[N] f;
  vector[N] y;
  vector[N] y_err;
  real mu_fc;
  real mu_kc;
  real mu_Q;
  real mu_Pdet;
  real sigma_fc;
  real sigma_kc;
  real sigma_Q;
  real sigma_Pdet;
  real scale;
  real sigma_k;
  real<lower=0> T;
}
parameters {
  real dfc;
  real<lower=0> kc;
  real<lower=0> Q;
  real<lower=0> Pdet;
  real<lower=0> k;
}
model {
    # Priors on fit parameters
    dfc ~ normal(0, sigma_fc);
    kc ~ normal(mu_kc, sigma_kc);
    Q ~ normal(mu_Q, sigma_Q);
    Pdet ~ normal(mu_Pdet, sigma_Pdet);
    k ~ normal(0, sigma_k);
    

    
    y ~ normal(
    ((2 * 1.381e-5 * T) / (pi() * Q * kc)) / scale * (dfc + mu_fc)^3 ./
            ((f .* f - (dfc + mu_fc)^2) .* (f .* f - (dfc + mu_fc)^2) + f .* f * (dfc + mu_fc)^2 / Q^2)
            + Pdet,
            k * y);
}
"""

exp_stan = """
data {
  int<lower=0> N;
  int<lower=0> M;
  vector[N] f;
  vector[N] y[M];
  real mu_fc;
  real mu_kc;
  real mu_Q;
  real sigma_fc;
  real sigma_kc;
  real sigma_Q;
  real<lower=0> mu_Pdet;
  real scale;
  real<lower=0> T;
}
parameters {
  real dfc;
  real<lower=0> kc;
  real<lower=0> Q;
  real<lower=0> Pdet;
}

model {
    vector[N] P;
    vector[N] beta;
    beta <- 1. ./ (((2 * 1.381e-5 * T) / (pi() * Q * kc)) / scale * (dfc + mu_fc)^3 ./
            ((f .* f - (dfc + mu_fc)^2) .* (f .* f - (dfc + mu_fc)^2) + f .* f * (dfc + mu_fc)^2 / Q^2)
            + Pdet);

    Pdet ~ exponential(inv(mu_Pdet));

    # Priors on fit parameters
    dfc ~ normal(0, sigma_fc);
    kc ~ normal(mu_kc, sigma_kc);
    Q ~ normal(mu_Q, sigma_Q);
    
    for (i in 1:M) {
        y[i] ~ exponential(beta);
    }
}
"""

gamma_code = """
data {
  int<lower=0> N;
  int<lower=0> M;
  vector[N] f;
  vector[N] y;
  vector[N] y_err;
  real mu_fc;
  real mu_kc;
  real mu_Q;
  real mu_Pdet;
  real sigma_fc;
  real sigma_kc;
  real sigma_Q;
  real sigma_Pdet;
  real scale;
  real<lower=0> T;
}
parameters {
  real dfc;
  real<lower=0> kc;
  real<lower=0> Q;
  real<lower=0> Pdet;
}
model {
    # Priors on fit parameters
    dfc ~ normal(0, sigma_fc);
    kc ~ normal(mu_kc, sigma_kc);
    Q ~ normal(mu_Q, sigma_Q);
    Pdet ~ exponential(mu_Pdet);
    

    
    y ~ gamma(M, M ./ (
    ((2 * 1.381e-5 * T) / (pi() * Q * kc)) / scale * (dfc + mu_fc)^3 ./
            ((f .* f - (dfc + mu_fc)^2) .* (f .* f - (dfc + mu_fc)^2) + f .* f * (dfc + mu_fc)^2 / Q^2)
            + Pdet)
            );
}
"""

