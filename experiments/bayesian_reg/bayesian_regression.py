import numpyro as npy
from numpyro import distributions as dist
from jax.random import PRNGKey, split
from numpyro.handlers import trace
from jax import numpy as jnp
from src.utils.trace_dataset import sample_many_traces, serialize_traces
import numpy as np
from scipy.stats import multivariate_normal, norm

def bayesian_linear_regression(key, lk_sigma, prior_sigma, x):
  ks = split(key, 3)
  x1 = npy.sample("x1", dist.Normal(0, prior_sigma), rng_key=ks[0])
  x2 = npy.sample("x2", dist.Normal(0, prior_sigma), rng_key=ks[1])
  
  x_ = x1 * x + x2
  
  obs = dist.Normal(x_, lk_sigma).sample(key=ks[2])[:,None]
  
  npy.deterministic("obs", obs)
  
  return obs

def analytical_covariance(x, sigma, alpha):
    cov1 = np.identity(2)*(sigma/alpha)**2
    x0_sum = len(x)
    x1_sum = np.sum(x)
    x2_sum = np.sum(x**2)
    cov2 = np.array([
        [x0_sum, x1_sum],
        [x1_sum, x2_sum],
    ])
    cov = (1./sigma**2)*(cov1+cov2)
    cov = np.linalg.inv(cov)
    return cov

# Analytical expression for the posterior mean
def analytical_mean(x, y, sigma, alpha):
    xy_sum = np.sum(x*y)
    y_sum = np.sum(y)
    vec = np.array([y_sum, xy_sum])
    cov = analytical_covariance(x, sigma, alpha)
    mu = np.matmul(cov, vec)/sigma**2
    return mu
  
# Distribution for 1D marginal posteriors
def get_distribution(plt, mu, sig, N=200):
    xlim = plt.gca().get_xlim()
    x = np.linspace(xlim[0], xlim[1], N)
    rv = norm(mu, sig)
    f = rv.pdf(x)
    return x, f

# Contours for 2D posterior distributions
def get_contours(plt, mu, cov, N=200):
    xlim = plt.gca().get_xlim(); ylim = plt.gca().get_ylim()
    X = np.linspace(xlim[0], xlim[1], N)
    Y = np.linspace(ylim[0], ylim[1], N)
    X, Y = np.meshgrid(X, Y); pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    return X, Y, Z

def plot_posterior(plt, mu, cov, proposal_samples,):
    plt.scatter([sample['x2'] for sample in proposal_samples], [sample['x1'] for sample in proposal_samples], alpha=0.5, s=5., label="proposal", color='red')
    if mu is not None and cov is not None:
        X, Y, Z = get_contours(plt, mu, cov)
        plt.contourf(X, Y, Z, alpha=0.5, cmap='viridis')
        plt.colorbar(label="Posterior density")
    plt.xlabel('bias')
    plt.ylabel('slope')
  
  

if __name__ == "__main__":
  tr = trace(bayesian_linear_regression).get_trace(PRNGKey(0), 0.1, 10.0, jnp.arange(6.0)+1)

  sampler_ = lambda key: bayesian_linear_regression(key, 0.1, 10.0, jnp.arange(6.0)+1)
  traces, default_trace, means_and_stds = sample_many_traces(sampler_, PRNGKey(0), 1000000, True, max_num_variables=100, max_traces_for_default_trace=10000)

  serialize_traces(traces[:-1000], "experiments/bayesian_regression/data/train_1MM.pkl")
  serialize_traces(traces[-1000:], "experiments/bayesian_regression/data/test_1MM.pkl")
  serialize_traces(means_and_stds, "experiments/bayesian_regression/data/means_and_stds.pkl")
    
  from IPython import embed; embed(using=False)