import os
import time

import jax.numpy as jnp
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from scipy.stats import multivariate_normal
from numpyro.distributions import ImproperUniform, constraints

# generate an cholesky root of an AR(1) correlation matrix 
# The AR(1) correlation matrix has an explicit formula for the cholesky root 
# in terms of rho. 
# It is a special case of a general formula developed by V.Madar (2016)
def AR1Root(rho,p):
    R = jnp.zeros((p, p))
    R = R.at[0,].set(pow(rho, jnp.arange(p)))
    #R[0,] = pow(rho, jnp.arange(p))
    c = jnp.sqrt(1 - rho**2)
    R2 = c * R[0,:]
    
    for i in range(1,p):
        R = R.at[i, jnp.arange(i,p)].set(R2[jnp.arange(0,p-i)])
        #R[i, jnp.arange(i,p)] = R2[jnp.arange(0,p-i)]
    return jnp.transpose(R)


class ARHDFA():
    '''
    Class representing a dynamic factor analysis model where the latent factors
    follow an autoregressive, conditional heteroskedastic [not yet implemented] model
    '''
    def __init__(self, num_timesteps=None, num_series=None, num_horizons = None,
                 p=1, q=1, intercept_by_series=False, 
                 ar_constraint='[0,1]', sigma_time='AR',
                 ar_var_constraint='[0,1]'):
        '''
        Initialize an ARCHDFA model
        
        Parameters
        ----------
        num_timesteps: integer or None
            Number of time steps. If None, will be set at the time of fitting
        num_series: integer
            Number of observed series, e.g. number of locations in panel data
        num_horizons: integer
            Number of observed horizons; defaults to 1
        p: integer
            Order of autoregressive processes for latent factors; defaults to 1
        q: integer
            Order of autoregressive processes for error of latent factors; defaults to 1
        intercept_by_series: boolean
            If True, estimate a separate intercept for each series. Otherwise,
            estimate a single intercept that is shared across all series.
            Defaults to False.
        ar_constraint: string
            Constraints on autoregressive coefficients. Either '[-1,1]' or '[0,1]'
        ar_var_constraint: string
            Constraints on autoregressive coefficients of variance. Either '[-1,1]' or '[0,1]'
        sigma_time: string
            Constrains on sigma for time, Either 'AR' or 'constant'

        Returns
        -------
        None
        '''
        if num_timesteps is not None and \
                (type(num_timesteps) is not int or num_timesteps <= 0):
            raise ValueError('num_timesteps must be None or a positive integer')
        self.num_timesteps = num_timesteps
        
        if type(num_series) is not int or num_series <= 0:
            raise ValueError('num_series must be a positive integer')
        self.num_series = num_series

        if type(num_horizons) is not int or num_horizons <= 0:
            raise ValueError('num_horizons must be a positive integer')
        self.num_horizons = num_horizons
        
        if type(p) is not int or p <= 0:
            raise ValueError('p must be a positive integer')
        self.p = p

        if type(q) is not int or q <= 0:
            raise ValueError('q must be a positive integer')
        self.q = q

        if type(intercept_by_series) is not bool:
            raise ValueError('intercept_by_series must be a boolean')
        self.intercept_by_series = intercept_by_series
        
        if ar_constraint not in ['[-1,1]', '[0,1]']:
            raise ValueError("ar_constraint must be '[-1,1]' or '[0,1]'")
        self.ar_constraint = ar_constraint

        if ar_var_constraint not in ['[-1,1]', '[0,1]']:
            raise ValueError("ar_var_constraint must be '[-1,1]' or '[0,1]'")
        self.ar_var_constraint = ar_var_constraint

        if sigma_time not in ['AR', 'constant']:
            raise ValueError("sigma_time must be 'AR' or 'constant'")
        self.sigma_time = sigma_time
        
        
    def model(self, y=None, non_nan_inds=None, nan_inds=None, num_nans=None):
        '''
        Auto-regressive dynamic factor analysis model
        
        Parameters
        ----------
        y: array with shape (num_timesteps, num_series, num_horizons)
            Observed data
        '''
        # acquire and/or validate number of time steps and series
        if y is not None:
            if self.num_timesteps is not None and self.num_timesteps != y.shape[0]:
                raise ValueError('if provided, require num_timesteps = y.shape[0]')
            if self.num_series is not None and self.num_series != y.shape[1]:
                raise ValueError('if provided, require num_series = y.shape[1]')
            if self.num_horizons is not None and self.num_horizons != y.shape[2]:
                raise ValueError('if provided, require num_horizons = y.shape[2]')
            self.num_timesteps, self.num_series, self.num_horizons = y.shape
        
        if self.num_timesteps is None or self.num_series is None or self.num_horizons is None:
            raise ValueError('Must provide either y or three of num_timesteps, num_series and num_horizons')
        
        # intercept for observation model, series-specific if requested
        # arranged as row vector for later broadcasting across timesteps
        if self.intercept_by_series:
            intercept_shape = (1, self.num_series)
        else:
            intercept_shape = (1,)

        intercept = numpyro.sample(
            'intercept',
            ImproperUniform(constraints.ordered_vector, (),
            event_shape=intercept_shape))

        # ar coefficients, shared across latent factors
        if self.ar_constraint == '[-1,1]':
            phi_l, phi_u = (-1, 1)
        elif self.ar_constraint == '[0,1]':
            phi_l, phi_u = (0, 1)
        phi = numpyro.sample(
            'phi',
            dist.Uniform(phi_l, phi_u),
            sample_shape=(1, self.p))
        # phi0 = intercept*(1-jnp.sum(phi))
        
        # mean (ARVar_mu) and ar coefficients (alpha) of the variance of the error term in the latent factor analysis
        if self.ar_var_constraint == '[-1,1]':
            alpha_l, alpha_u = (-1, 1)
        elif self.ar_var_constraint == '[0,1]':
            alpha_l, alpha_u = (0, 1)
        ARVar_mu = numpyro.sample(
            'ARVar_mu',
            dist.Normal(0,1),
            sample_shape=(1, 1)
        )
        # alpha = numpyro.sample(
        #     'alpha',
        #     dist.Uniform(alpha_l, alpha_u),
        #     sample_shape=(1, self.q)
        # )
        # alpha0 = ARVar_mu*(1-jnp.sum(alpha))
        
        # initial values for factors, p time steps before time 0
        y_0 = numpyro.sample(
            'y_0',
            dist.Normal(0, 1),
            sample_shape=(self.p, 1))
        
        log_sigma_eta_0 = numpyro.sample(
            'log_sigma_eta_0',
            dist.Normal(1),
            sample_shape=(self.q, 1))
        
        log_sigma_eps = numpyro.sample(
            'log_sigma_eps',
            dist.Normal(1),
            sample_shape=(1,))
        
        # The variance of innovations in the AR process for the variance of the error term in the latent factor analysis
        sigma_nu = numpyro.sample(
            'sigma_nu',
            dist.HalfNormal(1),
            sample_shape=(1,)
        )
        
        #get error variance of latent factors from AR(q) process
        def transition_ARvar(log_sigma_eta_prev, _):
            '''
            ARvar function for use with scan
            
            Parameters
            ----------
            log_sigma_eta_prev: array of shape (q, 1)
                error variance of the q time steps before time t
                the first row contains error variance of factor values for time t-1
            _: ignored, corresponds to integer time step value
            
            Returns
            -------
            log_sigma_eta: array of shape (q, 1)
                updated error variance of the q time steps ending at time t
                the first row contains error variance of factor values for time t
            log_sigma_eta_tt: array of shape (1,1)
                error variance at time t
            '''

            # calculate the mean for the error variance of the factors at time t, shape (1, 1)
            log_sigma_mu_t = (jnp.matmul(alpha, log_sigma_eta_prev) + alpha0)

            # sample variances at time t, shape (1, 1)
            log_sigma_eta = numpyro.sample('log_sigma_eta', dist.Normal(log_sigma_mu_t, sigma_nu))
            # updated variances for the q time steps ending at time t
            # shape (q, 1), first row is for time t
            log_sigma_eta_tt = jnp.concatenate((log_sigma_eta, log_sigma_eta_prev[:-1, :]), axis=0)

            return log_sigma_eta_tt, log_sigma_eta[0, :]
        
        
        timesteps = jnp.arange(self.num_timesteps)
        
        # standard deviation of innovations in AR process for latent factors (num_timepoints, 1)
        if self.sigma_time == 'constant':
            # log_sigma_eta_t = numpyro.sample('log_sigma_eta_t', dist.Normal(ARVar_mu, jnp.sqrt((sigma_nu**2/(1-jnp.sum(alpha**2))))))
            log_sigma_eta_t = numpyro.sample('log_sigma_eta_t', dist.Normal(ARVar_mu, sigma_nu))
        elif self.sigma_time == 'AR':
             _, log_sigma_eta_t = scan(transition_ARvar, log_sigma_eta_0, timesteps)

        # get AR(p) process
        def transition_AR(points_prev, timepoint):
            '''
            AR function for use with scan
            
            Parameters
            ----------
            points_prev: array of shape (p, 1)
                p time steps before time t
                the first row contains values for time t-1
            timepoint: corresponds to integer time step value
            
            Returns
            -------
            mu_t: array of shape (p, 1)
                updated y values of the p time steps ending at time t
                the first row contains y values for time t
            '''
            # calculate the mean for the factors at time t, shape (1, 1)
            m_t = numpyro.sample('m_t', dist.Normal(jnp.matmul(phi, points_prev), jnp.sqrt(jnp.exp(log_sigma_eta_t))))
            mu_t = jnp.concatenate((m_t, points_prev[:-1, :]), axis=0)
            return mu_t, m_t[0,:]
        
        # scan over time steps; latent factors shape is (num_timepoints, 1)
        _, m_t = scan(transition_AR, y_0, timesteps)
        
        # indicate nan values, and replace with 0
        # if y is not None:
        #     y_impute = numpyro.param('y_impute', jnp.zeros(num_nans))
        #     numpyro.deterministic('y_impute_det', y_impute)
        #     y = y.at[nan_inds].set(y_impute)
        
        # numpyro.sample(
        #     'y',
        #     dist.Normal(jnp.reshape(intercept + m_t, (-1,)), jnp.exp(log_sigma_eps)),
        #     obs = y)
        if nan_inds is None:
            numpyro.sample(
                'y',
                dist.Normal(jnp.reshape(intercept + m_t, (-1,)), jnp.sqrt(jnp.exp(log_sigma_eta_t))),
                obs = y)
        else:
            numpyro.sample(
                'y',
                dist.Normal(intercept + m_t[non_nan_inds, 0], jnp.sqrt(jnp.exp(log_sigma_eta_t))),
                obs = y[non_nan_inds])
    
    
    def fit(self, y, rng_key, num_warmup=1000, num_samples=1000, num_chains=1,thinning=1,
                print_summary=False):
            '''
            Fit model using MCMC
            
            Parameters
            ----------
            y: array with shape (num_timesteps, num_series, num_horizons)
                Observed data
            rng_key: random.PRNGKey
                Random number generator key to be used for MCMC sampling
            num_warmup: integer
                Number of warmup steps for the MCMC algorithm
            num_samples: integer
                Number of sampling steps for the MCMC algorithm
            num_chains: integer
                Number of MCMC chains to run
            print_summary: boolean
                If True, print a summary of estimation results
            
            Returns
            -------
            array with samples from the posterior distribution of the model parameters
            '''
            start = time.time()
            sampler = numpyro.infer.NUTS(self.model, max_tree_depth=14)
            #sampler = numpyro.infer.HMC(model=self.model, num_steps=12, adapt_step_size=True)
            self.mcmc = numpyro.infer.MCMC(
                sampler,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                thinning=thinning,
                progress_bar=False if 'NUMPYRO_SPHINXBUILD' in os.environ else True,
            )
            self.mcmc.run(
                rng_key,
                y=y,
                non_nan_inds=jnp.nonzero(~jnp.isnan(y)),
                nan_inds=jnp.nonzero(jnp.isnan(y)),
                num_nans=int(jnp.isnan(y).sum()))
            print('\nMCMC elapsed time:', time.time() - start)
            
            if print_summary:
                self.mcmc.print_summary()
            #return self.mcmc.get_samples()
            return self.mcmc
        
        
    def sample(self, rng_key, condition={}, num_samples=1):
        '''
        Draw a sample from the joint distribution of parameter values and data
        defined by the model, possibly conditioning on a set of fixed values.
        
        Parameters
        ----------
        rng_key: random.PRNGKey
            Random number generator key to be used for sampling
        condition: dictionary
            Optional dictionary of parameter values to hold fixed
        num_samples: integer
            Number of samples to draw. Ignored if condition is provided, in
            which case the number of samples will correspond to the shape of
            the entries in condition.
        
        Returns
        -------
        dictionary of arrays of sampled values
        '''
        if condition == {}:
            predictive = numpyro.infer.Predictive(self.model,
                                                  num_samples=num_samples)
        else:
            predictive = numpyro.infer.Predictive(self.model,
                                                  posterior_samples=condition)
        
        return predictive(rng_key)

    