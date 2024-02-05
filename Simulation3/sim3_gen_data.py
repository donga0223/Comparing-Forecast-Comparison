# Run with arhdfa as working directory

import pathlib
import pickle

import jax
import jax.numpy as jnp

from arhdfa_gendata2 import ARHDFA

save_path = pathlib.Path('Simulation3/ardfa_samples')
if not save_path.exists():
    save_path.mkdir(parents=True)

# seeds from random.org
# organized by sample size and value of theta 'sigma_t_vary_random', 'all_constant_random', 'sigma_t_vary_fix', 'all_constant_fix'
seeds = {
    'small': {
        '0.0': {
            'sigma_t_vary_random': 531184,
            'sigma_t_vary_fix': 531184,
            'all_constant_random': 278703,
            'all_constant_fix': 278703,
        },
        '1.0': {
            'sigma_t_vary_random': 453558,
            'sigma_t_vary_fix': 453558,
            'all_constant_random': 307503,
            'all_constant_fix': 307503,           
        },
        '10.0': {
            'sigma_t_vary_random': 905647,
            'sigma_t_vary_fix': 905647,
            'all_constant_random': 252784,
            'all_constant_fix': 252784,            
        },
    },
}

for samp_size in ['small']:
    # Define a DFA model with specified parameter values to use for data generation
    for condition in ['sigma_t_vary_random', 'all_constant_random', 'sigma_t_vary_fix', 'all_constant_fix']:
        if samp_size == 'small' and condition in ['sigma_t_vary_random', 'sigma_t_vary_fix']:
            n_location = 1
            dfa_model = ARHDFA(num_timesteps=52, num_series=n_location, num_horizons=1,
                                num_factors=1, p=1, q=1, sigma_factors_model='AR', 
                                loadings_constraint='simplex')
        elif samp_size == 'small' and condition in ['all_constant_random', 'all_constant_fix']:
            n_location = 1
            dfa_model = ARHDFA(num_timesteps=52, num_series=n_location, num_horizons=1,
                                num_factors=1, p=1, q=1, sigma_factors_model='constant', 
                                loadings_constraint='simplex')
        
    
        for theta in ['0.0', '1.0', '10.0']:
            # Draw a sample of size 1000 from the model,
            # specifying that the intercept is the given value of theta
            
            if condition in ['sigma_t_vary_random', 'all_constant_random']:
                sample = dfa_model.sample(
                    rng_key = jax.random.PRNGKey(seeds[samp_size][theta][condition]),
                    condition={'intercept': jnp.full(shape=(1000,1),
                                                    fill_value=float(theta))
                                })
                with open(save_path / f'{samp_size}_{theta}_{condition}.pkl', 'wb') as f:
                    pickle.dump(sample, f)

            elif condition in ['sigma_t_vary_fix', 'all_constant_fix']:
                sample = dfa_model.sample(
                    rng_key = jax.random.PRNGKey(seeds[samp_size][theta][condition]),
                    condition={'intercept': jnp.full(shape=(1000,1),
                                                    fill_value=float(theta)),
                                'sigma_eps_l':jnp.full(shape=(1000,n_location,1), fill_value=0.6),
                                'Psi_a':jnp.full(shape=(1000,1), fill_value=1.5),
                                'Psi_b':jnp.zeros((1000,1)),
                                'h_rho':jnp.zeros((1000,1))
                                })
                with open(save_path / f'{samp_size}_{theta}_{condition}.pkl', 'wb') as f:
                    pickle.dump(sample, f)

                   
