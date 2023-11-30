# Run with archdfa as working directory

import pathlib
import pickle

import jax
import jax.numpy as jnp

from archdfa_gendata import ARCHDFA

save_path = pathlib.Path('simulation/ardfa_samples')
if not save_path.exists():
    save_path.mkdir(parents=True)

# seeds from random.org
# organized by sample size and value of theta
seeds = {
    'small': {
        '0.0': 339114,
        '1.0': 454293,
        '10.0': 634110
    },
    'large': {
        '0.0': 822056,
        '1.0': 991187,
        '10.0': 754016
    } 
}

for samp_size in ['small', 'large']:
    # Define a DFA model with specified parameter values to use for data generation
    if samp_size == 'small':
        dfa_model = ARCHDFA(num_timesteps=52, num_series=10, num_horizons=4,
                            num_factors=4, p=1, q=1, sigma_factors_model='AR', 
                            loadings_constraint='simplex')
    else:
        dfa_model = ARCHDFA(num_timesteps=156, num_series=50, num_horizons=14, 
                            num_factors=4, p=1, q=1, sigma_factors_model='AR', 
                            loadings_constraint='simplex')

    
    for theta in ['0.0', '1.0', '10.0']:
        # Draw a sample of size 1000 from the model,
        # specifying that the intercept is the given value of theta
        sample = dfa_model.sample(
            rng_key = jax.random.PRNGKey(seeds[samp_size][theta]),
            condition={'intercept': jnp.full(shape=(1000,1),
                                             fill_value=float(theta))})
        with open(save_path / f'{samp_size}_{theta}.pkl', 'wb') as f:
            pickle.dump(sample, f)
