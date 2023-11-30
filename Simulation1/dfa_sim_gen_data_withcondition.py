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
        '0.0': {
            'all_vary': 531184,
            'sigma_t_vary': 531184,
            'sigma_l_vary': 278703,
            'sigma_h_vary': 278703,
            'all_constant': 278703,
        },
        '1.0': {
            'all_vary': 453558,
            'sigma_t_vary': 453558,
            'sigma_l_vary': 307503,
            'sigma_h_vary': 307503,
            'all_constant': 307503,
            
        },
        '10.0': {
            'all_vary': 905647,
            'sigma_t_vary': 905647,
            'sigma_l_vary': 252784,
            'sigma_h_vary': 252784,
            'all_constant': 252784,
            
        },
    },
    'large': {
        '0.0': {
            'all_vary': 685086,
            'sigma_t_vary': 685086,
            'sigma_l_vary': 568856,
            'sigma_h_vary': 568856,
            'all_constant': 568856,
            
        },
        '1.0': {
            'all_vary': 173016,
            'sigma_t_vary': 173016,
            'sigma_l_vary': 630072,
            'sigma_h_vary': 630072,
            'all_constant': 630072,
            
        },
        '10.0': {
            'all_vary': 637892,
            'sigma_t_vary': 637892,
            'sigma_l_vary': 551803,
            'sigma_h_vary': 551803,
            'all_constant': 551803            
        }
    }  
}

for samp_size in ['small']:
    # Define a DFA model with specified parameter values to use for data generation
    for condition in ['all_vary', 'sigma_t_vary', 'sigma_l_vary', 'sigma_h_vary', 'all_constant']:
        if samp_size == 'small' and condition in ['all_vary', 'sigma_t_vary']:
            n_location = 10
            dfa_model = ARCHDFA(num_timesteps=52, num_series=n_location, num_horizons=4,
                                num_factors=4, p=1, q=1, sigma_factors_model='AR', 
                                loadings_constraint='simplex')
        elif samp_size == 'small' and condition in ['sigma_l_vary', 'sigma_h_vary', 'all_constant']:
            n_location = 10
            dfa_model = ARCHDFA(num_timesteps=52, num_series=n_location, num_horizons=4,
                                num_factors=4, p=1, q=1, sigma_factors_model='constant', 
                                loadings_constraint='simplex')
        elif samp_size == 'large' and condition in ['all_vary', 'sigma_t_vary']:
            n_location = 50
            dfa_model = ARCHDFA(num_timesteps=156, num_series=n_location, num_horizons=14, 
                                num_factors=4, p=1, q=1, sigma_factors_model='AR', 
                                loadings_constraint='simplex')
        elif samp_size == 'large' and condition in ['sigma_l_vary', 'sigma_h_vary', 'all_constant']:
            n_location = 50
            dfa_model = ARCHDFA(num_timesteps=156, num_series=n_location, num_horizons=14, 
                                num_factors=4, p=1, q=1, sigma_factors_model='constant', 
                                loadings_constraint='simplex')

    
        for theta in ['0.0', '1.0', '10.0']:
            # Draw a sample of size 1000 from the model,
            # specifying that the intercept is the given value of theta
            
            if condition == 'all_vary':
                sample = dfa_model.sample(
                    rng_key = jax.random.PRNGKey(seeds[samp_size][theta][condition]),
                    condition={'intercept': jnp.full(shape=(1000,1),
                                                    fill_value=float(theta)),
                                'sigma_eps_l':jnp.full(shape=(1000,n_location,1), 
                                        fill_value=jnp.reshape(jnp.tile(jnp.array([0.5, 1.0, 0.3, 0.4, 0.5, 2.0, 0.6, 0.9, 1.5, 0.5]),1000), (1000,n_location,1))),
                                'Psi_b':jnp.full(shape=(1000,1), fill_value=4.0),
                                })
                with open(save_path / f'{samp_size}_{theta}_{condition}.pkl', 'wb') as f:
                    pickle.dump(sample, f)

            elif condition == 'sigma_t_vary':
                sample = dfa_model.sample(
                    rng_key = jax.random.PRNGKey(seeds[samp_size][theta][condition]),
                    condition={'intercept': jnp.full(shape=(1000,1),
                                                    fill_value=float(theta)),
                                'sigma_eps_l':jnp.full(shape=(1000,n_location,1), fill_value=0.5),
                                'Psi_a':jnp.full(shape=(1000,1), fill_value=1.5),
                                'Psi_b':jnp.zeros((1000,1)),
                                'h_rho':jnp.zeros((1000,1))
                                })
                with open(save_path / f'{samp_size}_{theta}_{condition}.pkl', 'wb') as f:
                    pickle.dump(sample, f)

            elif condition == 'sigma_l_vary':
                sample = dfa_model.sample(
                    rng_key = jax.random.PRNGKey(seeds[samp_size][theta][condition]),
                    condition={'intercept': jnp.full(shape=(1000,1),
                                                    fill_value=float(theta)),
                                'sigma_eps_l':jnp.full(shape=(1000,n_location,1), 
                                        fill_value=jnp.reshape(jnp.tile(jnp.array([0.5, 1.0, 0.3, 0.4, 0.5, 2.0, 0.6, 0.9, 1.5, 0.5]),1000), (1000,n_location,1))),
                                'Psi_a':jnp.full(shape=(1000,1), fill_value=1.5),
                                'Psi_b':jnp.zeros((1000,1)),
                                'h_rho':jnp.zeros((1000,1))
                                })
                with open(save_path / f'{samp_size}_{theta}_{condition}.pkl', 'wb') as f:
                    pickle.dump(sample, f)

            elif condition == 'sigma_h_vary':
                sample = dfa_model.sample(
                    rng_key = jax.random.PRNGKey(seeds[samp_size][theta][condition]),
                    condition={'intercept': jnp.full(shape=(1000,1),
                                                    fill_value=float(theta)),
                                'sigma_eps_l':jnp.full(shape=(1000,n_location,1), fill_value=0.5),
                                'Psi_b':jnp.full(shape=(1000,1), fill_value=4.0),
                                })
                with open(save_path / f'{samp_size}_{theta}_{condition}.pkl', 'wb') as f:
                    pickle.dump(sample, f)

            elif condition == 'all_constant':
                sample = dfa_model.sample(
                    rng_key = jax.random.PRNGKey(seeds[samp_size][theta][condition]),
                    condition={'intercept': jnp.full(shape=(1000,1),
                                                    fill_value=float(theta)),
                                'sigma_eps_l':jnp.full(shape=(1000,n_location,1), fill_value=0.5),
                                'Psi_a':jnp.full(shape=(1000,1), fill_value=1.5),
                                'Psi_b':jnp.zeros((1000,1)),
                                'h_rho':jnp.zeros((1000,1))
                                })
                with open(save_path / f'{samp_size}_{theta}_{condition}.pkl', 'wb') as f:
                    pickle.dump(sample, f)
                    
            

       
