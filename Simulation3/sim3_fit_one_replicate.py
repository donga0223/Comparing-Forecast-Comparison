import sys
import argparse
import pathlib
import pickle

import jax
import numpy as np
from io import StringIO
from arhdfa import ARHDFA

# ensure that the path where we will save model fits exists
save_path = pathlib.Path('Simulation3/arhdfa_fits')
if not save_path.exists():
    save_path.mkdir(parents=True)


def get_rng_key(sample_size, theta, condition, replicate):
    '''
    Get the RNG key to use for one simulation replicate
    '''
    # seeds from random.org
    # organized by sample size and value of theta
    seeds = {
        'small': {
            '0.0': {
                'sigma_t_vary_random': 88059,
                'sigma_t_vary_fix': 50595,
                'all_constant_random': 169446,
                'all_constant_fix': 193813,
            },
            '1.0': {
                'sigma_t_vary_random': 209787,
                'sigma_t_vary_fix': 722016,
                'all_constant_random': 494659,
                'all_constant_fix': 838800,       
            },
            '10.0': {
                'sigma_t_vary_random': 554549,
                'sigma_t_vary_fix': 580725,
                'all_constant_random': 424249,
                'all_constant_fix': 117557,        
            },
        },
    }

    # split into 1000 RNG keys, one per replicate at this combination of
    # sample size and theta
    keys = jax.random.split(jax.random.PRNGKey(seeds[sample_size][theta][condition]),
                            num=1000)
    return keys[replicate]


def run_simstudy_replicate(sample_size, theta, condition, sigmafactor, replicate):
    '''
    Fit DFA model and save outputs for one simulation replicate
    '''

    result_file = save_path / f'summary_{sample_size}_{theta}_{condition}_{sigmafactor}_{replicate}.pkl'
    
    if result_file.exists() and result_file.stat().st_size > 0:
        print(f'model fit file already exists; skipping {sample_size} {theta} {condition} {sigmafactor} {replicate}')
        return

    # load simulated data
    with open(f'Simulation3/ardfa_samples/{sample_size}_{theta}_{condition}.pkl', 'rb') as f:
        obs = pickle.load(f)['y'][replicate, ...]
    
    # instantiate and fit model
    dfa_model = ARHDFA(num_timesteps=obs.shape[0], num_series=obs.shape[1], 
                        num_horizons=obs.shape[2],num_factors=1, p=1, q=1, 
                        sigma_factors_model=sigmafactor, 
                        loadings_constraint='simplex')


    mcmc_samples = dfa_model.fit(y=obs, 
                                rng_key = get_rng_key(sample_size, theta, condition, replicate),
                                num_warmup=10000, num_samples=10000, num_chains=3)
  
    # save samples and mcmc summary
    intercept_l_95 = np.percentile(mcmc_samples['intercept'],2.5)
    intercept_u_95 = np.percentile(mcmc_samples['intercept'],97.5)

    # Save the summary to a string
    orig_stdout = sys.stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    dfa_model.mcmc.print_summary()
    sys.stdout = orig_stdout

    # Get the captured output as a string
    mcmc_summary = captured_output.getvalue()

    # Close the StringIO object after obtaining its value
    captured_output.close()

    result_dict = {
    'intercept_l_95': intercept_l_95,
    'intercept_u_95': intercept_u_95,
    'summary': mcmc_summary,
    }

    # Save the dictionary as a pickle file
    with open(result_file, 'wb') as f:
        pickle.dump(result_dict, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run estimation for simulation study, one sample size, theta, and replicate index')
    
    parser.add_argument('--sample_size',
                        help='sample size',
                        choices=['small', 'large'],
                        default='small')
    parser.add_argument('--theta',
                        help='parameter value for theta, the mean loss difference',
                        choices=['0.0', '1.0', '10.0'],
                        default='0.0')
    parser.add_argument('--condition',
                        help='condition',
                        choices=['sigma_t_vary_random', 'sigma_t_vary_fix', 'all_constant_random', 'all_constant_fix'],
                        default='all_constant')
    parser.add_argument('--sigmafactor',
                        help='condition',
                        choices=['constant', 'AR'],
                        default='constant')
    parser.add_argument('--replicate', type=int,
                        help='integer index of simulation replicate',
                        choices=list(range(1000)),
                        default=0)
                        
    args = parser.parse_args()

    run_simstudy_replicate(**vars(args))
        

