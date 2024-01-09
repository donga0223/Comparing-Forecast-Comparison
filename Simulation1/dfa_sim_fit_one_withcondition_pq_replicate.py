import sys
import argparse
import pathlib
import pickle
import time

import jax
import numpy as np
from archdfa import ARCHDFA

# ensure that the path where we will save model fits exists
#save_path = pathlib.Path('/work/pi_nick_umass_edu/simulation/ardfa_fits')
save_path = pathlib.Path('simulation/ardfa_ARVar_fits_pq')
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
                'all_vary': 32626,
                'sigma_t_vary': 194420,
                'sigma_l_vary': 217934,
                'sigma_h_vary': 279063,
                'all_constant': 754766
            },
            '1.0': {
                'all_vary': 163488,
                'sigma_t_vary': 485464,
                'sigma_l_vary': 710053,
                'sigma_h_vary': 12280,
                'all_constant': 838587
            },
            '10.0': {
                'all_vary': 747507,
                'sigma_t_vary': 106645,
                'sigma_l_vary': 972520,
                'sigma_h_vary': 6818092,
                'all_constant': 715733
            },
        },
        'large': {
            '0.0': {
                'all_vary': 265114,
                'sigma_t_vary': 209021,
                'sigma_l_vary': 897409,
                'sigma_h_vary': 695232,
                'all_constant': 79225
            },
            '1.0': {
                'all_vary': 914113,
                'sigma_t_vary': 628716,
                'sigma_l_vary': 364756,
                'sigma_h_vary': 757002,
                'all_constant': 951849
            },
            '10.0': {
                'all_vary': 853825,
                'sigma_t_vary': 273283,
                'sigma_l_vary': 217012,
                'sigma_h_vary': 77890,
                'all_constant': 683583
            }
        }  
    }

    # split into 1000 RNG keys, one per replicate at this combination of
    # sample size and theta
    keys = jax.random.split(jax.random.PRNGKey(seeds[sample_size][theta][condition]),
                            num=1000)
    return keys[replicate]


def run_simstudy_replicate(sample_size, theta, p, q, num_factors, condition, replicate):
    '''
    Fit DFA model and save outputs for one simulation replicate
    '''

    samples_file = save_path / f'samples_{sample_size}_{theta}_{p}_{q}_{num_factors}_{condition}_{replicate}.pkl'
    summary_file = save_path / f'summary_{sample_size}_{theta}_{p}_{q}_{num_factors}_{condition}_{replicate}.txt'
    if samples_file.exists() and samples_file.stat().st_size > 0:
        print(f'model fit file already exists; skipping {sample_size} {theta} {p} {q} {num_factors} {condition} {replicate}')
        return

    
    # load simulated data
    with open(f'simulation/ardfa_samples/{sample_size}_{theta}_{condition}.pkl', 'rb') as f:
        obs = pickle.load(f)['y'][replicate, ...]
    num_samples = 10000
    num_chains = 3
    thinning = 10
    
    # instantiate and fit model
    dfa_model = ARCHDFA(num_timesteps=obs.shape[0], num_series=obs.shape[1], 
                        num_horizons=obs.shape[2],num_factors=num_factors, p=p, q=q, 
                        sigma_factors_model='AR', 
                        loadings_constraint='simplex')


    mcmc_samples = dfa_model.fit(y=obs, 
                                rng_key = get_rng_key(sample_size, theta, condition, replicate),
                                num_warmup=10000, num_samples=num_samples, num_chains=num_chains, thinning = thinning)
  
    
    # save samples and mcmc summary
    
    with open(samples_file, 'wb') as f:
        pickle.dump(
            {k: mcmc_samples[k] for k in ['intercept', 'phi', 'alpha', 'ARVar_mu', 
                                        'beta0', 'beta1', 'log_sigma_eta', 'Psi_a', 'Psi_b', 
                                        'h_rho', 'log_sigma_eps_t', 'sigma_eps_l',
                                        'zHmean', 'Omega_tl']},
            f)

    orig_stdout = sys.stdout
    with open(summary_file, 'w') as f:
        sys.stdout = f
        dfa_model.mcmc.print_summary()
    sys.stdout = orig_stdout


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
    parser.add_argument('--p', type=int,
                        help='integer index of chosing p',
                        choices=list(range(1,3)),
                        default=1)
    parser.add_argument('--q', type=int,
                        help='integer index of chosing q',
                        choices=list(range(1,3)),
                        default=1)
    parser.add_argument('--num_factors', type=int,
                        help='integer index of chosing number of factors',
                        choices=list(range(1,10)),
                        default=1)
    parser.add_argument('--condition',
                        help='condition',
                        choices=['all_vary','sigma_t_vary', 'sigma_l_vary', 'sigma_h_vary', 'all_constant'],
                        default=0)
    parser.add_argument('--replicate', type=int,
                        help='integer index of simulation replicate',
                        choices=list(range(100)),
                        default=0)
                        
    args = parser.parse_args()

    run_simstudy_replicate(**vars(args))
        

