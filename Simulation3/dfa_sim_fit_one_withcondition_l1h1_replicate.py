import sys
import argparse
import pathlib
import pickle

import jax
from archdfa import ARCHDFA

# ensure that the path where we will save model fits exists
#save_path = pathlib.Path('/work/pi_nick_umass_edu/simulation/ardfa_fits')
save_path = pathlib.Path('simulation/ardfa_ARVar_fits_l1h1_2')
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

    samples_file = save_path / f'samples_{sample_size}_{theta}_{condition}_{sigmafactor}_{replicate}.pkl'
    summary_file = save_path / f'summary_{sample_size}_{theta}_{condition}_{sigmafactor}_{replicate}.txt'
    if samples_file.exists() and samples_file.stat().st_size > 0:
        print(f'model fit file already exists; skipping {sample_size} {theta} {condition} {sigmafactor} {replicate}')
        return

    
    # load simulated data
    with open(f'simulation/ardfa_samples_l1h1_2/{sample_size}_{theta}_{condition}.pkl', 'rb') as f:
        obs = pickle.load(f)['y'][replicate, ...]
    
    # instantiate and fit model
    dfa_model = ARCHDFA(num_timesteps=obs.shape[0], num_series=obs.shape[1], 
                        num_horizons=obs.shape[2],num_factors=1, p=1, q=1, 
                        sigma_factors_model=sigmafactor, 
                        loadings_constraint='simplex')


    mcmc_samples = dfa_model.fit(y=obs, 
                                rng_key = get_rng_key(sample_size, theta, condition, replicate),
                                num_warmup=10000, num_samples=10000, num_chains=3)
  
    # save samples and mcmc summary
    if sigmafactor == 'AR':
        with open(samples_file, 'wb') as f:
            pickle.dump(
                {k: mcmc_samples[k] for k in ['intercept', 'phi', 'alpha', 'ARVar_mu', 
                                            'beta0', 'beta1', 'log_sigma_eta', 'Psi_a', 'Psi_b', 
                                            'h_rho', 'log_sigma_eps_t', 'sigma_eps_l']},
                f)
        orig_stdout = sys.stdout
        with open(summary_file, 'w') as f:
            sys.stdout = f
            dfa_model.mcmc.print_summary()
        sys.stdout = orig_stdout

    elif sigmafactor == 'constant':
        with open(samples_file, 'wb') as f:
            pickle.dump(
                {k: mcmc_samples[k] for k in ['intercept', 'phi', 'alpha', 'ARVar_mu', 
                                            'log_sigma_eta_t', 'Psi_a', 'Psi_b', 
                                            'h_rho', 'log_sigma_eps_t', 'sigma_eps_l']},
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
        

