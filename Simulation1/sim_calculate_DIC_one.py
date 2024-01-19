import sys
import argparse
import pathlib
import pickle
import time

import jax
import numpy as np
import pandas as pd
from archdfa import ARCHDFA
from archdfa import log_likelihoods
from archdfa import log_likelihoods_bar
#ps=[1,2,3], qs = [1,2,3], num_factors = [2,3,4,5,6], num_samples=10000, num_chains=3, thinning=10
#ps=[1,2], qs = [1], num_factors = [3], num_samples=100, num_chains=3, thinning=10):
def choose_best_model(sample_size, theta, condition, ps=[1,2,3], qs = [1,2,3], num_factors = [2,3,4,5,6], num_samples=10000, num_chains=3, thinning=10):
    records = []
    save_path = pathlib.Path('simulation/ardfa_ARVar_fits_pq')
    for replicate in range(10):
        # load simulated data
        with open(f'simulation/ardfa_samples/{sample_size}_{theta}_{condition}.pkl', 'rb') as f:
            obs = pickle.load(f)['y'][replicate, ...]
        
        for p in ps:
            print(p)
            for q in qs:
                print(q)
                for num_factor in num_factors:
                    print(num_factor)
                    #load mcmc sample
                    name = f'{sample_size}_{theta}_{p}_{q}_{num_factor}_{condition}_{replicate}'
                    samples_file = save_path / f'samples_{name}.pkl'
                    if samples_file.exists() and samples_file.stat().st_size > 0:
                        with open(samples_file, 'rb') as f:
                            mcmc_samples = pickle.load(f)

                        ##calculate DIC 
                        start = time.time() 
                        intercept = mcmc_samples['intercept']
                        zHmean = mcmc_samples['zHmean']
                        Omega_tl = mcmc_samples['Omega_tl']
                        llhood = log_likelihoods1(intercept, zHmean, Omega_tl, obs, num_samples, num_chains, thinning)
                        
                        D_bar = np.mean(-2*np.sum(llhood, axis = (1,2)))

                        post_means_est = {
                            param: np.mean(mcmc_samples[param], axis=0)[np.newaxis, ...] \
                                for param in ['intercept', 'zHmean', 'Omega_tl']
                        }

                        D_theta_bar = log_likelihoods_bar1(post_means_est, obs)
                        n_par_eff_est = D_bar - D_theta_bar
                        DIC_est = n_par_eff_est + D_bar
                        running_time = time.time() - start

                        records.append({
                            'sample_size': sample_size,
                            'theta': theta,
                            'condition': condition,
                            'replicate': replicate,
                            'p': p,
                            'q': q,
                            'num_factor': num_factor,
                            'DIC': DIC_est,
                            'running_time': running_time
                        })
    df_records = pd.DataFrame(records)

    # Now you can use to_string on the DataFrame
    with open(f'simulation/DIC/DIC_{sample_size}_{theta}_{condition}.txt', 'w') as f:
        f.write(df_records.to_string(index=False))
    
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
                        choices=['all_vary','sigma_t_vary', 'sigma_l_vary', 'sigma_h_vary', 'all_constant'],
                        default=0)
    args = parser.parse_args()

    choose_best_model(**vars(args))
        
