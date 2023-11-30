import sys
import argparse
import pathlib
import pickle

import csv
import numpy as np
import pandas as pd
import jax.numpy as jnp

import jax
from archdfa import ARCHDFA

# ensure that the path where we will save model fits exists
save_path = pathlib.Path('application')
if not save_path.exists():
    save_path.mkdir(parents=True)


def get_rng_key(data1, data2, chain=None):
    '''
    Get the RNG key to use for real data fitting
    '''
    # seeds from random.org
    # organized by model 
    seeds = {
        'COVIDhub_4_week_ensemble': 455472,
        'COVIDhub_baseline': 100818,
        'CU_select': 270992, 
        'GT_DeepCOVID': 376966, 
        'JHUAPL_Bucky': 875341
    } 
    if data2 == 'COVIDhub_baseline':
        data_n = 0
    elif data2 == 'CU_select':
        data_n = 1
    elif data2 == 'GT_DeepCOVID':
        data_n = 2
    elif data2 == 'JHUAPL_Bucky':
        data_n = 3
    else:
        data_n = 4
    
    keys = jax.random.split(jax.random.PRNGKey(seeds[data1]),
                            num=1000)
    if chain is not None:
        keys_chain = jax.random.split(jax.random.PRNGKey(keys[data_n][0]),
                                num=1000)
        return keys_chain[chain]
    else:
        return keys[data_n]


def run_realdata(data1, data2, val, chain=None):
    '''
    Fit DFA model and save outputs for one simulation replicate
    '''

    if chain is None:
        samples_file = save_path / f'fits/fits_{data1}_{data2}_{val}.pkl'
        summary_file = save_path / f'fits/summary_{data1}_{data2}_{val}.txt'
        if samples_file.exists() and samples_file.stat().st_size > 0:
            print(f'model fit file already exists; skipping {data1} {data2} {val}')
            return
    elif chain is not None:
        samples_file = save_path / f'fits/fits_{data1}_{data2}_{chain}_{val}.pkl'
        summary_file = save_path / f'fits/summary_{data1}_{data2}_{chain}_{val}.txt'
        if samples_file.exists() and samples_file.stat().st_size > 0:
            print(f'model fit file already exists; skipping {data1} {data2} {chain} {val}')
            return

        
    # load simulated data

    def loaddata(data1, data2, val):
        #mypath = save_path / f'real_data/diff_{data1}_{data2}.csv'
        mypath = save_path / f'real_data/diff_{data1}_{data2}.csv'
        df = pd.read_csv(mypath)
        lix = pd.MultiIndex.from_product([np.unique(df.location), np.unique(df.relative_horizon)])
        df_pivot = (df.pivot_table(val, 'reference_date', ['location', 'relative_horizon'], aggfunc='first')).reindex(lix, axis=1)
        mydf = jnp.array(df_pivot.groupby(level=0, axis=1).agg(lambda x: [*x.values]).to_numpy().tolist())
        return mydf

    obs =loaddata(data1, data2, val)      

    # instantiate and fit model
    dfa_model = ARCHDFA(num_timesteps=obs.shape[0], num_series=obs.shape[1], 
                        num_horizons=obs.shape[2],num_factors=5, p=1, q=1, 
                        sigma_factors_model='AR', 
                        loadings_constraint='simplex')

    if chain is None:
        mcmc_samples = dfa_model.fit(y=obs, 
                                    rng_key = get_rng_key(data1, data2),
                                    num_warmup=10000, num_samples=10000, num_chains=5)
    elif chain is not None:
        mcmc_samples = dfa_model.fit(y=obs, 
                                    rng_key = get_rng_key(data1, data2, chain),
                                    num_warmup=10000, num_samples=10000, num_chains=1)

    # save samples and mcmc summary
    with open(samples_file, 'wb') as f:
        pickle.dump(
            {k: mcmc_samples[k] for k in ['intercept', 'phi', 'alpha', 'ARVar_mu', 
                                        'beta0', 'beta1', 'log_sigma_eta', 'Psi_a', 'Psi_b', 
                                        'h_rho', 'log_sigma_eps_t', 'sigma_eps_l', 
                                        'sigma_nu', 'factor_loadings', 'sigma_zeta']},
            f)

    orig_stdout = sys.stdout
    with open(summary_file, 'w') as f:
        sys.stdout = f
        dfa_model.mcmc.print_summary()
    sys.stdout = orig_stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run estimation for application study, one data1, data2, and val index')
    
    parser.add_argument('--data1',
                        help='data1',
                        choices=['COVIDhub_4_week_ensemble', 'COVIDhub_baseline', 'CU_select', 'GT_DeepCOVID', 'JHUAPL_Bucky', 'USC_SI_kJalpha'],
                        default='COVIDhub_4_week_ensemble')
    parser.add_argument('--data2',
                        help='data2',
                        choices=['COVIDhub_baseline', 'CU_select', 'GT_DeepCOVID', 'JHUAPL_Bucky', 'USC_SI_kJalpha'],
                        default='COVIDhub_baseline')
    parser.add_argument('--val', 
                        help='variable',
                        choices=['diff','diff_pop'],
                        default='diff')
    parser.add_argument('--chain', type=int,
                        help='integer index of simulation mcmc chain',
                        choices=list(range(10)),
                        default=None)
    args = parser.parse_args()
    
    run_realdata(**vars(args))
