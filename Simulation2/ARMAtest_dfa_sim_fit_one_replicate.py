import sys
import argparse
import pathlib
import pickle

import jax.numpy as jnp
import jax
from archdfa import ARCHDFA
from ARMA import ARMA_gen_byvar_acf

# ensure that the path where we will save model fits exists
#save_path = pathlib.Path('/work/pi_nick_umass_edu/simulation/ardfa_fits')
save_path = pathlib.Path('simulation/ardfa_ARMAtest')
if not save_path.exists():
    save_path.mkdir(parents=True)


def get_rng_key(timetype, acf, replicate):
    '''
    Get the RNG key to use for one simulation replicate
    '''
    # seeds from random.org
    # organized by sample size and value of theta

    seeds = {
        'AR':{
            '0.0': 32626,
            '0.1': 194420,
            '0.2': 217934,
            '0.3': 611565,
            '0.4': 788635,
            '0.5': 827114,
            '0.6': 943746, 
            '0.7': 934070,
            '0.8': 627076
            },
        'MA':{
            '0.0': 162511,
            '0.1': 438548, 
            '0.2': 488623,
            '0.3': 896113,
            '0.4': 396397,
            '0.5': 38253,
            '0.6': 537941,
            '0.7': 947525,
            '0.8': 336828
        },
        'MA5_same':{
            '0.0': 744139,
            '0.1': 84766,
            '0.2': 197792,
            '0.3': 95909,
            '0.4': 447525,
            '0.5': 868719,
            '0.6': 960541,
            '0.7': 24513,
            '0.8': 556563
        },
        'MA5_exp':{
            '0.0': 490958,
            '0.1': 44914,
            '0.2': 17240,
            '0.3': 539219,
            '0.4': 173721,
            '0.5': 528320,
            '0.6': 87571,
            '0.7': 533194,
            '0.8': 624167
        }
    }

    

    # split into 1000 RNG keys, one per replicate at this combination of
    # sample size and theta
    keys = jax.random.split(jax.random.PRNGKey(seeds[timetype][acf]),
                            num=1000)
    return keys[replicate]


def run_simstudy_replicate(timetype, acf, sigmafactor, replicate, intercept):
    '''
    Fit DFA model and save outputs for one simulation replicate
    '''

    samples_file = save_path / f'samples_{timetype}_{acf}_{sigmafactor}_{replicate}_{intercept}.pkl'
    summary_file = save_path / f'summary_{timetype}_{acf}_{sigmafactor}_{replicate}_{intercept}.txt'
    if samples_file.exists() and samples_file.stat().st_size > 0:
        print(f'model fit file already exists; skipping {timetype} {acf} {sigmafactor} {replicate} {intercept}')
        return

    
    # generate simulated data
    obs = ARMA_gen_byvar_acf(heterotype = 'nohetero', timeseries_type = timetype, marginal_var = 'True', acf = float(acf), replicate = replicate, sample_size = 52, intercept = intercept)
        
    dfa_model = ARCHDFA(num_timesteps=52, num_series=1, 
                        num_horizons=1,num_factors=1, p=1, q=1, 
                        sigma_factors_model=sigmafactor, 
                        loadings_constraint='simplex')


    mcmc_samples = dfa_model.fit(y=jnp.array(obs), 
                                    rng_key = get_rng_key(timetype, acf, replicate),
                                    num_warmup=10000, num_samples=10000, num_chains=3)



    # save samples and mcmc summary

    with open(samples_file, 'wb') as f:
        pickle.dump(
            {k: mcmc_samples[k] for k in ['intercept']},
            f)

    orig_stdout = sys.stdout
    with open(summary_file, 'w') as f:
        sys.stdout = f
        dfa_model.mcmc.print_summary()
    sys.stdout = orig_stdout


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run estimation for simulation study, one sample size, theta, and replicate index')

    parser.add_argument('--timetype',
                        help='timetype',
                        choices=['AR', 'MA', 'MA5_same', 'MA5_exp'],
                        default='MA')
    parser.add_argument('--acf',
                        help='condition',
                        choices=['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'],
                        default=0)
    parser.add_argument('--sigmafactor',
                        help='sigma_factor_model',
                        choices=['AR', 'constant'],
                        default='AR')
    parser.add_argument('--replicate', type=int,
                        help='integer index of simulation replicate',
                        choices=list(range(1000)),
                        default=0)
    parser.add_argument('--intercept', type=int,
                        help='integer index of intercept of data',
                        choices=[0,1,5,10],
                        default=0)
                        
    args = parser.parse_args()

    run_simstudy_replicate(**vars(args))
        

