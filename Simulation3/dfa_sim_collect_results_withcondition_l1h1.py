import pathlib
import pickle
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

save_path = pathlib.Path('/work/pi_nick_umass_edu/ARVarDFA/simulation/ardfa_ARVar_fits_l1h1_2')

records = []
chain = None
for sample_size in ['small']:
    print(sample_size)
    for theta in ['0.0', '1.0', '10.0']:
        print(theta)
        for condition in ['sigma_t_vary_random', 'sigma_t_vary_fix', 'all_constant_random', 'all_constant_fix']:
            print(condition)
            for sigmafactor in ['constant', 'AR']:
                print(sigmafactor)
                for replicate in range(1000):
                    name = f'{sample_size}_{theta}_{condition}_{sigmafactor}_{replicate}'
                    samples_file = save_path / f'samples_{name}.pkl'
                    if samples_file.exists() and samples_file.stat().st_size > 0:
                        with open(samples_file, 'rb') as f:
                            samples = pickle.load(f)
                        records.append({
                            'sample_size': sample_size,
                            'theta': theta,
                            'replicate': replicate,
                            'condition': condition,
                            'sigmafactor': sigmafactor,
                            'l_95': np.percentile(samples['intercept'], 2.5),
                            'u_95': np.percentile(samples['intercept'], 97.5)
                        })
                


intervals = pd.DataFrame.from_records(records)
coverage = intervals \
    .assign(incl_theta = lambda x: (x.theta.astype(float) >= x.l_95) & (x.theta.astype(float) <= x.u_95)) \
    .groupby(['sample_size', 'theta', 'condition', 'sigmafactor']) \
    ['incl_theta'] \
    .mean()

print("\n95% interval coverage rates:")
print(coverage)

power = intervals \
    .loc[intervals.theta != '0.0'] \
    .assign(incl_zero = lambda x: (0 < x.l_95) | (0 > x.u_95)) \
    .groupby(['sample_size', 'theta', 'condition', 'sigmafactor']) \
    ['incl_zero'] \
    .mean()

print("\nPower of test from inverting 95% interval (i.e., percent of intervals that don't include 0):")
print(power)


