import pathlib
import pickle
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

save_path = pathlib.Path('/work/pi_nick_umass_edu/ARHDFA/Simulation3/arhdfa_fits')

records = []
chain = None
for sample_size in ['small']:
    print(sample_size)
    for theta in ['0.0', '1.0', '10.0']:
        print(theta)
        for condition in ['sigma_t_vary_random', 'all_constant_random']:#, 'all_constant_fix', 'sigma_t_vary_fix']
            print(condition)
            for sigmafactor in ['constant', 'AR']:
                print(sigmafactor)
                for replicate in range(200):
                    name = f'{sample_size}_{theta}_{condition}_{sigmafactor}_{replicate}'
                    result_file = save_path / f'summary_{name}.pkl'
                    if result_file.exists() and result_file.stat().st_size > 0:
                        with open(result_file, 'rb') as f:
                            result = pickle.load(f)
                        records.append({
                            'sample_size': sample_size,
                            'theta': theta,
                            'replicate': replicate,
                            'condition': condition,
                            'sigmafactor': sigmafactor,
                            'replicate': replicate,
                            'l_95': result['intercept_l_95'],
                            'u_95': result['intercept_u_95']
                        })
                


intervals = pd.DataFrame.from_records(records)
intervals.to_csv('/work/pi_nick_umass_edu/ARHDFA/Simulation3/Simulation3_result.txt', sep='\t', index=True)

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


