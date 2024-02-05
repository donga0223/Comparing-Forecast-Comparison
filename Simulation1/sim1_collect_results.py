import pathlib
import pickle
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

save_path = pathlib.Path('/work/pi_nick_umass_edu/ARHDFA/Simulation1/arhdfa_fits')

records = []
for sample_size in ['small']:
    print(sample_size)
    for theta in ['0.0', '1.0', '10.0']:
        print(theta)
        for p in [1,2,3]:
            print(p)
            for q in [1,2,3]:
                print(q)
                for num_factor in [2,3,4,5,6]:
                    for condition in ['all_constant', 'all_vary', 'sigma_t_vary', 'sigma_l_vary', 'sigma_h_vary']:
                        for replicate in range(100):
                            name = f'{sample_size}_{theta}_{p}_{q}_{num_factor}_{condition}_{replicate}'
                            result_file = save_path / f'summary_{name}.pkl'
                            if result_file.exists() and result_file.stat().st_size > 0:
                                with open(result_file, 'rb') as f:
                                    result = pickle.load(f)
                                records.append({
                                    'sample_size': sample_size,
                                    'theta': theta,
                                    'condition': condition,
                                    'p': p,
                                    'q': q,
                                    'num_factor': num_factor,
                                    'replicate': replicate,
                                    'DIC': result['DIC_est'],
                                    #'DIC_alt': result['DIC_est_alt'],
                                    'l_95': result['intercept_l_95'],
                                    'u_95': result['intercept_u_95']
                                })
                    


intervals = pd.DataFrame.from_records(records)
intervals.to_csv('/work/pi_nick_umass_edu/ARHDFA/Simulation1/Simulation1_result.txt', sep='\t', index=True)

#coverage = intervals \
#    .assign(incl_theta = lambda x: (x.theta.astype(float) >= x.l_95) & (x.theta.astype(float) <= x.u_95)) \
#    .groupby(['sample_size', 'theta', 'condition', 'p', 'q', 'num_factor']) \
#    ['incl_theta'] \
#    .mean()

#print("\n95% interval coverage rates:")
#print(coverage)

#power = intervals \
#    .loc[intervals.theta != '0.0'] \
#    .assign(incl_zero = lambda x: (0 < x.l_95) | (0 > x.u_95)) \
#    .groupby(['sample_size', 'theta', 'condition', 'p', 'q', 'num_factor']) \
#    ['incl_zero'] \
#    .mean()

#print("\nPower of test from inverting 95% interval (i.e., percent of intervals that don't include 0):")
#print(power)

min_dic_indices = intervals.groupby(['sample_size', 'theta', 'condition', 'replicate'])['DIC'].min()
min_dic_df = min_dic_indices.reset_index()

min_dic_df.to_csv('/work/pi_nick_umass_edu/ARHDFA/Simulation1/Simulation1_min_DIC.txt', sep='\t', index=True)