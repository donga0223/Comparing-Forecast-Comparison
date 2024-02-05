import pathlib
import pickle
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

save_path = pathlib.Path('/work/pi_nick_umass_edu/ARHDFA/Simulation2/arhdfa_fits')

timetypes = ['AR', 'MA','MA5_same', 'MA5_exp']
acfs = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
sigmafactors = ['constant', 'AR']
ps = [1,2,3,4]
num_replicates = 100
thetas = [0,1]

records = []
for timetype in timetypes:
    print(timetype)
    for acf in acfs:
        print(acf)
        for sigmafactor in sigmafactors:
            for theta in thetas:
                for p in ps:
                    for replicate in range(num_replicates):
                        name = f'{timetype}_{acf}_{sigmafactor}_{theta}_{p}_{replicate}'
                        result_file = save_path / f'summary_{name}.pkl'
                        if result_file.exists() and result_file.stat().st_size > 0:
                            with open(result_file, 'rb') as f:
                                result = pickle.load(f)
                            records.append({
                                'timetype': timetype,
                                'sigmafactor': sigmafactor,
                                'acf': acf,
                                'theta': theta,
                                'p': p,
                                'replicate': replicate,
                                'DIC': result['DIC_est'],
                                'l_95': result['intercept_l_95'],
                                'u_95': result['intercept_u_95']
                            })
                


intervals = pd.DataFrame.from_records(records)
intervals.to_csv('/work/pi_nick_umass_edu/ARHDFA/Simulation2/Simulation2_result.txt', sep='\t', index=True)

#coverage = intervals \
#    .assign(incl_theta = lambda x: (x.theta.astype(float) >= x.l_95) & (x.theta.astype(float) <= x.u_95)) \
#    .groupby(['timetype', 'sigmafactor', 'acf', 'theta', 'p']) \
#    ['incl_theta'] \
#    .mean()

#print("\n95% interval coverage rates:")
#print(coverage)

#power = intervals \
#    .loc[intervals.theta != '0.0'] \
#    .assign(incl_zero = lambda x: (0 < x.l_95) | (0 > x.u_95)) \
#    .groupby(['timetype', 'sigmafactor', 'acf', 'theta', 'p']) \
#    ['incl_zero'] \
#    .mean()

#print("\nPower of test from inverting 95% interval (i.e., percent of intervals that don't include 0):")
#print(power)

min_dic_indices = intervals.groupby(['timetype', 'sigmafactor', 'acf', 'theta', 'replicate'])['DIC'].min()
min_dic_df = min_dic_indices.reset_index()

min_dic_df.to_csv('/work/pi_nick_umass_edu/ARHDFA/Simulation2/Simulation2_min_DIC.txt', sep='\t', index=True)