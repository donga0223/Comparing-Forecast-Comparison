import pathlib
import pickle
import numpy as np
import pandas as pd

save_path = pathlib.Path('/work/pi_nick_umass_edu/ARVarDFA/simulation/ardfa_ARMAtest')

timetypes = ['AR', 'MA','MA5_same', 'MA5_exp']
acfs = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
num_replicates = 1000
#intercepts = [1,5,10]
intercepts = [5]

records = []
for timetype in timetypes:
    print(timetype)
    for acf in acfs:
        print(acf)
        for intercept in intercepts:
            print(intercept)
            for replicate in range(num_replicates):
                #print(replicate)
                name = f'{timetype}_{acf}_{replicate}_{intercept}'
                samples_file = save_path / f'samples_{name}.pkl'
                if samples_file.exists() and samples_file.stat().st_size > 0:
                    with open(samples_file, 'rb') as f:
                        samples = pickle.load(f)
                    records.append({
                        'timetype': timetype,
                        'acf': acf,
                        'intercept': intercept,
                        'replicate': replicate,
                        'l_95': np.percentile(samples['intercept'], 2.5),
                        'u_95': np.percentile(samples['intercept'], 97.5)
                    })
        


intervals = pd.DataFrame.from_records(records)
coverage = intervals \
    .assign(incl_zero = lambda x: (0 >= x.l_95) & (0 <= x.u_95)) \
    .groupby(['timetype', 'acf', 'intercept']) \
    ['incl_zero'] \
    .mean()

print("\n95% interval coverage rates:")
print(coverage)

typeIerror = intervals \
    .assign(typeI = lambda x: (0 < x.l_95) | (0 > x.u_95)) \
    .groupby(['timetype', 'acf', 'intercept']) \
    ['typeI'] \
    .mean()

print("\n type I error rates:")
print(typeIerror)

power = intervals \
    .loc[intervals.intercept != '0.0'] \
    .assign(incl_zero = lambda x: (0 < x.l_95) | (0 > x.u_95)) \
    .groupby(['timetype', 'acf', 'intercept']) \
    ['incl_zero'] \
    .mean()

print("\nPower of test from inverting 95% interval (i.e., percent of intervals that don't include 0):")
print(power)


with open('ARMAtest_dfa_sim_5_summary.txt', 'w') as f:
    f.write(intervals.to_string(index=False))


