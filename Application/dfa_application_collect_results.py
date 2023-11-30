import pathlib
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

save_path = pathlib.Path('/work/pi_nick_umass_edu/ARVarDFA/application/fits')
data1s = ["COVIDhub_4_week_ensemble", "COVIDhub_baseline", "CU_select", "GT_DeepCOVID", "JHUAPL_Bucky", "USC_SI_kJalpha"]
#data1s = ["COVIDhub_4_week_ensemble", "COVIDhub_baseline"]

#name = f'{"COVIDhub_4_week_ensemble"}_{"CU_select"}_{1}_{"diff_pop"}'

chain = 5
records = []
for combo in combinations(data1s, 2):
    print(combo)
    for val in ['diff', 'diff_pop']:
        print(val)
        if chain is None:
            name = f'{combo[0]}_{combo[1]}_{val}'
            samples_file = save_path / f'fits_{name}.pkl'
            if samples_file.exists() and samples_file.stat().st_size > 0:
                with open(samples_file, 'rb') as f:
                    samples = pickle.load(f)
                    records.append({
                            'model': name,
                            'variable': val, 
                            'l_95': np.percentile(samples['intercept'], 2.5),
                            'u_95': np.percentile(samples['intercept'], 97.5),
                            'mean': np.mean(samples['intercept'])
                        })
        elif chain is not None:
            for n_chain in range(chain):
                print(n_chain)
                name = f'{combo[0]}_{combo[1]}_{n_chain}_{val}'
                samples_file = save_path / f'fits_{name}.pkl'
                if samples_file.exists() and samples_file.stat().st_size > 0:
                    with open(samples_file, 'rb') as f:
                        samples = pickle.load(f)
                        if n_chain == 0:
                                all_intercept = samples['intercept']
                        elif n_chain != 0:
                            if combo[0] == "JHUAPL_Bucky" and combo[1] == "USC_SI_kJalpha" and val == 'diff_pop' and n_chain == 4:
                                print("skip it")
                            else:
                                var_name = f'intercept_{n_chain}'
                                globals()[var_name] = samples['intercept']
                                all_intercept = np.concatenate((all_intercept, eval(var_name)), axis=1)
            
            records.append({
                    'model': name,
                    'variable': val, 
                    'l_95': np.percentile(all_intercept, 2.5),
                    'u_95': np.percentile(all_intercept, 97.5),
                    'mean': np.mean(all_intercept)
                })



        

intervals = pd.DataFrame.from_records(records)


with open(samples_file, 'wb') as f:
        pickle.dump(all_intercept, f)



print("\n95% intercept credible interval:")
print(intervals)

#power = intervals \
#    .loc[intervals.theta != '0.0'] \
#    .assign(incl_zero = lambda x: (0 < x.l_95) | (0 > x.u_95)) \
#    .groupby(['sample_size', 'theta']) \
#    ['incl_zero'] \
#    .mean()

#print("\nPower of test from inverting 95% interval (i.e., percent of intervals that don't include 0):")
#print(power)


file_path = save_path /f'application_arhdfa.txt'
intervals = intervals.astype(float)

# Save the NumPy array to a text file
np.savetxt(file_path, intervals, fmt='%f', delimiter='\t')



file_path = 'latent_factors.txt'


# Save the NumPy array to a text file
np.savetxt(file_path, aaa, fmt='%f', delimiter='\t')