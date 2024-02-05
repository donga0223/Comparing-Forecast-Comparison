import os   
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'TKAgg'

import pickle
import pandas as pd
from datetime import date

from DM_CBB import CBB_test
from DM_CBB import stochastic_function


sample_sizes = ['small']
thetas = ['0.0', '1.0', '10.0']
conditions = ['sigma_t_vary_random', 'all_constant_random']#, 'all_constant_fix', 'sigma_t_vary_fix']
iter = 100
def run_DM_CBB():
    records = []
    for sample_size in sample_sizes:
        for theta in thetas:
            for condition in conditions:
                for replicate in range(iter):
                    with open(f'Simulation3/ardfa_samples/{sample_size}_{theta}_{condition}.pkl', 'rb') as f:
                        obs = pickle.load(f)['y'][replicate, ...]
    
                    _, DM_QS_stat, DM_QS_p, CBB_QS_p, opt_block, mean_d, V_d = CBB_test(obs[:,0,0],
                                                        seed = int(stochastic_function(replicate)))
                    records.append({
                        'sample_size': sample_size,
                        'theta': theta,
                        'condition': condition,
                        'replicate': replicate,
                        'DM_QS_stat': DM_QS_stat, 
                        'DM_QS_p': DM_QS_p, 
                        'CBB_QS_p': CBB_QS_p, 
                        'opt_block': opt_block, 
                        'mean_d': mean_d, 
                        'V_d': V_d
                    })
    results = pd.DataFrame.from_records(records)
    results.to_csv('/work/pi_nick_umass_edu/ARHDFA/Simulation3/Simulation3_DM_CBB_result.txt', sep='\t', index=True)

if __name__ == "__main__":   
    run_DM_CBB()

                            

            