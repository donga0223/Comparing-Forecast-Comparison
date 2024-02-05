import os   
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'TKAgg'

import pickle
import pandas as pd
from datetime import date

from DM_CBB import CBB_test
from DM_CBB import stochastic_function

iter = 1000
def run_DM_CBB():
    records = []
    for sample_size in ['small']:
        print(sample_size)
        for theta in ['0.0', '1.0', '10.0']:
            print(theta)
            for condition in ['all_vary', 'sigma_t_vary', 'sigma_l_vary', 'sigma_h_vary', 'all_constant']:
                print(condition)
                with open(f'Simulation1/arhdfa_samples/{sample_size}_{theta}_{condition}.pkl', 'rb') as f:
                    obs = pickle.load(f)['y']
                mean_y = obs.mean(axis=(2,3))
                for i in range(iter):
                    _, DM_QS_stat, DM_QS_p, CBB_QS_p, opt_block, mean_d, V_d = CBB_test(mean_y[i,:],
                                                        seed = int(stochastic_function(i)))
                    #DM_p_all.append(DM_p)
                    #DM_stat_all.append(DM_stat)
                    records.append({
                        'sample_size': sample_size,
                        'theta': theta,
                        'condition': condition,
                        'replicate': i,
                        'DM_QS_stat': DM_QS_stat, 
                        'DM_QS_p': DM_QS_p, 
                        'CBB_QS_p': CBB_QS_p, 
                        'opt_block': opt_block, 
                        'mean_d': mean_d, 
                        'V_d': V_d
                    })
    results = pd.DataFrame.from_records(records)
    results.to_csv('/work/pi_nick_umass_edu/ARHDFA/Simulation1/Simulation1_DM_CBB_result_1000.txt', sep='\t', index=True)

if __name__ == "__main__":   
    run_DM_CBB()
                             

                