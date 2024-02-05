import os   
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'TKAgg'

import pickle
import pandas as pd
from datetime import date

from DM_CBB import CBB_test
from DM_CBB import stochastic_function
from sim2_ARMA import ARMA_gen_byvar_acf


timetypes = ['AR', 'MA','MA5_same', 'MA5_exp']
thetas = [0,1]
iter = 100
def run_DM_CBB():
    records = []
    for timetype in timetypes:
        print(timetype)
        if timetype == 'MA':
            acfs = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
        else:
            acfs = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
        for acf in acfs:
            print(acf)
            for theta in thetas:
                print(theta)
                for replicate in range(iter):
                    obs = ARMA_gen_byvar_acf(heterotype = 'nohetero', timeseries_type = timetype, 
                            marginal_var = 'True', acf = float(acf), replicate = replicate, 
                            sample_size = 52, intercept = theta)
                    _, DM_QS_stat, DM_QS_p, CBB_QS_p, opt_block, mean_d, V_d = CBB_test(obs[:,0,0],
                                                        seed = int(stochastic_function(replicate)))
                    records.append({
                        'timetype': timetype,
                        'acf': acf,
                        'theta': theta,
                        'replicate': replicate,
                        'DM_QS_stat': DM_QS_stat, 
                        'DM_QS_p': DM_QS_p, 
                        'CBB_QS_p': CBB_QS_p, 
                        'opt_block': opt_block, 
                        'mean_d': mean_d, 
                        'V_d': V_d
                    })
    results = pd.DataFrame.from_records(records)
    results.to_csv('/work/pi_nick_umass_edu/ARHDFA/Simulation2/Simulation2_DM_CBB_result.txt', sep='\t', index=True)

if __name__ == "__main__":   
    run_DM_CBB()

                            

            