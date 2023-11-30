import pathlib
import pickle
import numpy as np
import pandas as pd
from itertools import combinations

save_path = pathlib.Path('/work/pi_nick_umass_edu/ARVarDFA/application/fits')
data1s = ["COVIDhub_4_week_ensemble", "COVIDhub_baseline", "CU_select", "GT_DeepCOVID", "JHUAPL_Bucky", "USC_SI_kJalpha"]
#data1s = ["COVIDhub_4_week_ensemble", "GT_DeepCOVID"]

#name = f'{"COVIDhub_4_week_ensemble"}_{"CU_select"}_{0}_{"diff_pop"}'

chain = 5 
## , 'sigma_nu', , 'sigma_zeta', 'factor_loadings' 없음
var_list = ['ARVar_mu', 'alpha', 'sigma_nu', 'Psi_a', 'Psi_b', 'h_rho', 'beta0', 'beta1', 'sigma_zeta', 'phi', 'sigma_eps_l', 'factor_loadings'] # as one parameter  
## add factor_loadings as well
#var_list = ['Psi_a']#, 'Psi_a', 'Psi_b', 'alpha', 'beta0', 'beta1', 'h_rho', 'phi']#, 'sigma_nu', 'sigma_zeta']

for val in ['diff']:
    print(val)
    for combo in combinations(data1s, 2):
        print(combo)
        if chain is None:
            name = f'{combo[0]}_{combo[1]}_{val}'
            samples_file = save_path / f'fits_{name}.pkl'
            if samples_file.exists() and samples_file.stat().st_size > 0:
                with open(samples_file, 'rb') as f:
                    samples = pickle.load(f)
                    
        elif chain is not None:
            for n_chain in range(chain):
                print(n_chain)
                name = f'{combo[0]}_{combo[1]}_{n_chain}_{val}'
                samples_file = save_path / f'fits_{name}.pkl'
                if samples_file.exists() and samples_file.stat().st_size > 0:
                    with open(samples_file, 'rb') as f:
                        samples = pickle.load(f)
                        for var in var_list:
                            #print(var)
                            my_var = f'all_{var}' 
                            if name == f'{"COVIDhub_4_week_ensemble"}_{"COVIDhub_baseline"}_{0}_{val}':
                                globals()[my_var] = samples[var]
                            else:
                                var_name = f'{combo[0]}_{combo[1]}_{n_chain}_{val}_{var}'
                                globals()[var_name] = samples[var]
                                globals()[my_var] = np.concatenate((eval(my_var), eval(var_name)), axis=1)


        res = {
            'all_ARVar_mu' : all_ARVar_mu,
            'all_alpha' : all_alpha, 
            'all_sigma_nu' : all_sigma_nu, 
            'all_Psi_a' : all_Psi_a, 
            'all_Psi_b' : all_Psi_b, 
            'all_h_rho' : all_h_rho, 
            'all_beta0' : all_beta0, 
            'all_beta1' : all_beta1, 
            'all_sigma_zeta' : all_sigma_zeta, 
            'all_phi' : all_phi,
            'sigma_eps_l' : all_sigma_eps_l,
            'factor_loadings' : all_factor_loadings
        }
    
        save_file = save_path / f'all_{val}_for_simulation.pkl'
        with open(save_file, 'wb') as f:
                pickle.dump(res, f)



#power = intervals \
#    .loc[intervals.theta != '0.0'] \
#    .assign(incl_zero = lambda x: (0 < x.l_95) | (0 > x.u_95)) \
#    .groupby(['sample_size', 'theta']) \
#    ['incl_zero'] \
#    .mean()

#print("\nPower of test from inverting 95% interval (i.e., percent of intervals that don't include 0):")
#print(power)


