import os
import argparse
import pathlib
import time
from itertools import combinations

def submit_jobs(test_run):
    shdir = 'application/ardfa_sh'
    pathlib.Path(shdir).mkdir(parents=True, exist_ok=True)
    logdir = 'application/ardfa_logs'
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    savedir = '/work/pi_nick_umass_edu/ARVarDFA/application/fits'
    
    if test_run:
        data1 = ['COVIDhub_4_week_ensemble']
        data2 = ['COVIDhub_baseline']
        val = ['diff']
    else:
        #data1s = ["COVIDhub_4_week_ensemble", "COVIDhub_baseline"]
        data1s = ["COVIDhub_4_week_ensemble", "COVIDhub_baseline", "CU_select", "GT_DeepCOVID", "JHUAPL_Bucky", "USC_SI_kJalpha"]
        #data2s = ["COVIDhub_4_week_ensemble", "COVIDhub_baseline", "CU_select", "GT_DeepCOVID", "JHUAPL_Bucky", "USC_SI_kJalpha"]
        vals = ['diff', 'diff_pop']
        num_chain = 5 # 10  or None

    
    for combo in combinations(data1s, 2):  # 2 for pairs, 3 for triplets, etc
        for val in vals:
            if num_chain is None:
                name = f'{combo[0]}_{combo[1]}_{val}'
                samples_file = pathlib.Path(savedir) / f'fits_{name}.pkl'
                if samples_file.exists() and samples_file.stat().st_size > 0:
                    print(f'Skipping {name}, fitting file already exists')
                    continue
                cmd = f'module load miniconda/22.11.1-1\n' \
                    f'conda activate forecastskill\n' \
                    f'python dfa_application_fit_one.py --data1 {combo[0]} --data2 {combo[1]} --val {val}'
                
                print(f"Launching {name}")

                sh_contents = f'#!/bin/bash\n' \
                                f'#SBATCH --job-name="{name}"\n' \
                                f'#SBATCH --ntasks=1 \n' \
                                f'#SBATCH -c 1 # Number of Cores per Task\n' \
                                f'#SBATCH --nodes=1 # Requested number of nodes\n' \
                                f'#SBATCH --mem=4000 # Requested Memory\n' \
                                f'#SBATCH --output="{logdir}/{name}.out" \n' \
                                f'#SBATCH --error="{logdir}/{name}.err" \n' \
                                f'#SBATCH --partition cpu-long # Partition\n' \
                                f'#SBATCH --time 50:00:00 # Job time limit\n' + cmd
                
                shfile = pathlib.Path(shdir) / f'{name}.sh'
                with open(shfile, 'w') as f:
                    f.write(sh_contents)
                
                os.system(f'sbatch {shfile}')
                time.sleep(1)
            
            elif num_chain is not None:
                for chain in range(num_chain):
                    name = f'{combo[0]}_{combo[1]}_{chain}_{val}'
                    samples_file = pathlib.Path(savedir) / f'fits_{name}.pkl'
                    if samples_file.exists() and samples_file.stat().st_size > 0:
                        print(f'Skipping {name}, fitting file already exists')
                        continue
                    cmd = f'module load miniconda/22.11.1-1\n' \
                        f'conda activate forecastskill\n' \
                        f'python dfa_application_fit_one.py --data1 {combo[0]} --data2 {combo[1]} --val {val} --chain {chain}'
                    
                    print(f"Launching {name}")

                    sh_contents = f'#!/bin/bash\n' \
                                    f'#SBATCH --job-name="{name}"\n' \
                                    f'#SBATCH --ntasks=1 \n' \
                                    f'#SBATCH -c 1 # Number of Cores per Task\n' \
                                    f'#SBATCH --nodes=1 # Requested number of nodes\n' \
                                    f'#SBATCH --mem=4000 # Requested Memory\n' \
                                    f'#SBATCH --output="{logdir}/{name}.out" \n' \
                                    f'#SBATCH --error="{logdir}/{name}.err" \n' \
                                    f'#SBATCH --partition cpu-long # Partition\n' \
                                    f'#SBATCH --time 100:00:00 # Job time limit\n' + cmd
                    
                    shfile = pathlib.Path(shdir) / f'{name}.sh'
                    with open(shfile, 'w') as f:
                        f.write(sh_contents)
                    
                    os.system(f'sbatch {shfile}')
                    time.sleep(1)
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run estimation for application study, multiple combinations of data1, data2, and variable index')
    
    parser.add_argument('--test_run',
                        action=argparse.BooleanOptionalAction,
			help='If specified, run only a small selection of setting for data1, data2, and variable index for testing purposes.')
    parser.set_defaults(test_run=False)
    
    args = parser.parse_args()
    
    submit_jobs(**vars(args))

    