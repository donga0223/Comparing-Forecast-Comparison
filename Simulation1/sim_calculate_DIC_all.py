import os
import argparse
import pathlib
import time

def submit_jobs(test_run):
    shdir = 'simulation/ardfa_sh'
    pathlib.Path(shdir).mkdir(parents=True, exist_ok=True)
    logdir = 'simulation/ardfa_logs_pq'
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    savedir = '/work/pi_nick_umass_edu/ARVarDFA/simulation/DIC'
    
    if test_run:
        sample_sizes = ['small']
        thetas = ['0.0']
        ps = [1]
        qs = [1]
        num_factors = [4]
        conditions = ['all_constant']
        num_replicates = 1
    else:
        sample_sizes = ['small']
        thetas = ['0.0']#, '1.0', '10.0']
        ps = [1]#,2,3]
        qs = [1]#,2,3]
        num_factors = [2,3,4,5,6]
        #conditions = ['all_vary', 'sigma_t_vary', 'sigma_l_vary', 'sigma_h_vary', 'all_constant']
        conditions = ['all_vary', 'all_constant']
        num_replicates = 5
        
    for sample_size in sample_sizes:
        for theta in thetas:
            for p in ps:
                for q in qs:
                    for num_factor in num_factors:
                        for condition in conditions:
                            for replicate in range(num_replicates):
                                name = f'{sample_size}_{theta}_{p}_{q}_{num_factor}_{condition}_{replicate}'
                                samples_file = pathlib.Path(savedir) / f'DIC_{name}.txt'
                                if samples_file.exists() and samples_file.stat().st_size > 0:
                                    print(f'Skipping {name}, samples file already exists')
                                    continue
                    
                                cmd = f'module load miniconda/22.11.1-1\n' \
                                    f'conda activate forecastskill\n' \
                                    f'python sim_calculate_DIC_one.py --sample_size {sample_size} --theta {theta} --p {p} --q {q} --num_factor {num_factor} --condition {condition} --replicate {replicate}'
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
                                            f'#SBATCH --time 250:00:00 # Job time limit\n' + cmd
                                
                                shfile = pathlib.Path(shdir) / f'{name}.sh'
                                with open(shfile, 'w') as f:
                                    f.write(sh_contents)
                                
                                os.system(f'sbatch {shfile}')
                                time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run estimation for simulation study, multiple combinations of sample size, theta, condition, and replicate index')
    
    parser.add_argument('--test_run',
                        action=argparse.BooleanOptionalAction,
			help='If specified, run only a small selection of setting for sample size, theta, condition, and replicate index for testing purposes.')
    parser.set_defaults(test_run=False)
    
    args = parser.parse_args()
    
    submit_jobs(**vars(args))

    