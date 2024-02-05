To replicate the first simulation study, follow these steps:

1. Edit `config.json` in the repository root to specify the location where output files should be saved.

2. In a terminal with the repository root as the working directory, run the following commands:

```
conda activate forecastskill
python Simulation1/sim_gen_data.py
```

3. To fit a single replicate, run a command like `python sim1_fit_one_replicate.py --sample_size small --theta 0.0 --p 1 --q 1 --num_factors 4 --condition all_vary --replicate 0`

4. Or to fit many replicates, run `python sim1_fit_all_replicates.py`. Note that this assumes you're working on the Unity cluster at UMass.

5. Run `python sim1_collect_results.py` to produce summary .txt file with results.

6. To fit many replicates for DM and CBB, run `python sim1_fit_DM_CBB.py`

7. If you submit the DM_CBB work mentioned above to the cluster, run `python sim1_fit_DM_CBB_sh.py` 