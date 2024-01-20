# Comparing-Forecast-Performance

This is the repository for the paper `Inference about differences in predictive skill between infectious disease forecasting models`

[Data](https://github.com/donga0223/Comparing-Forecast-Comparison/tree/main/Data) contains R code for getting data from COVID19forecast hub and obtaining scores, and calculating loss differences.

[Simulation1](https://github.com/donga0223/Comparing-Forecast-Comparison/tree/main/Simulation1) includes Python code for simulating data. In this simulation, we generate data using our proposed model and fit the generated data using our proposed model (ARHDFA), Diebold Mariano test (DM-test), and Circular Block Bootstrap (CBB).

[Simulation2](https://github.com/donga0223/Comparing-Forecast-Comparison/tree/main/Simulation2) contains Python code for simulating data. In this simulation, we generate data from a simple ARMA model and fit the generated data using our proposed model (ARHDFA), Dynamic Factor Analysis (DFA), Diebold Mariano test (DM-test), and Circular Block Bootstrap (CBB)."

[Simulation3](https://github.com/donga0223/Comparing-Forecast-Comparison/tree/main/Simulation3) contains Python code for simulating data. However, it's a simplified model with only one location and one horizon. The data is generated using our proposed model (ARHDFA) and Dynamic Factor Analysis (DFA), where DFA does not include heteroskedasticity. The code also includes fitting the simulated data using ARHDFA, DFA, Diebold Mariano test (DM-test), and Circular Block Bootstrap (CBB).

[Application](https://github.com/donga0223/Comparing-Forecast-Comparison/tree/main/Application) contains python code to fit the model to real data.

## Python environment

We have used conda for environment management. To install the conda environment, run the following command with the repository root as your working directory:

```
conda env create -f environment.yml
```
