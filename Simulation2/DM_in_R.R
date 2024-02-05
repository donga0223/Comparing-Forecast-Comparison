# Note: it is necessary to run `conda activate forecastskill` in the terminal
# before starting the R session used to run this script.

# Set the working directory to Simulation2
library(here)
setwd(file.path(here(), "Simulation2"))

# load the dm.test function
source("../R/dmtest.R")

# setup python and load the ARMA module used for simulating data
library(reticulate)
reticulate::conda_python(envname = "forecastskill")
ARMA <- reticulate::import_from_path("sim2_ARMA")

# get DM test p-values for one simulation replicate
dm_one_replicate <- function(replicate, timetype = "MA5_same", acf_val = 0.5,
                             theta = 0) {
  d <- ARMA$ARMA_gen_byvar_acf(heterotype = "nohetero",
                               timeseries_type = timetype,
                               marginal_var = "True",
                               acf = acf_val,
                               intercept = theta,
                               replicate = replicate)
  dim(d) <- dim(d)[1]

  get_dm_pval <- function(d, h, varestimator) {
    dm.test(d = d, h = h, varestimator = varestimator)$p.value
  }

  result <- tidyr::expand_grid(
    replicate = replicate,
    timetype = timetype,
    acf_val = acf_val,
    h = c(1, 5, 10),
    varestimator = c("acf", "bartlett"))
  result$pval <- purrr::pmap_dbl(
    result[c("h", "varestimator")],
    get_dm_pval,
    d = d)

  return(result)
}

dm_results <- purrr::pmap(
  expand.grid(
    replicate = 1:1000,
    timetype = c("MA", "MA5_exp", "MA5_same", "AR"),
    acf = 0.5
  ),
  dm_one_replicate) |>
  purrr::list_rbind()

dm_results |>
  dplyr::group_by(timetype, h, varestimator) |>
  dplyr::summarize(
    error_rate = mean(pval < 0.05)
  ) |>
  as.data.frame()


# load an array from a pickle file
# this code is not used here, but could be used if we save simulated data
# use_condaenv("forecastskill", required = TRUE)
# a <- reticulate::py_load_object(
#   "ARVarDFA/Simulation1/ardfa_samples_cluster/small_0.0_all_constant.pkl")

# a <- a$y$tolist()

# do.call(cbind, a[[1]])
