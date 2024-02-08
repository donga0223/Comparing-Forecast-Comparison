
#remotes::install_github("epiforecasts/scoringutils")

library(stringr)
library(covidHubUtils)
library(doParallel)
library(tidyverse)
library(scoringutils)
library(generics)
#remotes::install_github("reichlab/covidData")
library(covidData)
##################################################
### Load forecast and observed data 
##################################################
#hub_source <- "../../covid/covid19-forecast-hub"


death_targets <- paste(1:5, "wk ahead inc death")

incl_locations <- covidData::fips_codes %>%
  dplyr::filter(nchar(location) == 2, location <= "56", location != "11") %>%
  dplyr::pull(location)

dates <- seq.Date(from = as.Date("2021-07-17"),
                  to = as.Date("2022-04-30"),
                  by = 7) %>%
  as.character()

models = c("BPagano-RtDriven", "COVIDhub-4_week_ensemble", "COVIDhub-baseline", "COVIDhub-trained_ensemble",
           "JHUAPL-Bucky", "RobertWalraven-ESG", "SteveMcConnell-CovidComplete", "USC-SI_kJalpha")
forecasts_deaths <- load_forecasts(
  dates = dates,
  models = models,
  date_window_size = 6,
  locations = incl_locations,
  #types = "quantile",
  targets = death_targets,
  source = "zoltar",
  hub_repo_path = hub_repo_path,
  verbose = FALSE,
  as_of = NULL,
  hub = c("US")
) %>% align_forecasts()


truth_data <- load_truth(
  truth_source = "JHU",
  target_variable = "inc death",
  locations = incl_locations,
)

score_forecasts_deaths <- score_forecasts(
  forecasts = forecasts_deaths,
  truth = truth_data
) %>% align_forecasts()

View(score_forecasts_deaths[1,])


location <- read.csv("Data/locations.txt", header = TRUE)

calculate_diff <- function(model1, model2){
  data1 <- score_forecasts_deaths %>% 
    filter(model == model1) %>% 
    select("reference_date", "location", "relative_horizon", "wis") 
  data2 <- score_forecasts_deaths %>% 
    filter(model == model2) %>% 
    select("reference_date", "location", "relative_horizon", "wis")
  merged_data <- merge(data1, data2, by.x = c("reference_date", "location", "relative_horizon"), 
                       by.y = c("reference_date", "location", "relative_horizon")) %>%
    left_join(location, by = "location") %>% 
    mutate(diff = wis.x - wis.y) %>%
    mutate(diff_pop = diff/(population/100000))
  write.csv(merged_data, paste("Data/death_diff_scores/", model1, "_", model2, ".csv", sep=""))
}

for(i in 1:(length(models)-1)){
  for(j in (i+1):length(models)){
calculate_diff(models[i], models[j])
  }
}

