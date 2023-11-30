
#remotes::install_github("epiforecasts/scoringutils")

library(stringr)
library(covidHubUtils)
library(doParallel)
library(tidyverse)
library(scoringutils)
library(generics)

#setwd("/Users/dongahkim/Creative Cloud Files/epi/comparing-forecasting-performance")

##################################################
### Load forecast and observed data 
##################################################

inc_hosp_targets <- paste(0:14, "day ahead inc hosp")

location_states <- hub_locations %>%
  filter(geo_type == "state") %>% #remove counties
  pull(fips) #pull just the location codes 

forecasts_hosp <- load_forecasts(
  #models = c("COVIDhub-ensemble", "COVIDhub-baseline", "CU-select", "Karlen-pypm",  "UMass-MechBayes"),
  #models = c("COVIDhub-ensemble", "COVIDhub-baseline"),
  #dates = as.Date("2020-10-05") + seq(0, length.out=100, by=7),
  dates = seq.Date(as.Date("2020-10-05"), as.Date("2022-08-31"), by = 7),
  date_window_size = 6,
  locations = location_states,
  types = c("point", "quantile"),
  targets = inc_hosp_targets,
  source = "zoltar",
  verbose = FALSE,
  as_of = NULL,
  hub = c("US")
)


write.csv(forecasts_hosp, "Data/forecasts_hosp.csv") #Save your data to a .csv file for future use.


truth_data <- load_truth(
  truth_source = "JHU",
  #target_variable = "inc hosp",
  locations = location_states,
)


truth_data_hosp <- truth_data %>% filter(target_variable == "inc hosp")


write.csv(truth_data_hosp, "Data/truth_data_hosp.csv") #Save your data to a .csv file for future use.

##################################################
### Select models based on lower levels of missing data.
##################################################
#forecasts_hosp <- read.csv("Data_description/forecasts_hosp.csv")
#truth_data_hosp <- read.csv("Data_description/truth_data_hosp.csv")

unique_date <- unique(forecasts_hosp$forecast_date)
unique_date <- as.Date(unique_date)
unique_date <- unique_date[order(unique_date)]

forecasts_hosp$model <- str_replace_all(forecasts_hosp$model, "-", "_")
unique(forecasts_hosp$model)
mymodel <- unique(forecasts_hosp$model). ## total model

forecasts_hosp_copy <- forecasts_hosp


forecasts_hosp_copy$location <- as.numeric(forecasts_hosp$location)


## create several forecast_data by each model 
by_model <- function(mymodel){
  forecasts_hosp_copy[,-1] %>%  filter(model == mymodel, location != "US", location <=56, location != 11, horizon <=14) #, type == "quantile") 
}

for(i in 1:length(mymodel)){
  assign(paste("forecasts_hosp", mymodel[i], sep = "_"), by_model(mymodel[i]))
}

## add a reference date variable to each dataset
align_bymodel <- function(data_bymodel){
  data_bymodel[["forecast_date"]] <- as.Date(data_bymodel[["forecast_date"]])
  data_bymodel[["target_end_date"]] <- as.Date(data_bymodel[["target_end_date"]])
  data_bymodel %>% align_forecasts()
}

for(i in 1:length(mymodel)){
  assign(paste("forecasts_hosp", mymodel[i], "ref", sep = "_"), align_bymodel(get(paste("forecasts_hosp", mymodel[i], sep = "_"))))
}


## check the percentage of forecasts reported by each team within the data we are using
date_list <- list()
for(i in 1:length(mymodel)){
  date_list[[i]] <- unique(get(paste("forecasts_hosp", mymodel[i], "ref", sep = "_"))$reference_date)
}

unique_reference_date <- unique(as.Date(unlist(date_list)))
date_check <- matrix(NA, nrow = length(unique_reference_date), ncol = length(mymodel))
rownames(date_check) <- as.character(unique_reference_date)
colnames(date_check) <- mymodel
head(date_check)
for(i in 1:length(mymodel)){
  date_check[,i] <- as.character(unique_reference_date) %in%  as.character(date_list[[i]])*1
}

date_check

write.csv(date_check, "Data/date_check.csv") # if you want to save this to .csv file



##################################################
### Calculate the score for the selected model
##################################################

selected_model <- mymodel[c(3, 4, 12, 15, 18, 34)]

##add scores to the data 
scores_by_model <- function(data_bymodel){
  data_bymodel[["location"]] <- sprintf("%02d", as.numeric(as.character(data_bymodel[["location"]])))
  #data_bymodel[["location"]] <- as.character(data_bymodel[["location"]])
  data_bymodel[["target_end_date"]] <- as.Date(data_bymodel[["target_end_date"]])
  truth_data_hosp[["target_end_date"]] <- as.Date(truth_data_hosp[["target_end_date"]])
  
  scores <- score_forecasts(
    forecasts = data_bymodel,
    return_format = "wide",
    truth = truth_data_hosp
  )
  return(scores)
}


for(i in 1:length(selected_model)){
  assign(paste("scores", selected_model[i], sep = "_"), scores_by_model(get(paste("forecasts_hosp", selected_model[i], sep = "_"))))
  print(i)
}

head(scores_COVIDhub_4_week_ensemble)

## add reference date
align_bymodel <- function(data_bymodel){
  data_bymodel[["forecast_date"]] <- as.Date(data_bymodel[["forecast_date"]])
  data_bymodel[["target_end_date"]] <- as.Date(data_bymodel[["target_end_date"]])
  data_bymodel %>% align_forecasts()
}

for(i in 1:length(selected_model)){
  assign(paste("scores", selected_model[i], "ref", sep = "_"), align_bymodel(get(paste("scores", selected_model[i], sep = "_"))))
  print(i)
}

head(scores_COVIDhub_4_week_ensemble_ref$reference_date)

## save it to .csv file
for(i in 1:length(selected_model)){
  write.csv(get(paste("scores", selected_model[i], "ref", sep = "_")),paste("Data/scores_lossdiff/scores_", selected_model[i], "_ref", ".csv", sep = ""))
}

##############################################################################
## Calculate the loss diff
##############################################################################


diff_function <- function(model1, model2, location){
  scores_diff <- merge(model1, model2, by.x = c("reference_date", "location", "relative_horizon"), by.y = c("reference_date", "location", "relative_horizon"))
  #scores_diff <- dplyr::left_join(model1, model2,  by = c("reference_date location horizon" = "target_end_date location horizon"))
  #scores_diff <- cbind(model1, model2$target_end_date, model2$wis)
  scores_diff$diff <- scores_diff$wis.x - scores_diff$wis.y
  scores_diff_pop <- left_join(scores_diff, location, by = "location")
  scores_diff_pop$diff_pop <- scores_diff_pop$diff/(scores_diff_pop$population/100000)
  res <- scores_diff_pop %>% 
    dplyr::select(abbreviation, location, reference_date, forecast_date.x , forecast_date.y, target_end_date.x, target_end_date.y, relative_horizon, diff,  diff_pop)
  return(res)
}

location <- read.csv("Data_description/locations.txt", header = TRUE)


for(i in 1:(length(selected_model)-1)){
  for(j in (i+1):length(selected_model)){
    model1 <- get(paste("scores", selected_model[i], "ref", sep = "_"))
    model2 <- get(paste("scores", selected_model[j], "ref", sep = "_"))
    assign(paste("diff",selected_model[i], selected_model[j],sep="_"),  diff_function(model1, model2, location))
    write.csv(get(paste("diff",selected_model[i], selected_model[j],sep="_")),paste("Data/scores_lossdiff/diff_", selected_model[i], "_", selected_model[j], ".csv", sep = ""))
    
  }
}


