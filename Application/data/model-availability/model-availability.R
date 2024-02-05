library(tidyverse)
library(covidHubUtils)
library(covidData)

# hub source:
#  - set to "zoltar" for easy and reproducible access without a local clone of
#    the covid19 forecast hub
#  - set to a path to a local clone of the forecast hub for faster processing
# hub_source <- "zoltar"
hub_source <- "../../covid/covid19-forecast-hub"

# query parameters -- targets and locations
death_targets <- paste(1:5, "wk ahead inc death")
incl_locations <- covidData::fips_codes %>%
    dplyr::filter(nchar(location) == 2, location <= "56", location != "11") %>%
    dplyr::pull(location)

# forecast due dates, reference dates for death forecasts; all are Mondays
dates <- seq.Date(from = as.Date("2020-11-09"),
                  to = as.Date("2023-03-27"),
                  by = 7) %>%
    as.character()

# assemble data frame with availability of death forecasts by model
# and reference date
if (hub_source == "zoltar") {
  source <- "zoltar"
  hub_repo_path <- NULL
} else {
  source <- "local_hub_repo"
  hub_repo_path <- hub_source
}
death_forecast_avail <- purrr::map_dfr(
    dates,
    function(date) {
        message(date)
        load_forecasts(
            dates = date,
            date_window_size = 6,
            locations = incl_locations,
            types = "quantile",
            targets = death_targets,
            source = source,
            hub_repo_path = hub_repo_path,
            verbose = FALSE,
            as_of = NULL,
            hub = c("US")
        ) %>%
            # align forecasts that may have been submitted on different dates
            # around a common reference date, keep only up to relative horizon
            # of 28 dayes (relative to the reference date)
            align_forecasts() %>%
            dplyr::filter(relative_horizon <= 4) %>%
            # keep only model/location/date/horizon combos with all quantiles
            dplyr::group_by(model, location, reference_date,
                            relative_horizon) %>%
            dplyr::summarize(n_quantiles = dplyr::n(), .groups = "drop") %>%
            dplyr::filter(n_quantiles == 23L) %>%
            # for each model/location/reference date, track how many horizons
            # are available
            dplyr::group_by(model, location, reference_date) %>%
            dplyr::summarize(
                d_1wk_avail = (all(seq_len(1) %in% relative_horizon)),
                d_2wk_avail = (all(seq_len(2) %in% relative_horizon)),
                d_3wk_avail = (all(seq_len(3) %in% relative_horizon)),
                d_4wk_avail = (all(seq_len(4) %in% relative_horizon)),
                .groups = "drop"
            ) %>%
            # for each model/reference date, count how many locations have all
            # of horizons 1, horizons 1 through 2, or horizons 1 through 3, 
            # all of horizons 1 through 4
            dplyr::group_by(model, reference_date) %>%
            dplyr::summarize(
                n_loc_d1wk_avail = sum(d_1wk_avail),
                n_loc_d2wk_avail = sum(d_2wk_avail),
                n_loc_d3wk_avail = sum(d_3wk_avail),
                n_loc_d4wk_avail = sum(d_4wk_avail)
            )
    })



saveRDS(
  death_forecast_avail,
  "Application/data/model-availability/death_forecast_avail.rds")



## plot for death_n_loc_d1wk_available_all

death_forecast_avail <- readRDS(
  "Application/data/model-availability/death_forecast_avail.rds")

removed_models <- c("CU-scenario_mid", "CU-scenario_low", "CU-scenario_high",
                    "CU-nochange", "COVIDhub-ensemble", "COVIDhub_CDC-ensemble")

pdf("plots-death/death_n_loc_d4wk_avail_all.pdf", width = 12, height = 8)
ggplot(data = death_forecast_avail |>
    dplyr::filter(!(model %in% removed_models))) +
    geom_tile(mapping = aes(x = reference_date,
                            y = model,
                            fill = n_loc_d4wk_avail)) +
    theme_bw()
dev.off()

get_model_subset <- function(min_date, max_date, p_threshold) {
  result <- death_forecast_avail |>
    dplyr::filter(!(model %in% removed_models),
                  reference_date >= as.Date(min_date),
                  reference_date <= as.Date(max_date)) |>
    dplyr::group_by(model) |>
    dplyr::summarize(
      n1 = sum(n_loc_d1wk_avail),
      n2 = sum(n_loc_d2wk_avail),
      n3 = sum(n_loc_d3wk_avail),
      n4 = sum(n_loc_d4wk_avail),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      p1 = n1 / max(n1),
      p2 = n2 / max(n1),
      p3 = n3 / max(n1),
      p4 = n4 / max(n1)
    ) |>
    dplyr::filter(
      p1 >= p_threshold,
      p2 >= p_threshold,
      p3 >= p_threshold,
      p4 >= p_threshold
    ) |>
    as.data.frame()

  return(result)
}

get_model_subset(min_date = "2020-11-07", max_date = "2021-09-25",
                 p_threshold = 0.95)

get_model_subset(min_date = "2021-07-17", max_date = "2022-04-30",
                 p_threshold = 0.95)
