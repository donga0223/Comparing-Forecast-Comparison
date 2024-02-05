#' Conduct a Diebold-Mariano test provided differences in forecast skill.
#'
#' This function has been taken from the `forecast` package for R, and lightly
#' adapted to take a difference in forecast scores as inputs rather than
#' separate forecast errors for each model.
dm.test <- function(d, alternative = c("two.sided", "less", "greater"),
    h = 1, varestimator = c("acf", "bartlett")) {
  alternative <- match.arg(alternative)
  varestimator <- match.arg(varestimator)
  h <- as.integer(h)
  if (h < 1L) {
    stop("h must be at least 1")
  }
  if (h > length(d)) {
    stop("h cannot be longer than the number of forecast score differences")
  }
  d.cov <- acf(d, na.action = na.omit, lag.max = h - 1, type = "covariance",
               plot = FALSE)$acf[, , 1]
  n <- length(d)
  if (varestimator == "acf" | h == 1L) {
    d.var <- sum(c(d.cov[1], 2 * d.cov[-1])) / n
  } else {
    d.var <- sum(c(d.cov[1], 2 * (1 - seq_len(h - 1) / h) * d.cov[-1])) / n
  }
  dv <- d.var
  if (dv > 0) {
    STATISTIC <- mean(d, na.rm = TRUE)/sqrt(dv)
  } else if (h == 1) {
    stop("Variance of DM statistic is zero")
  } else {
    warning("Variance is negative. Try varestimator = bartlett. Proceeding with horizon h=1.")
    return(dm.test(d, alternative, h = 1, varestimator))
  }
  k <- ((n + 1 - 2 * h + (h / n) * (h - 1)) / n)^(1 / 2)
  STATISTIC <- STATISTIC * k
  names(STATISTIC) <- "DM"
  if (alternative == "two.sided") {
      PVAL <- 2 * pt(-abs(STATISTIC), df = n - 1)
  }
  else if (alternative == "less") {
      PVAL <- pt(STATISTIC, df = n - 1)
  }
  else if (alternative == "greater") {
      PVAL <- pt(STATISTIC, df = n - 1, lower.tail = FALSE)
  }
  PARAMETER <- c(h)
  names(PARAMETER) <- c("Forecast horizon")
  structure(list(statistic = STATISTIC, parameter = PARAMETER, 
      alternative = alternative, varestimator = varestimator, 
      p.value = PVAL, method = "Diebold-Mariano Test", data.name = c(deparse(substitute(e1)), 
          deparse(substitute(e2)))), class = "htest")
}