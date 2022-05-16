library(R6)
library(mlr3proba)

MeasureSurvCAntolini <- R6Class("MeasureSurvCAntolini",
  inherit = MeasureSurv,
  public = list(
    initialize = function() {
      ps = ps(
        method = p_fct(c("antolini", "adj_antolini"), default = "antolini")
      )
      ps$values = list(method = "antolini")
      super$initialize(
        id = "surv.c_antolini",
        range = 0:1,
        minimize = FALSE,
        packages = character(),
        predict_type = "distr",
        param_set = ps
      )
    }
  ),
  private = list(
    .score = function(prediction, task, train_set, ...) {
      #browser()
      source(here::here("survival-benchmark", "R", "utils", "utils.R"))
      ps = self$param_set$values
      unique_times <- sort(unique(prediction$truth[, 1]))
      distribution <- prediction$data$distr
      mtc <- findInterval(unique_times, as.numeric(colnames(distribution)))
      cdf <- t(distribution[, mtc])
      rownames(cdf) <- unique_times
      if (ps$method == "antolini") {
        is_concordant <- is_concordant_antolini
        is_comparable <- is_comparable_antolini
      }
      else {
        is_concordant <- is_concordant
        is_comparable <- is_comparable
      }

      return(sum_concordant_disc(cdf, prediction$truth[, 1], prediction$truth[, 2], sapply(prediction$truth[, 1], function(x) which(rownames(cdf) == x)), is_concordant) /
        sum_comparable(prediction$truth[, 1], prediction$truth[, 2], is_comparable))
    }
  )
)

mlr_measures$add("surv.c_antolini", MeasureSurvCAntolini)
