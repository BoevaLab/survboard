library(ranger)
library(mlr3proba)
library(mlr3tuningspaces)
library(mlr3misc)

# Adapted from: https://github.com/mlr-org/mlr3learners/blob/HEAD/R/LearnerSurvRanger.R
LearnerSurvRangerCustom <- R6Class("LearnerSurvRangerCustom",
  inherit = mlr3proba::LearnerSurv,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ps(
        alpha = p_dbl(default = 0.5, tags = "train"),
        favor_clinical = p_lgl(default = FALSE, tags = "train"),
        holdout = p_lgl(default = FALSE, tags = "train"), # FIXME: do we need this?
        importance = p_fct(c("none", "impurity", "impurity_corrected", "permutation"), tags = "train"),
        keep.inbag = p_lgl(default = FALSE, tags = "train"),
        max.depth = p_int(default = NULL, lower = 0L, special_vals = list(NULL), tags = "train"),
        min.node.size = p_int(1L, default = 3L, special_vals = list(NULL), tags = "train"),
        minprop = p_dbl(default = 0.1, tags = "train"),
        mtry = p_int(lower = 1L, special_vals = list(NULL), tags = "train"),
        mtry.ratio = p_dbl(lower = 0, upper = 1, tags = "train"),
        num.random.splits = p_int(1L, default = 1L, tags = "train"), # requires = quote(splitrule == "extratrees")
        # num.threads                  = p_int(1L, default = 1L, tags = c("train", "predict", "threads")),
        num.trees = p_int(1L, default = 500L, tags = c("train", "predict")),
        oob.error = p_lgl(default = TRUE, tags = "train"),
        regularization.factor = p_uty(default = 1, tags = "train"),
        regularization.usedepth = p_lgl(default = FALSE, tags = "train"),
        replace = p_lgl(default = TRUE, tags = "train"),
        respect.unordered.factors = p_fct(c("ignore", "order", "partition"), default = "ignore", tags = "train"), # for splitrule == "extratrees", def = partition
        sample.fraction = p_dbl(0L, 1L, tags = "train"), # for replace == FALSE, def = 0.632
        save.memory = p_lgl(default = FALSE, tags = "train"),
        scale.permutation.importance = p_lgl(default = FALSE, tags = "train"), # requires = quote(importance == "permutation")
        seed = p_int(default = NULL, special_vals = list(NULL), tags = c("train", "predict")),
        split.select.weights = p_dbl(0, 1, tags = "train"),
        splitrule = p_fct(c("logrank", "extratrees", "C", "maxstat"), default = "logrank", tags = "train"),
        verbose = p_lgl(default = TRUE, tags = c("train", "predict")),
        write.forest = p_lgl(default = TRUE, tags = "train")
      )

      super$initialize(
        id = "surv.ranger_custom",
        param_set = ps,
        predict_types = c("distr", "crank"),
        feature_types = c("logical", "integer", "numeric", "character", "factor", "ordered"),
        properties = c("weights", "importance", "oob_error"),
        packages = c("mlr3learners", "ranger"),
        man = "mlr3learners::mlr_learners_surv.ranger"
      )
    },

    #' @description
    #' The importance scores are extracted from the model slot `variable.importance`.
    #' @return Named `numeric()`.
    importance = function() {
      if (is.null(self$model)) {
        stopf("No model stored")
      }
      if (self$model$importance.mode == "none") {
        stopf("No importance stored")
      }

      sort(self$model$variable.importance, decreasing = TRUE)
    },

    #' @description
    #' The out-of-bag error is extracted from the model slot `prediction.error`.
    #' @return `numeric(1)`.
    oob_error = function() {
      self$model$prediction.error
    }
  ),
  private = list(
    .train = function(task) {
      source(here::here("survboard", "R", "utils", "utils.R"))
      pv <- self$param_set$get_values(tags = "train")
      pv <- convert_ratio(pv, "mtry", "mtry.ratio", length(task$feature_names))
      targets <- task$target_names
      if (pv$favor_clinical) {
        always_split_variables <- grep("clinical", task$feature_names, value = TRUE)
        pv$always.split.variables <- always_split_variables
      }
      pv <- pv[-grep("favor_clinical", names(pv))]
      mlr3misc::invoke(ranger::ranger,
        formula = NULL,
        dependent.variable.name = targets[1L],
        status.variable.name = targets[2L],
        data = task$data(),
        case.weights = task$weights$weight,
        num.threads = 1,
        .args = pv
      )
    },
    .predict = function(task) {
      source(here::here("survboard", "R", "utils", "utils.R"))
      pv <- self$param_set$get_values(tags = "predict")
      newdata <- ordered_features(task, self)

      prediction <- mlr3misc::invoke(predict, self$model, data = newdata, .args = pv)
      mlr3proba::.surv_return(times = prediction$unique.death.times, surv = prediction$survival)
    }
  )
)

mlr_learners$add("surv.ranger_custom", LearnerSurvRangerCustom)
