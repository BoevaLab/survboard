library(mlr3proba)
library(mlr3tuning)
library(splitTools)
library(withr)

# Adapted from mlr3extralearners: https://github.com/mlr-org/mlr3extralearners/blob/main/R/learner_CoxBoost_surv_cv_coxboost.R
#' @title Survival Cox Model with Cross-Validation Likelihood Based Boosting Learner
#' @author RaphaelS1
#' @name mlr_learners_surv.cv_coxboost
#'
#' @template class_learner
#' @templateVar id surv.cv_coxboost
#' @templateVar caller cv.CoxBoost
#'
#' @details
#' Use [LearnerSurvCoxboost] and [LearnerSurvCVCoxboost] for Cox boosting without and with internal
#' cross-validation of boosting step number, respectively. Tuning using the internal optimizer in
#' [LearnerSurvCVCoxboost] may be more efficient when tuning `stepno` only. However, for tuning
#' multiple hyperparameters, \CRANpkg{mlr3tuning} and [LearnerSurvCoxboost] will likely give better
#' results.
#'
#' If `penalty == "optimCoxBoostPenalty"` then [CoxBoost::optimCoxBoostPenalty] is used to determine
#' the penalty value to be used in [CoxBoost::cv.CoxBoost].
#'
#' @references
#' `r format_bib("binder2009boosting")`
#'
#' @template seealso_learner
#' @template example
#' @export
LearnerSurvCVCoxboostCustom <- R6Class("LearnerSurvCVCoxboostCustom",
  inherit = mlr3proba::LearnerSurv,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ps(
        maxstepno = p_int(default = 100, lower = 0, tags = c("train", "cvpars")),
        K = p_int(default = 10, lower = 2, tags = c("train", "cvpars")),
        type = p_fct(default = "verweij", levels = c("verweij", "naive"), tags = c("train", "cvpars")),
        folds = p_uty(default = NULL, tags = c("train", "cvpars")),
        minstepno = p_int(default = 50, lower = 0, tags = c("train", "optimPenalty")),
        start.penalty = p_dbl(tags = c("train", "optimPenalty")),
        iter.max = p_int(default = 10, lower = 1, tags = c("train", "optimPenalty")),
        upper.margin = p_dbl(default = 0.05, lower = 0, upper = 1, tags = c("train", "optimPenalty")),
        unpen.index = p_uty(tags = "train"),
        standardize = p_lgl(default = TRUE, tags = "train"),
        penalty = p_dbl(special_vals = list("optimCoxBoostPenalty"), tags = c("train", "optimPenalty")),
        criterion = p_fct(default = "pscore", levels = c("pscore", "score", "hpscore", "hscore"), tags = "train"),
        stepsize.factor = p_dbl(default = 1, tags = "train"),
        sf.scheme = p_fct(default = "sigmoid", levels = c("sigmoid", "linear"), tags = "train"),
        pendistmat = p_uty(tags = "train"),
        connected.index = p_uty(tags = "train"),
        x.is.01 = p_lgl(default = FALSE, tags = "train"),
        return.score = p_lgl(default = TRUE, tags = "train"),
        trace = p_lgl(default = FALSE, tags = "train"),
        at.step = p_uty(tags = "predict"),
        penalize_clinical = p_lgl(default = FALSE, tags = "train"),
        stratify_by_event = p_lgl(default = TRUE, tags = "train")
      )

      super$initialize(
        # see the mlr3book for a description: https://mlr3book.mlr-org.com/extending-mlr3.html
        id = "surv.cv_coxboost_custom",
        packages = c("mlr3extralearners", "CoxBoost", "pracma"),
        feature_types = c("integer", "numeric"),
        predict_types = c("distr", "crank", "lp"),
        param_set = ps,
        properties = "weights"
      )
    }
  ),
  private = list(
    .train = function(task) {

      # set column names to ensure consistency in fit and predict
      self$state$feature_names <- task$feature_names
      opt_pars <- self$param_set$get_values(tags = "optimPenalty")
      cv_pars <- self$param_set$get_values(tags = "cvpars")
      cox_pars <- setdiff(
        self$param_set$get_values(tags = "train"),
        c(opt_pars, cv_pars)
      )

      if ("weights" %in% task$properties) {
        cox_pars$weights <- as.numeric(task$weights$weight)
      }

      data <- task$data()
      tn <- task$target_names
      time <- data[[tn[1L]]]
      status <- data[[tn[2L]]]
      if (grepl("penalize_clinical", names(self$param_set))) {
        if (self$param_set$penalize_clinical == FALSE) {
          unpenalized_indices = grep("clinical", colnames(data[, !tn, with = FALSE]))
        } else {
          unpenalized_indices = NULL
        }
        self$param_set = self$param_set[-grep("penalize_clinical", names(self$param_set))]
      }

      if (grepl("stratify_by_event", names(self$param_set))) {
        if (self$param_set$stratify_by_event == TRUE) {
          folds = create_folds(status, k = self$param_set$K, type = "stratified", invert = TRUE)
          folds = sapply(1:length(status), function(x) unname(which(sapply(folds, function(y) x %in% y))))
          self$param_set = self$param_set[-grep("K", names(self$param_set))]
        } else {
          folds = NULL
        }
        self$param_set = self$param_set[-grep("stratify_by_event", names(self$param_set))]
      }

      data <- as.matrix(data[, !tn, with = FALSE])

      pen_optim <- FALSE
      if (!is.null(opt_pars$penalty)) {
        if (opt_pars$penalty == "optimCoxBoostPenalty") {
          pen_optim <- TRUE
          opt_pars$penalty <- NULL
        }
      } else {
        cv_pars <- insert_named(cv_pars, list(penalty = NULL))
      }

      with_package("CoxBoost", {
        if (pen_optim) {
          optim <- mlr3misc::invoke(
            CoxBoost::optimCoxBoostPenalty,
            time = time,
            status = status,
            x = data,
            folds = folds,
            unpen.index = unpenalized_indices,
            .args = c(opt_pars, cv_pars)
          )

          return(mlr3misc::invoke(
            CoxBoost::CoxBoost,
            time = time,
            status = status,
            x = data,
            stepno = optim$cv.res$optimal.step,
            penalty = optim$penalty,
            .args = cox_pars
          ))
        } else {
          optimal_step <- mlr3misc::invoke(
            CoxBoost::cv.CoxBoost,
            time = time,
            status = status,
            x = data,
            .args = c(cv_pars, cox_pars)
          )$optimal.step

          return(mlr3misc::invoke(
            CoxBoost::CoxBoost,
            time = time,
            status = status,
            x = data,
            stepno = optimal_step,
            .args = cox_pars
          ))
        }
      })
    },
    .predict = function(task) {
      pars <- self$param_set$get_values(tags = "predict")

      # get newdata and ensure same ordering in train and predict
      newdata <- as.matrix(task$data(cols = self$state$feature_names))

      lp <- as.numeric(mlr3misc::invoke(predict,
        self$model,
        newdata = newdata,
        .args = pars,
        type = "lp"
      ))

      surv <- mlr3misc::invoke(predict,
        self$model,
        newdata = newdata,
        .args = pars,
        type = "risk",
        times = sort(unique(self$model$time))
      )

      mlr3proba::.surv_return(
        times = sort(unique(self$model$time)),
        surv = surv,
        lp = lp
      )
    }
  )
)

mlr_learners$add("surv.cv_coxboost_custom", LearnerSurvCVCoxboostCustom)
