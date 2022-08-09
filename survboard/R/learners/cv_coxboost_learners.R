suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3proba)
  library(R6)
})

# Adapted from: https://github.com/mlr-org/mlr3extralearners/blob/main/R/learner_CoxBoost_surv_cv_coxboost.R

#' Fits CoxBoost using `mlr3` and `mlr3proba`. Identical to `LearnerSurvCVCoxboost`
#' but using stratified CV internally. Refer to `LearnerSurvCVCoxboost`
#' and `CoxBoost` for detailed documentation.
LearnerSurvCVCoxboostCustom <- R6Class("LearnerSurvCVCoxboostCustom",
  inherit = mlr3proba::LearnerSurv,
  public = list(
    initialize = function() {
      ps <- ps(
        favor_clinical = p_lgl(default = FALSE, tags = "train"),
        maxstepno = p_int(default = 100, lower = 0, tags = "train"),
        K = p_int(default = 10, lower = 2, tags = "train"),
        type = p_fct(default = "verweij", levels = c("verweij", "naive"), tags = "train"),
        folds = p_uty(default = NULL, tags = "train"),
        minstepno = p_int(default = 50, lower = 0, tags = "train"),
        start.penalty = p_dbl(tags = "train"),
        iter.max = p_int(default = 10, lower = 1, tags = "train"),
        upper.margin = p_dbl(default = 0.05, lower = 0, upper = 1, tags = "train"),
        standardize = p_lgl(default = TRUE, tags = "train"),
        penalty = p_dbl(special_vals = list("optimCoxBoostPenalty"), tags = "train"),
        criterion = p_fct(default = "pscore", levels = c("pscore", "score", "hpscore", "hscore"), tags = "train"),
        stepsize.factor = p_dbl(default = 1, tags = "train"),
        sf.scheme = p_fct(default = "sigmoid", levels = c("sigmoid", "linear"), tags = "train"),
        pendistmat = p_uty(tags = "train"),
        connected.index = p_uty(tags = "train"),
        x.is.01 = p_lgl(default = FALSE, tags = "train"),
        return.score = p_lgl(default = TRUE, tags = "train"),
        trace = p_lgl(default = FALSE, tags = "train"),
        at.step = p_uty(tags = "predict")
      )

      super$initialize(
        id = "surv.cv_coxboost_custom",
        packages = c("mlr3extralearners", "CoxBoost", "pracma"),
        feature_types = c("integer", "numeric"),
        predict_types = c("distr", "crank", "lp"),
        param_set = ps
      )
    }
  ),
  private = list(
    .train = function(task) {
      suppressPackageStartupMessages({
        source(here::here("survboard", "R", "utils", "utils.R"))
        source(here::here("survboard", "R", "utils", "imports.R"))
        library(withr)
        library(splitTools)
      })

      self$state$feature_names <- task$feature_names
      pars <- self$param_set$get_values(tags = "train")

      optim_args <- c("minstepno", "start.penalty", "iter.max", "upper.margin", "penalty")
      cv_args <- c("maxstepno", "type", "folds")
      `%nin%` <- Negate(`%in%`)
      nfolds <- pars$K

      if (pars$favor_clinical) {
        pars$unpen.index <- grep("clinical", task$feature_names)
      }

      pars <- pars[-grep("favor_clinical", names(pars), fixed = TRUE)]
      pars <- pars[-grep("K", names(pars), fixed = TRUE)]
      opt_pars <- pars[names(pars) %in% optim_args]
      cv_pars <- pars[names(pars) %in% cv_args]
      cox_pars <- pars[names(pars) %nin% c(names(opt_pars), names(cv_pars))]

      data <- task$data()
      tn <- task$target_names
      time <- data[[tn[1L]]]
      status <- data[[tn[2L]]]
      # Create foldids for reproducibility using stratification on the event.
      foldids <- create_folds(status, k = nfolds, invert = TRUE, type = "stratified")
      cv_pars[["folds"]] <- foldids
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
            .args = c(opt_pars, cv_pars, cox_pars),
            K = nfolds
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
            .args = c(cv_pars, cox_pars),
            K = nfolds
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
      suppressPackageStartupMessages({
        source(here::here("survboard", "R", "utils", "utils.R"))
        source(here::here("survboard", "R", "utils", "imports.R"))
      })
      pars <- self$param_set$get_values(tags = "predict")
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

      return(mlr3proba::.surv_return(
        times = sort(unique(self$model$time)),
        surv = surv,
        lp = lp
      ))
    }
  )
)

mlr_learners$add("surv.cv_coxboost_custom", LearnerSurvCVCoxboostCustom)
