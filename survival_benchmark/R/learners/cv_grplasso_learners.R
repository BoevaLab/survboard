library(R6)
library(mlr3)
library(mlr3proba)
library(mlr3tuningspaces)
library(grpreg)

#' @title Cross-Validated  GLM with Elastic Net Regularization Survival Learner
#'
#' @name mlr_learners_surv.cv_glmnet
#'
#' @description
#' Generalized linear models with elastic net regularization.
#' Calls [glmnet::cv.glmnet()] from package \CRANpkg{glmnet}.
#'
#' The default for hyperparameter `family` is set to `"cox"`.
#'
#' @templateVar id surv.cv_glmnet
#' @template learner
#'
#' @references
#' `r format_bib("friedman_2010")`
#'
#' @export
#' @template seealso_learner
#' @template example
LearnerSurvCVGrpLasso <- R6Class("LearnerSurvCVGrpLasso",
  inherit = mlr3proba::LearnerSurv,
  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ps(
        nfolds = p_int(3, 10, default = 5, tags = "train"),
        favor_clinical = p_lgl(default = FALSE, tags = "train")
      )

      super$initialize(
        id = "surv.cv_grplasso",
        param_set = ps,
        feature_types = c("logical", "integer", "numeric"),
        predict_types = c("lp", "distr"),
        packages = c("mlr3learners", "grplasso"),
      )
    }
  ),
  private = list(
    .train = function(task) {
      #browser()
      library(splitTools)
      source(here::here("survival-benchmark", "R", "utils", "utils.R"))
      data <- as_numeric_matrix(task$data(cols = task$feature_names))
      target <- task$truth()
      pv <- self$param_set$get_values(tags = "train")
      foldids <- create_folds(target[, 2], k = pv$nfolds, invert = TRUE, type = "stratified")
      foldids_formatted <- rep(1, nrow(data))
      for (i in 2:length(foldids)) {
        foldids_formatted[foldids[[i]]] <- i
      }
      pv <- pv[-grep("nfolds", names(pv))]
      pv$fold <- foldids_formatted

      block_order <- c(
        "clinical",
        "gex",
        "cnv",
        "rppa",
        "mirna",
        "mut",
        "meth"
      )

      blocks <- sapply(block_order, function(x) grep(x, task$feature_names))

      blocks <- blocks[sapply(blocks, length) > 1]
      groups <- rep(1, length(task$feature_names))
      for (i in 1:length(blocks)) {
        groups[blocks[[i]]] <- i
      }

      list(mlr3misc::invoke(
        cv.grpsurv,
        X = data,
        y = target,
        group = groups - as.numeric(pv$favor_clinical),
        fold = foldids_formatted
      ), data, target)
    },
    .predict = function(task) {
      browser()
      source(here::here("survival-benchmark", "R", "utils", "utils.R"))
      model <- self$model[[1]]
      newdata <- as_numeric_matrix(ordered_features(task, self))
      train_data <- self$model[[2]]
      train_target <- self$model[[3]]
      lp <- as.numeric(predict(model$fit, newdata, type = "link", which = which.min(model$cve)))
      surv <- get_survival_prediction_linear_cox(
        train_target[, 1],
        train_target[, 2],
        as.numeric(predict(model$fit, train_data, type = "link", which = which.min(model$cve))),
        lp
      )
      .surv_return(
        times = train_target[, 1],
        surv = surv,
        lp = lp
      )
    }
  )
)

mlr_learners$add("surv.cv_grplasso", LearnerSurvCVGrpLasso)