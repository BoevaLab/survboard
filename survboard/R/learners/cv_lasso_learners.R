library(R6)
library(mlr3)
library(mlr3proba)
library(mlr3tuningspaces)

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
LearnerSurvCVGlmnetCustom <- R6Class("LearnerSurvCVGlmnetCustom",
  inherit = mlr3proba::LearnerSurv,
  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ps(
        s                    = p_fct(c("lambda.1se", "lambda.min"), default = "lambda.min", tags = "predict"),
        standardize          = p_lgl(default = TRUE, tags = "train"),
        favor_clinical = p_lgl(default = FALSE, tags = "train"),
        nfolds = p_int(3, 10, default = 5, tags = "train")
      )

      super$initialize(
        id = "surv.cv_glmnet_custom",
        param_set = ps,
        feature_types = c("logical", "integer", "numeric"),
        predict_types = c("distr", "crank", "lp"),
        packages = c("mlr3learners", "glmnet"),
      )
    }
  ),
  private = list(
    .train = function(task) {
      library(splitTools)
      source(here::here("survival-benchmark", "R", "utils", "utils.R"))
      data <- as_numeric_matrix(task$data(cols = task$feature_names))
      target <- task$truth()
      pv = self$param_set$get_values(tags = "train")
      foldids <- create_folds(target[, 2], k = pv$nfolds, invert = TRUE, type = "stratified")
      foldids_formatted <- rep(1, nrow(data))
      for (i in 2:length(foldids)) {
        foldids_formatted[foldids[[i]]] <- i
      }
      
      pv$foldid <- foldids_formatted
      pv <- pv[-grep("nfolds", names(pv))]
      pv$family <- "cox"
      penalty.factor <- rep(1, length(task$feature_names))
      if (pv$favor_clinical) {
        penalty.factor[grep("clinical", task$feature_names)] <- 0
        pv <- pv[-grep("favor_clinical", names(pv))]
      }
      
      pv$penalty.factor <- penalty.factor

      list(glmnet_invoke(data, target, pv, cv = TRUE), data, target)
    },
    .predict = function(task) {
      library(coefplot)
      source(here::here("survival-benchmark", "R", "utils", "utils.R"))
      model <- self$model[[1]]
      train_data <- self$model[[2]]
      train_target <- self$model[[3]]
      newdata <- as_numeric_matrix(ordered_features(task, self))
      pv <- self$param_set$get_values(tags = "predict")
      pv <- rename(pv, "predict.gamma", "gamma")
      if (unname(model$nzero[which.min(model$cvlo)]) == 0) {
        lp <- rep(0, nrow(newdata))
        coefficients <- rep(0, ncol(newdata))
      }
      else {
        lp <- as.numeric(invoke(predict, model, newx = newdata, type = "link", .args = pv))
        coefficients <- setNames(extract.coef(model)[, 1], rownames(extract.coef(model)))
      }
      surv <- get_survival_prediction_linear_cox(
        train_target,
        train_data,
        coefficients,
        newdata
      )
      mlr3proba::.surv_return(
        times = train_target[, 1],
        surv = surv,
        lp = lp
      )
    }
  )
)

mlr_learners$add("surv.cv_glmnet_custom", LearnerSurvCVGlmnetCustom)