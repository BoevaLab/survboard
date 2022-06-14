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
        s = p_fct(c("lambda.1se", "lambda.min"), default = "lambda.min", tags = "predict"),
        standardize = p_lgl(default = TRUE, tags = "train"),
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
      library(mlr3misc)
      source(here::here("survival_benchmark", "R", "utils", "utils.R"))
      data <- as_numeric_matrix(task$data(cols = task$feature_names))
      target <- task$truth()
      pv <- self$param_set$get_values(tags = "train")
      foldids <- create_folds(target[, 2], k = pv$nfolds, invert = TRUE, type = "stratified")
      foldids_formatted <- rep(1, nrow(data))
      for (i in 2:length(foldids)) {
        foldids_formatted[foldids[[i]]] <- i
      }

      pv$foldid <- foldids_formatted
      pv <- pv[-which(names(pv) == "nfolds")]
      pv$family <- "cox"
      penalty.factor <- rep(1, length(task$feature_names))
      if (pv$favor_clinical) {
        penalty.factor[which(sapply(strsplit(task$feature_names, "\\_"), function(x) x[[1]]) == "clinical")] <- 0
        pv <- pv[-which(names(pv) == "favor_clinical")]
      }

      pv$penalty.factor <- penalty.factor
      glmnet_fit <- mlr3misc::invoke(glmnet::cv.glmnet,
        data, target,
        .args = pv
      )
      tmp <- extract.coef(glmnet_fit)
      coefficients <- tmp[, 1]
      names(coefficients) <- rownames(tmp)
      browser()
      cox_helper <- transform_cox_model(coefficients, data, target)
      cox_helper
    },
    .predict = function(task) {
      browser()
      source(here::here("survival_benchmark", "R", "utils", "utils.R"))
      newdata <- as_numeric_matrix(ordered_features(task, self))
      newdata <- data.frame(newdata)[, colnames(newdata) %in% names(self$model$coefficients)]
      surv <- pec::predictSurvProb(self$model, newdata, self$model$y[, 1])
      lp <- predict(self$model, newdata)
      mlr3proba::.surv_return(
        times = self$model$y[, 1],
        surv = surv,
        lp = lp
      )
    }
  )
)

mlr_learners$add("surv.cv_glmnet_custom", LearnerSurvCVGlmnetCustom)
