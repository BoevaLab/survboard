suppressPackageStartupMessages({
  library(R6)
  library(mlr3)
  library(mlr3proba)
  library(mlr3tuningspaces)
})

# Adapted from: https://github.com/mlr-org/mlr3extralearners/blob/main/R/learner_glmnet_surv_cv_glmnet.R

#' Fits cv.glmnet using `mlr3` and `mlr3proba`. Identical to `LearnerSurvCVGlmnet`
#' but using stratified CV internally and the option to favor clinical variables. Refer to `LearnerSurvCVGlmnet`
#' and `cv.glmnet` for detailed documentation.
LearnerSurvCVGlmnetCustom <- R6Class("LearnerSurvCVGlmnetCustom",
  inherit = mlr3proba::LearnerSurv,
  public = list(
    initialize = function() {
      ps <- ps(
        s = p_fct(c("lambda.1se", "lambda.min"), default = "lambda.min", tags = "predict"),
        standardize = p_lgl(default = TRUE, tags = "train"),
        favor_clinical = p_lgl(default = FALSE, tags = "train"),
        nfolds = p_int(3, 10, default = 5, tags = "train"),
        alpha = p_dbl(0, 1, default = 1, tags = "train"),
	nlambda = p_int(5, 100, default = 100, tags = "train")
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
      suppressPackageStartupMessages({
        library(mlr3misc)
        library(coefplot)
        source(here::here("survboard", "R", "utils", "utils.R"))
        source(here::here("survboard", "R", "utils", "imports.R"))
      })

      data <- as_numeric_matrix(task$data(cols = task$feature_names))
      target <- task$truth()
      pv <- self$param_set$get_values(tags = "train")
      foldids_formatted <- get_folds(target[, 2], pv$nfolds, nrow(data))
      pv$foldid <- foldids_formatted
      pv <- pv[-which(names(pv) == "nfolds")]
      pv$family <- "cox"
      penalty.factor <- rep(1, length(task$feature_names))

      # Favor clinical variables by setting their penalty factors to zero.
      # Unused by us in practice.
      if (pv$favor_clinical) {
        penalty.factor[which(sapply(strsplit(task$feature_names, "\\_"), function(x) x[[1]]) == "clinical")] <- 0
        pv <- pv[-which(names(pv) == "favor_clinical")]
      }

      pv$penalty.factor <- penalty.factor
      glmnet_fit <- mlr3misc::invoke(glmnet::cv.glmnet,
        data, target,
        .args = pv
      )
      tmp <- coefplot::extract.coef(glmnet_fit)
      coefficients <- tmp[, 1]
      names(coefficients) <- rownames(tmp)
      cox_helper <- transform_cox_model(coefficients, data, target)
      return(cox_helper)
    },
    .predict = function(task) {
      suppressPackageStartupMessages({
        source(here::here("survboard", "R", "utils", "utils.R"))
        source(here::here("survboard", "R", "utils", "imports.R"))
        library(pec)
        library(mlr3proba)
      })
      newdata <- as_numeric_matrix(ordered_features(task, self))
      newdata <- data.frame(newdata)[, colnames(newdata) %in% names(self$model$coefficients), drop = FALSE]
      surv <- pec::predictSurvProb(self$model, newdata, self$model$y[, 1])
      lp <- predict(self$model, newdata)
      return(mlr3proba::.surv_return(
        times = self$model$y[, 1],
        surv = surv,
        lp = lp
      ))
    }
  )
)

mlr_learners$add("surv.cv_glmnet_custom", LearnerSurvCVGlmnetCustom)
