library(survival)
library(glmnet)
library(coefplot)
library(pec)

# Adapted from LearnerSurvCV: https://mlr3learners.mlr-org.com/reference/mlr_learners_surv.glmnet.html
LearnerSurvCVGlmnetCustom <- R6Class("LearnerSurvCVGlmnetCustom",
  inherit = mlr3proba::LearnerSurv,
  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ps(
        alignment            = p_fct(c("lambda", "fraction"), default = "lambda", tags = "train"),
        alpha                = p_dbl(0, 1, default = 1, tags = "train"),
        big                  = p_dbl(default = 9.9e35, tags = "train"),
        devmax               = p_dbl(0, 1, default = 0.999, tags = "train"),
        dfmax                = p_int(0L, tags = "train"),
        eps                  = p_dbl(0, 1, default = 1.0e-6, tags = "train"),
        epsnr                = p_dbl(0, 1, default = 1.0e-8, tags = "train"),
        exclude              = p_uty(tags = "train"),
        exmx                 = p_dbl(default = 250.0, tags = "train"),
        fdev                 = p_dbl(0, 1, default = 1.0e-5, tags = "train"),
        foldid               = p_uty(default = NULL, tags = "train"),
        gamma                = p_uty(tags = "train"),
        grouped              = p_lgl(default = TRUE, tags = "train"),
        intercept            = p_lgl(default = TRUE, tags = "train"),
        keep                 = p_lgl(default = FALSE, tags = "train"),
        lambda               = p_uty(tags = "train"),
        lambda.min.ratio     = p_dbl(0, 1, tags = "train"),
        lower.limits         = p_uty(default = -Inf, tags = "train"),
        maxit                = p_int(1L, default = 100000L, tags = "train"),
        mnlam                = p_int(1L, default = 5L, tags = "train"),
        mxit                 = p_int(1L, default = 100L, tags = "train"),
        mxitnr               = p_int(1L, default = 25L, tags = "train"),
        nfolds               = p_int(3L, default = 10L, tags = "train"),
        nlambda              = p_int(1L, default = 100L, tags = "train"),
        offset               = p_uty(default = NULL, tags = "train"),
        parallel             = p_lgl(default = FALSE, tags = "train"),
        penalty.factor       = p_uty(tags = "train"),
        pmax                 = p_int(0L, tags = "train"),
        pmin                 = p_dbl(0, 1, default = 1.0e-9, tags = "train"),
        prec                 = p_dbl(default = 1e-10, tags = "train"),
        predict.gamma        = p_dbl(default = "gamma.1se", special_vals = list("gamma.1se", "gamma.min"), tags = "predict"),
        relax                = p_lgl(default = FALSE, tags = "train"),
        s                    = p_dbl(0, 1, special_vals = list("lambda.1se", "lambda.min"), default = "lambda.1se", tags = "predict"),
        standardize          = p_lgl(default = TRUE, tags = "train"),
        standardize.response = p_lgl(default = FALSE, tags = "train"),
        thresh               = p_dbl(0, default = 1e-07, tags = "train"),
        trace.it             = p_int(0, 1, default = 0, tags = "train"),
        type.gaussian        = p_fct(c("covariance", "naive"), tags = "train"),
        type.logistic        = p_fct(c("Newton", "modified.Newton"), default = "Newton", tags = "train"),
        type.measure         = p_fct(c("deviance", "C"), default = "deviance", tags = "train"),
        type.multinomial     = p_fct(c("ungrouped", "grouped"), default = "ungrouped", tags = "train"),
        upper.limits         = p_uty(default = Inf, tags = "train"),
        stratify_by_event = p_lgl(default = TRUE, tags = "train"),
        penalize_clinical = p_lgl(default = FALSE, tags = "train")
      )

      super$initialize(
        id = "surv.cv_glmnet_custom",
        param_set = ps,
        feature_types = c("logical", "integer", "numeric"),
        predict_types = c("crank", "lp", "distr"),
        properties = c("weights", "selected_features"),
        packages = c("mlr3learners", "glmnet")
      )
    },

    #' @description
    #' Returns the set of selected features as reported by [glmnet::predict.glmnet()]
    #' with `type` set to `"nonzero"`.
    #'
    #' @param lambda (`numeric(1)`)\cr
    #' Custom `lambda`, defaults to the active lambda depending on parameter set.
    #'
    #' @return (`character()`) of feature names.
    selected_features = function(lambda = NULL) {
      glmnet_selected_features <- function(self, lambda = NULL) {
        if (is.null(self$model)) {
          stopf("No model stored")
        }

        assert_number(lambda, null.ok = TRUE, lower = 0)
        lambda <- lambda %??% glmnet_get_lambda(self)
        nonzero <- predict(self$model, type = "nonzero", s = lambda)
        if (is.data.frame(nonzero)) {
          nonzero <- nonzero[[1L]]
        } else {
          nonzero <- unlist(map(nonzero, 1L), use.names = FALSE)
          nonzero <- if (length(nonzero)) sort(unique(nonzero)) else integer()
        }

        glmnet_feature_names(self$model)[nonzero]
      }

      glmnet_selected_features(self, lambda)
    }
  ),
  private = list(
    .train = function(task) {
      data <- as_numeric_matrix(task$data(cols = task$feature_names))
      target <- task$truth()
      pv <- self$param_set$get_values(tags = "train")
      pv$family <- "cox"
      if ("weights" %in% task$properties) {
        pv$weights <- task$weights$weight
      }
      if (grepl("penalize_clinical", names(self$param_set))) {
        if (self$param_set$penalize_clinical == FALSE) {
          unpenalized_indices = grep("clinical", colnames(data))
          penalty_factor = rep(1, ncol(data))
          penalty_factor[unpenalized_indices] = 0
        } else {
          unpenalized_indices = rep(1, ncol(data))
        }
        self$param_set = self$param_set[-grep("penalize_clinical", names(self$param_set))]
      }
      
      if (grepl("stratify_by_event", names(self$param_set))) {
        if (self$param_set$stratify_by_event == TRUE) {
          folds = create_folds(target[, 2], k = self$param_set$nfolds, type = "stratified", invert = TRUE)
          folds = sapply(1:length(target[, 2]), function(x) unname(which(sapply(folds, function(y) x %in% y))))
          self$param_set = self$param_set[-grep("nfolds", names(self$param_set))]
        } else {
          folds = NULL
        }
        self$param_set = self$param_set[-grep("stratify_by_event", names(self$param_set))]
      }
      list(
        mlr3misc::invoke(
          glmnet::cv.glmnet,
          x = data,
          y = target,
          foldid = folds,
          penalty.factor = penalty_factor,
          .args = pv
        ),
        data,
        target
      )
    },
    .predict = function(task) {
      newdata <- as_numeric_matrix(ordered_features(task, self))
      pv <- self$param_set$get_values(tags = "predict")
      pv <- rename(pv, "predict.gamma", "gamma")

      time <- self$model[[3]][, 1]
      status <- self$model[[3]][, 2]
      y_train <- cbind(time, status, )
      cox_helper <- coxph(
        Surv(time, status) ~ .,
        data = cbind(y_train, data.frame(X_train[, colnames(X_train) %in% selected_variables])),
        init = extract.coef(learner, lambda = "lambda.min")[, 1],
        iter.max = 0,
        x = TRUE,
        ties = "breslow"
      )

      survival_probabilities <- pec::predictSurvProb(cox_helper,
        newdata = data.frame(X_test[, colnames(X_train) %in% selected_variables]),
        times = time
      )

      lp <- as.numeric(mlr3misc::invoke(predict, self$model, newx = newdata, type = "link", .args = pv))
      mlr3proba::.surv_return(
        times = time,
        surv = survival_probabilities,
        crank = lp,
        lp = lp
      )
    }
  )
)

mlr_learners$add("surv.cv_glmnet_custom", LearnerSurvCVGlmnetCustom)

