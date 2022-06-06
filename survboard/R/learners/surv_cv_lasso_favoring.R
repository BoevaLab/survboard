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
LearnerSurvCVGlmnetFavoring <- R6Class("LearnerSurvCVGlmnetFavoring",
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
        upper.limits         = p_uty(default = Inf, tags = "train")
      )

      super$initialize(
        id = "surv.cv_glmnet",
        param_set = ps,
        feature_types = c("logical", "integer", "numeric"),
        predict_types = c("crank", "lp", "distr"),
        packages = c("mlr3learners", "glmnet"),
      )
    }
  ),
  private = list(
    .train = function(task) {
      data <- as_numeric_matrix(task$data(cols = task$feature_names))
      target <- task$truth()
      
      pv <- self$param_set$get_values(tags = "train")
      pv$family <- "cox"
      penalty.factor <- rep(1, length(task$feature_names))
      penalty.factor[grep("clinical", task$feature_names)] <- 0
      pv$penalty.factor <- penalty.factor

      glmnet_invoke(data, target, pv, cv = TRUE)
    },
    .predict = function(task) {
      browser()
      newdata <- as_numeric_matrix(ordered_features(task, self))
      train_data <- task$backend$data(which(!task$backend$rownames %in% task$row_roles$use), task$feature_names)
      train_target <- task$backend$data(which(!task$backend$rownames %in% task$row_roles$use ), c("time", "status"))
      pv <- self$param_set$get_values(tags = "predict")
      pv <- rename(pv, "predict.gamma", "gamma")
      lp <- as.numeric(invoke(predict, self$model, newx = newdata, type = "link", .args = pv))
      surv <- get_survival_prediction_linear_cox(
        train_target$time,
        train_target$status,
        as.numeric(invoke(predict, self$model, newx = train_data, type = "link", .args = pv)),
        lp
      )
      .surv_return(
        times = train_target$times,
        surv = surv,
        crank = crank,
        lp = lp
      )
    }
  )
)

get_survival_prediction_linear_cox <- function(t, delta, f.x_train, f.x_test) {
  cumulative_baseline_hazard <- basehaz.gbm_extrapolate(t, delta, f.x_train, cumulative = TRUE)
  survival_function <- exp(-matrix(rep(f.x_test, length(cumulative_baseline_hazard)), byrow = FALSE, nrow = length(f.x_test)) * matrix(rep(cumulative_baseline_hazard, length(f.x_test)), nrow = length(f.x_test), ncol = length(cumulative_baseline_hazard), byrow = TRUE))
  return(survival_function)
}

ordered_features = function(task, learner) {
  cols = names(learner$state$data_prototype)
  task$data(cols = intersect(cols, task$feature_names))
}


as_numeric_matrix = function(x) { # for svm / #181
  x = as.matrix(x)
  if (is.logical(x)) {
    storage.mode(x) = "double"
  }
  x
}


glmnet_invoke = function(data, target, pv, cv = FALSE) {
  saved_ctrl = glmnet::glmnet.control()
  on.exit(invoke(glmnet::glmnet.control, .args = saved_ctrl))
  glmnet::glmnet.control(factory = TRUE)
  is_ctrl_pars = names(pv) %in% names(saved_ctrl)
  
  if (any(is_ctrl_pars)) {
    invoke(glmnet::glmnet.control, .args = pv[is_ctrl_pars])
    pv = pv[!is_ctrl_pars]
  }
  
  invoke(
    if (cv) glmnet::cv.glmnet else glmnet::glmnet,
    x = data, y = target, .args = pv
  )
}


mlr_learners$add("surv.checking", LearnerSurvCVGlmnetFavoring)

task = mlr3proba::as_task_surv(
  rats[, 1:4]  ,
  time = "time",
  event = "status"
)

lrn = lrn("surv.checking")
lrn$train(task)
resamplings = rsmp("cv", folds = 2)
design = benchmark_grid(task, lrn, resamplings)
bmr = benchmark(design)



basehaz.gbm_extrapolate <- function(t, delta, f.x, t.eval = NULL, smooth = FALSE,
                                    cumulative = TRUE) {
  t.unique <- sort(unique(t[delta == 1]))
  alpha <- length(t.unique)
  for (i in 1:length(t.unique)) {
    alpha[i] <- sum(t[delta == 1] == t.unique[i]) /
      sum(exp(f.x[t >= t.unique[i]]))
  }

  if (!smooth && !cumulative) {
    if (!is.null(t.eval)) {
      stop("Cannot evaluate unsmoothed baseline hazard at t.eval.")
    }
  } else {
    if (smooth && !cumulative) {
      lambda.smooth <- supsmu(t.unique, alpha)
    } else {
      if (smooth && cumulative) {
        lambda.smooth <- supsmu(t.unique, cumsum(alpha))
      } else { # (!smooth && cumulative) - THE DEFAULT
        lambda.smooth <- list(x = t.unique, y = cumsum(alpha))
      }
    }
  }


  obj <- if (!is.null(t.eval)) {
    approx(lambda.smooth$x, lambda.smooth$y, xout = t.eval, rule = 2)$y
  } else {
    approx(lambda.smooth$x, lambda.smooth$y, xout = t, rule = 2)$y
  }

  return(obj)
}
