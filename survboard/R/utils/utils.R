# Most of the below adapted from various mlr3 and mlr3proba utils
# https://github.com/mlr-org/mlr3proba/issues
# https://github.com/mlr-org/mlr3

# p = probability for levs[2] => matrix with probs for levs[1] and levs[2]
pvec2mat <- function(p, levs) {
  stopifnot(is.numeric(p))
  y <- matrix(c(1 - p, p), ncol = 2L, nrow = length(p))
  colnames(y) <- levs
  y
}


ordered_features <- function(task, learner) {
  cols <- names(learner$state$data_prototype)
  task$data(cols = intersect(cols, task$feature_names))
}


as_numeric_matrix <- function(x) { # for svm / #181
  x <- as.matrix(x)
  if (is.logical(x)) {
    storage.mode(x) <- "double"
  }
  x
}


swap_levels <- function(x) {
  factor(x, levels = rev(levels(x)))
}


rename <- function(x, old, new) {
  if (length(x)) {
    ii <- match(names(x), old, nomatch = 0L)
    names(x)[ii > 0L] <- new[ii]
  }
  x
}


extract_loglik <- function(self) {
  require_namespaces(self$packages)
  if (is.null(self$model)) {
    stopf("Learner '%s' has no model stored", self$id)
  }
  stats::logLik(self$model)
}


opts_default_contrasts <- list(contrasts = c("contr.treatment", "contr.poly"))



glmnet_get_lambda <- function(self, pv) {
  if (is.null(self$model)) {
    stopf("Learner '%s' has no model stored", self$id)
  }

  pv <- pv %??% self$param_set$get_values(tags = "predict")
  s <- pv$s

  if (is.character(s)) {
    self$model[[s]]
  } else if (is.numeric(s)) {
    s
  } else { # null / missing
    if (inherits(self$model, "cv.glmnet")) {
      self$model[["lambda.1se"]]
    } else if (length(self$model$lambda) == 1L) {
      self$model$lambda
    } else {
      default <- self$param_set$default$s
      warningf("Multiple lambdas have been fit. Lambda will be set to %s (see parameter 's').", default)
      default
    }
  }
}


glmnet_feature_names <- function(model) {
  beta <- model$beta
  if (is.null(beta)) {
    beta <- model$glmnet.fit$beta
  }

  rownames(if (is.list(beta)) beta[[1L]] else beta)
}


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


glmnet_invoke <- function(data, target, pv, cv = FALSE) {
  library(mlr3misc)
  saved_ctrl <- glmnet::glmnet.control()
  on.exit(mlr3misc::invoke(glmnet::glmnet.control, .args = saved_ctrl))
  glmnet::glmnet.control(factory = TRUE)
  is_ctrl_pars <- names(pv) %in% names(saved_ctrl)

  if (any(is_ctrl_pars)) {
    mlr3misc::invoke(glmnet::glmnet.control, .args = pv[is_ctrl_pars])
    pv <- pv[!is_ctrl_pars]
  }

  mlr3misc::invoke(
    if (cv) glmnet::cv.glmnet else glmnet::glmnet,
    x = data, y = target, .args = pv
  )
}

#' @title Convert a Ratio Hyperparameter
#'
#' @description
#' Given the named list `pv` (values of a [ParamSet]), converts a possibly provided hyperparameter
#' called `ratio` to an integer hyperparameter `target`.
#' If both are found in `pv`, an exception is thrown.
#'
#' @param pv (named `list()`).
#' @param target (`character(1)`)\cr
#'   Name of the integer hyperparameter.
#' @param ratio (`character(1)`)\cr
#'   Name of the ratio hyperparameter.
#' @param n (`integer(1)`)\cr
#'   Ratio of what?
#'
#' @return (named `list()`) with new hyperparameter settings.
#' @noRd
convert_ratio <- function(pv, target, ratio, n) {
  library(mlr3misc)
  switch(mlr3misc::to_decimal(c(target, ratio) %in% names(pv)) + 1L,
    # !mtry && !mtry.ratio
    pv,

    # !mtry && mtry.ratio
    {
      pv[[target]] <- max(ceiling(pv[[ratio]] * n), 1)
      remove_named(pv, ratio)
    },


    # mtry && !mtry.ratio
    pv,

    # mtry && mtry.ratio
    stopf("Hyperparameters '%s' and '%s' are mutually exclusive", target, ratio)
  )
}


get_survival_prediction_linear_cox <- function(train_target, train_data, coefficients, newdata) {
  library(survival)
  library(pec)
  if (!all(coefficients == 0)) {
    train_data <- train_data[, which(colnames(train_data) %in% names(coefficients))]
    newdata <- newdata[, which(colnames(newdata) %in% names(coefficients))]
  }

  train_matrix <- cbind(train_target, train_data)
  colnames(train_matrix)[1:2] <- c("OS_days", "OS")
  cox_helper <- coxph(Surv(OS_days, OS) ~ ., x = TRUE, init = coefficients, iter.max = 0, data = data.frame(train_matrix))
  surv <- pec::predictSurvProb(cox_helper, data.frame(newdata), train_target[, 1])
  return(surv)
}

get_prioritylasso_block_order <- function(target, data, blocks, foldid, lambda.type, favor_clinical) {
  library(glmnet)
  library(coefplot)
  mean_absolute_coefficients <- c()
  for (i in 1:length(blocks)) {
    tmp <- cv.glmnet(
      data[
        ,
        which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == blocks[i])
      ],
      y = target,
      foldid = foldid,
      type.measure = "deviance",
      family = "cox",
      alpha = 0
    )
    mean_absolute_coefficients <- c(mean_absolute_coefficients, mean(abs(extract.coef(tmp, lambda.type)[, 1])))
  }
  block_order <- blocks[sort(mean_absolute_coefficients, index.return = TRUE, decreasing = TRUE)$ix]
  if (favor_clinical) {
    block_order <- c(
      "clinical",
      block_order[-which(names(block_order) == "clinical")]
    )
  }
  return(block_order)
}
