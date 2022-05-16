library(survival)
library(SGL)
library(coefplot)
library(pec)

# Adapted from LearnerSurvCV: https://mlr3learners.mlr-org.com/reference/mlr_learners_surv.glmnet.html
LearnerSurvGrpreg <- R6Class("LearnerSurvGrpreg",
  inherit = mlr3proba::LearnerSurv,
  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ps(
        nfolds = p_int(3, 10, default = 5, tags = "train"),
        stratify_by_event = p_lgl(default = TRUE, tags = "train"),
        penalize_clinical = p_lgl(default = TRUE, tags = "train")
      )

      super$initialize(
        id = "surv.cv_grpreg",
        param_set = ps,
        feature_types = c("logical", "integer", "numeric"),
        predict_types = c("crank", "lp", "distr"),
        properties = c()
      )
    },
  ),
  private = list(
    .train = function(task) {
      #data <- as_numeric_matrix(task$data(cols = task$feature_names))
      #target <- task$truth()
      pv <- self$param_set$get_values(tags = "train")
      
      data <- task$data()
      tn <- task$target_names
      time <- data[[tn[1L]]]
      status <- data[[tn[2L]]]
      data <- data[, !tn, with = FALSE]
      if (grepl("stratify_by_event", names(self$param_set))) {
        if (self$param_set$stratify_by_event == TRUE) {
          folds = create_folds(status, k = self$param_set$nfolds, type = "stratified", invert = TRUE)
          folds = sapply(1:length(status), function(x) unname(which(sapply(folds, function(y) x %in% y))))
          self$param_set = self$param_set[-grep("nfolds", names(self$param_set))]
        } else {
          folds = NULL
        }
        self$param_set = self$param_set[-grep("stratify_by_event", names(self$param_set))]
      }
      groups = unique(sapply(sapply(colnames(data), function(x) strsplit(x, "\\_")), function(y) y[1]))
      groups = as.factor(unname(sapply(sapply(sapply(colnames(data), function(x) strsplit(x, "\\_")), function(y) y[1]), function(z) grep(z, groups))))
      if (grepl("penalize_clinical", names(self$param_set))) {
        if (self$param_set$stratify_by_event == TRUE) {
          groups[grep("clinical", colnames(data))] = 0
        }
        self$param_set = self$param_set[-grep("penalize_clinical", names(self$param_set))]
      }
      list(
        mlr3misc::invoke(
          grpreg::cv.grpsurv,
          x = data,
          y = Surv(time, status),
          foldid = folds,
          penalty.factor = penalty_factor,
          .args = pv
        ),
        data,
        time,
        status
      )
    },
    .predict = function(task) {
      browser()
      selected_variables = names(which(self$model[[1]]$fit$beta[, self$model[[1]]$min] != 0))
      newdata <- as_numeric_matrix(ordered_features(task, self))
      time = self[[3]]
      status = self[[4]]
      y_train <- cbind(time, status)
      cox_helper <- coxph(
        Surv(time, status) ~ .,
        data = cbind(y_train, data.frame(X_train[, colnames(self[[2]]) %in% selected_variables])),
        init = self$model[[1]]$fit$beta[which(fit$fit$beta[, fit$min] != 0), fit$min],
        iter.max = 0,
        x = TRUE,
        ties = "breslow"
      )
      #newdata <- as.matrix(task$data(cols = self$state$feature_names))
      survival_probabilities <- pec::predictSurvProb(cox_helper,
        newdata = data.frame(newdata[, colnames(self$model[[2]]) %in% selected_variables]),
        times = time
      )
      
      lp = newdata[, colnames(self$model[[2]]) %in% selected_variables] %*% self$model[[1]]$fit$beta[which(self$model[[1]]$fit$beta[, self$model[[1]]$min] != 0), self$model[[1]]$min]

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

mlr_learners$add("surv.cv_grpreg", LearnerSurvGrpreg)

