library(R6)
library(mlr3)
library(mlr3proba)
library(mlr3tuningspaces)
library(prioritylasso)

LearnerSurvCVPrioritylasso <- R6Class("LearnerSurvCVPrioritylasso",
  inherit = mlr3proba::LearnerSurv,
  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ps(
        block1.penalization = p_lgl(default = TRUE, tags = "train"),
        lambda.type = p_fct(c("lambda.min", "lambda.1se"), default = "lambda.min", tags = "train"),
        standardize = p_lgl(default = TRUE, tags = "train"),
        nfolds = p_int(3, 10, default = 5, tags = "train"),
        cvoffset = p_lgl(default = TRUE, tags = "train"),
        cvoffsetnfolds = p_int(3, 10, default = 5),
        favor_clinical = p_lgl(default = FALSE, tags = "train")
      )

      super$initialize(
        id = "surv.cv_prioritylasso",
        param_set = ps,
        feature_types = c("logical", "integer", "numeric"),
        predict_types = c("distr", "crank", "lp")
      )
    }
  ),
  private = list(
    .train = function(task) {
      browser()
      library(prioritylasso)
      library(splitTools)
      source(here::here("survival_benchmark", "R", "utils", "utils.R"))
      data <- as_numeric_matrix(task$data(cols = task$feature_names))
      target <- task$truth()

      pv <- self$param_set$get_values(tags = "train")
      pv$family <- "cox"
      foldids <- create_folds(target[, 2], k = pv$nfolds, invert = TRUE, type = "stratified")
      foldids_formatted <- rep(1, nrow(data))
      for (i in 2:length(foldids)) {
        foldids_formatted[foldids[[i]]] <- i
      }
      favor_clinical <- pv$favor_clinical
      block_order <- get_prioritylasso_block_order(
        target, data, unique(sapply(strsplit(task$feature_names, "\\_"), function(x) x[1])), foldids_formatted, pv$lambda.type, favor_clinical
      )
      blocks <- lapply(block_order, function(x) which(sapply(strsplit(task$feature_names, "\\_"), function(y) y[[1]]) == x))
      blocks <- blocks[sapply(blocks, length) > 1]
      names(blocks) <- paste0("bp", 1:length(blocks))
      pv <- pv[-which(names(pv) == "favor_clinical")]

      prioritylasso_fit <- mlr3misc::invoke(
        prioritylasso,
        X = data, Y = target, .args = pv,
        foldid = foldids_formatted, blocks = blocks,
        type.measure = "deviance"
      )


      cox_helper <- transform_cox_model(prioritylasso_fit$coefficients[which(prioritylasso_fit$coefficients != 0)], data[, sapply(names(prioritylasso_fit$coefficients[which(prioritylasso_fit$coefficients != 0)]), function(x) which(colnames(data) == x))], target)
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

mlr_learners$add("surv.cv_prioritylasso", LearnerSurvCVPrioritylasso)
