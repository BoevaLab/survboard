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
        predict_types = c("crank", "lp", "distr")
      )
    }
  ),
  private = list(
    .train = function(task) {
      library(prioritylasso)
      library(splitTools)
      source(here::here("survival-benchmark", "R", "utils", "utils.R"))
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
      blocks <- sapply(block_order, function(x) grep(x, task$feature_names))
      blocks <- blocks[sapply(blocks, length) > 1]
      names(blocks) <- paste0("bp", 1:length(blocks))
      pv <- pv[-grep("favor_clinical", names(pv))]
      
      list(mlr3misc::invoke(
        prioritylasso, X = data, Y = target, .args = pv,
        foldid = foldids_formatted, blocks = blocks, 
        type.measure = "deviance"
      ), data, target)
    },
    .predict = function(task) {
      library(prioritylasso)
      source(here::here("survival-benchmark", "R", "utils", "utils.R"))
      #browser()
      model <- self$model[[1]]
      train_data <- self$model[[2]]
      train_target <- self$model[[3]]
      newdata <- as_numeric_matrix(ordered_features(task, self))
      pv <- self$param_set$get_values(tags = "predict")
      #lp <- as.numeric(invoke(predict, model, newdata = newdata, type = "link", .args = pv))
      coefficients <- model$coefficients[which(model$coefficients!=0)]
      lp <- as.numeric(newdata[, which(model$coefficients != 0)] %*% as.matrix(coefficients))
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

mlr_learners$add("surv.cv_prioritylasso", LearnerSurvCVPrioritylasso)