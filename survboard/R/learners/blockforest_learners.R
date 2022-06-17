library(mlr3)
library(mlr3proba)
library(R6)

# Adapted from: https://github.com/mlr-org/mlr3learners/blob/HEAD/R/LearnerSurvRanger.R
LearnerSurvBlockForest <- R6Class("LearnerSurvBlockForest",
  inherit = mlr3proba::LearnerSurv,
  public = list(
    initialize = function() {
      ps <- ps(
        block.method = p_fct(c("BlockForest", "RandomBlock", "BLockVarSel", "VarProb", "SplitWeights"), default = "BlockForest", tags = "train"),
        num.trees = p_int(1, 2000, default = 2000, tags = "train"),
        mtry = p_uty(default = NULL, tags = "train"),
        nsets = p_int(1, 300, default = 300, tags = "train"),
        num.trees.pre = p_int(1, 1500, default = 1500, tags = "train"),
        splitrule = p_fct(c("logrank", "extratrees", "C", "maxstat"), default = "extratrees", tags = "train"),
        always.select.block = p_int(0, 1, default = 0, tags = "train")
      )

      ps$values <- list(
        block.method = "BlockForest",
        num.trees = 2000,
        mtry = NULL,
        nsets = 300,
        num.trees.pre = 1500,
        splitrule = "extratrees",
        always.select.block = 0
      )

      super$initialize(
        id = "surv.block_forest",
        param_set = ps,
        predict_types = c("distr"),
        feature_types = c("logical", "integer", "numeric", "character", "factor", "ordered"),
        packages = c("mlr3learners", "blockForest")
      )
    }
  ),
  private = list(
    .train = function(task) {
      library(blockForest)
      library(survival)
      pv <- self$param_set$get_values(tags = "train")
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
      names(blocks) <- paste0("bp", 1:length(blocks))
      mlr3misc::invoke(
        blockForest::blockfor,
        X = task$data(cols = task$feature_names),
        y = task$truth(),
        blocks = blocks,
        # We force `num.threads = 1` to prevent 
        # multi-threading during training, since
        # we only parallelize outer cross-validation
        # folds for statistical models.
        num.threads = 1,
        .args = pv
      )
    },
    .predict = function(task) {
      pv <- self$param_set$get_values(tags = "predict")
      prediction <- mlr3misc::invoke(predict, self$model$forest, data = task$data(cols = task$feature_names), .args = pv)
      mlr3proba::.surv_return(times = prediction$unique.death.times, surv = prediction$survival)
    }
  )
)

mlr_learners$add("surv.blockforest", LearnerSurvBlockForest)
