suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3proba)
  library(R6)
})

# Adapted from: https://github.com/mlr-org/mlr3learners/blob/HEAD/R/LearnerSurvRanger.R

#' Fits a BlockForest method using `mlr3` and `mlr3proba`.
#' For full documentation of all parameters please refer to the documentation
#' of `BlockForest::blockfor`.
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
        always.select.block = p_int(0, 1, default = 0, tags = "train"),
	 respect.unordered.factors = p_fct(c("ignore", "order", "partition"), default = "ignore", tags = "train") # for splitrule == "extratrees", def = partition
      )

      super$initialize(
        id = "surv.block_forest",
        param_set = ps,
	properties = c("missings"),
        predict_types = c("distr"),
        feature_types = c("logical", "integer", "numeric", "character", "factor", "ordered"),
        packages = c("mlr3learners", "blockForest")
      )
    }
  ),
  private = list(
    .train = function(task) {
      suppressPackageStartupMessages({
        library(blockForest)
        library(survival)
        library(here)
        source(here::here("survboard", "R", "utils", "utils.R"))
      })
      #print("BLOCKFOREST")
      #print(sum(is.na(task$data(cols = task$feature_names))))
      #stop("HEY BF")
      pv <- self$param_set$get_values(tags = "train")
      # Get indices of different modalities for usage during BlockForest.
      blocks <- get_block_assignment(
        c(
          "clinical",
          "gex",
          "cnv",
          "rppa",
          "mirna",
          "mut",
          "meth"
        ),
        task$feature_names
      )
      return(mlr3misc::invoke(
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
      ))
    },
    .predict = function(task) {
      pv <- self$param_set$get_values(tags = "predict")
      prediction <- mlr3misc::invoke(predict, self$model$forest, data = task$data(cols = task$feature_names), .args = pv)
      return(mlr3proba::.surv_return(times = prediction$unique.death.times, surv = prediction$survival))
    }
  )
)

mlr_learners$add("surv.blockforest", LearnerSurvBlockForest)
