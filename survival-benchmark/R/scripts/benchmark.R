library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
library(mlr3proba)

remove_constants <- po("removeconstants")
impute <- po("imputeconstant", affect_columns = selector_type(c("factor")), constant = "NA", check_levels = FALSE)
encode <- po("encode", method = "treatment")

source(here::here("survival-benchmark", "R", "learners", "blockforest_learners.R"))
source(here::here("survival-benchmark", "R", "learners", "coxboost_learners.R"))
source(here::here("survival-benchmark", "R", "learners", "cv_lasso_learners.R"))
source(here::here("survival-benchmark", "R", "learners", "cv_prioritylasso_learners.R"))
source(here::here("survival-benchmark", "R", "learners", "ranger_learners.R"))
source(here::here("survival-benchmark", "R", "measures", "surv_measure_antolini_c.R"))


learners <- list(
  # BlockForest
  impute %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.block_forest", block.method = "BlockForest", num.trees = 2000, mtry = NULL, nsets = 300, num.trees.pre = 1500, splitrule = "extratrees", always.select.block = 0)
  ),
  # BlockForest favoring clinical data
  impute %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.block_forest", block.method = "BlockForest", num.trees = 2000, mtry = NULL, nsets = 300, num.trees.pre = 1500, splitrule = "extratrees", always.select.block = 1)
  ),

  # Group Lasso
  impute %>>% encode %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.cv_grplasso", nfolds = 5, favor_clincial = FALSE)
  ),

  # Group Lasso favoring clinical data
  impute %>>% encode %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.cv_grplasso", nfolds = 5, favor_clincial = TRUE)
  ),

  # Prioritylasso
  impute %>>% encode %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.cv_prioritylasso", block1.penalization = TRUE, lambda.type = "lambda.min", standardize = TRUE, nfolds = 5, cvoffset = TRUE, cvoffsetnfolds = 5, favor_clinical = FALSE)
  ),


  # Prioritylasso favoring
  impute %>>% encode %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.cv_prioritylasso", block1.penalization = FALSE, lambda.type = "lambda.min", standardize = TRUE, nfolds = 5, cvoffset = TRUE, cvoffsetnfolds = 5, favor_clinical = TRUE)
  ),

  # Ranger
  impute %>>% remove_constants %>>% po("learner",
                                       learner = AutoTuner$new(
                                         learner = lrn("surv.ranger", 
                                                       mtry.ratio = to_tune(0.0, 1),
                                                       replace = to_tune(c(TRUE, FALSE)),
                                                       sample.fraction = to_tune(0.1, 1)
                                                       ),
                                         resampling = rsmp("cv", folds = 5),
                                         measure = msr("surv.c_antolini"),
                                         terminator = trm("evals", n_evals = 50),
                                         tuner = tnr("random_search"))
  ),

  # Ranger favoring clinical variables
  impute %>>% remove_constants %>>% po("learner",
                                       learner = AutoTuner$new(
                                         learner = lrn("surv.ranger_favor", 
                                                       mtry.ratio = to_tune(0.0, 1),
                                                       replace = to_tune(c(TRUE, FALSE)),
                                                       sample.fraction = to_tune(0.1, 1)
                                         ),
                                         resampling = rsmp("cv", folds = 5),
                                         measure = msr("surv.c_antolini"),
                                         terminator = trm("evals", n_evals = 50),
                                         tuner = tnr("random_search"))
  ),

  # Lasso
  impute %>>% encode %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.cv_glmnet_custom", s = "lambda.min", standardize = TRUE, favor_clinical = FALSE)
  ),

  # Lasso favoring clinical variables
  impute %>>% encode %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.cv_glmnet_custom", s = "lambda.min", standardize = TRUE, favor_clinical = TRUE)
  ),

  # CoxBoost
  impute %>>% encode %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.coxboost_cv_custom",
      favor_clinical = FALSE,
      penalty = "optimCoxBoostPenalty"
    )
  ),

  # CoxBoost favoring clinical variables
  impute %>>% encode %>>% remove_constants %>>% po("learner",
    learner = lrn("surv.coxboost_cv_custom",
      favor_clinical = TRUE,
      penalty = "optimCoxBoostPenalty"
    )
  ),
)
#rsmp <- rsmp("repeated_cv", repeats = 5, folds = 5)
set.seed(1)
resampling <- rsmp("cv", folds = 2)
future::plan("multisession")
#learners <- list(
#  # BlockForest
#  impute %>>% remove_constants %>>% po("learner",
#                                       learner = lrn("surv.blockforest", block.method = "BlockForest"#, num.trees = 5, mtry = NULL, nsets = 2, num.trees.pre = 5, splitrule = "extratrees", always.select.block = 0)
  #),
  # BlockForest favoring clinical data
  #impute %>>% remove_constants %>>% po("learner",
  #                                     learner = lrn("surv.blockforest", block.method = "BlockForest", num.trees = 5, mtry = NULL, nsets = 2, num.trees.pre = 5, splitrule = "extratrees", always.select.block = 1)
#  ))

learners <- list(
  # BlockForest
  impute %>>% remove_constants %>>% encode %>>% po("learner",
                                       learner = lrn("surv.cv_prioritylasso", block1.penalization = TRUE, lambda.type = "lambda.min", standardize = TRUE, nfolds =5, cvoffset = FALSE, cvoffsetnfolds = 5, favor_clinical = FALSE)
  ),
  impute %>>% remove_constants %>>% encode %>>% po("learner",
                                                   learner = lrn("surv.cv_prioritylasso", block1.penalization = FALSE, lambda.type = "lambda.min", standardize = TRUE, nfolds =5, cvoffset = FALSE, cvoffsetnfolds = 5, favor_clinical = TRUE))
  #impute %>>% remove_constants %>>% encode %>>% po("learner",
                                                                                                    #learner = lrn("surv.cv_glmnet_custom", s = "lambda.min", standardize = TRUE, favor_clinical = FALSE, nfolds = 5)),
#impute %>>% remove_constants %>>% encode %>>% po("learner",
                                                 #learner = lrn("surv.cv_glmnet_custom", s = "lambda.min", standardize = TRUE, favor_clinical = TRUE, nfolds = 5))
  #impute %>>% remove_constants %>>% po("learner",
  #                                     learner = AutoTuner$new(
  #                                       learner = lrn("surv.ranger_custom", 
  #                                                     mtry.ratio = to_tune(0.0, 1),
  #                                                     replace = to_tune(c(TRUE, FALSE)),
  #                                                     sample.fraction = to_tune(0.1, 1),
  #                                                     favor_clinical = TRUE
  #                                       ),
  #                                       resampling = rsmp("cv", folds = 2),
  #                                       measure = msr("surv.c_antolini", method = "adj_antolini"),
  #                                       terminator = trm("evals", n_evals = 2),
  #                                       tuner = tnr("random_search"))
  #),
  #impute %>>% remove_constants %>>% po("learner",
  #                                     learner = AutoTuner$new(
  #                                       learner = lrn("surv.ranger_custom", 
  #                                                     mtry.ratio = to_tune(0.0, 1),
  #                                                     replace = to_tune(c(TRUE, FALSE)),
  #                                                     sample.fraction = to_tune(0.1, 1),
  #                                                     favor_clinical = FALSE
  #                                       ),
  #                                       resampling = rsmp("cv", folds = 2),
  #                                       measure = msr("surv.c_antolini", method = "adj_antolini"),
  #                                       terminator = trm("evals", n_evals = 2),
  #                                       tuner = tnr("random_search"))
  #)
)

bmr <- benchmark(benchmark_grid(tasks, learners, resampling))


# TCGA

tasks <- list()

for (cancer in c("PAAD")) {
  data <- vroom::vroom(
    here::here(
      "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
      "survival_benchmark", "data", "processed", "TARGET",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  splits <- NA
  lel <-     data.frame(data[, -grep("patient_id", colnames(data))]) %>% mutate(across(where(is.character), as.factor))
  tmp <- as_task_surv(
,
    time = "OS_days",
    event = "OS",
    type = "right"
  )
  tmp$add_strata("OS")
  tasks <- append(tasks, tmp)
}

# ICGC

# TARGET
