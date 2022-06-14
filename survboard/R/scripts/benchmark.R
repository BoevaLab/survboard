library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
library(mlr3proba)
library(rjson)
library(dplyr)
future::plan("multisession")
options(future.globals.onReference = "ignore")
config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)

remove_constants <- po("removeconstants")
impute <- po("imputeconstant", affect_columns = selector_type(c("factor")), constant = "NA", check_levels = FALSE)
encode <- po("encode", method = "treatment")
source(here::here("survboard", "R", "learners", "blockforest_learners.R"))
source(here::here("survboard", "R", "learners", "cv_coxboost_learners.R"))
source(here::here("survboard", "R", "learners", "cv_lasso_learners.R"))
source(here::here("survboard", "R", "learners", "cv_prioritylasso_learners.R"))
source(here::here("survboard", "R", "learners", "ranger_learners.R"))
set.seed(42)

learners <- list(
  remove_constants %>>% po("learner",
    id = "BlockForest",
    learner = lrn("surv.blockforest",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      block.method = "BlockForest", num.trees = 2000, mtry = NULL, nsets = 300, num.trees.pre = 1500, splitrule = "extratrees", always.select.block = 0
    )
  ),
  remove_constants %>>% po("learner",
    id = "BlockForest_favoring",
    learner = lrn("surv.blockforest",
      block.method = "BlockForest",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      num.trees = 2000, mtry = NULL, nsets = 300, num.trees.pre = 1500, splitrule = "extratrees", always.select.block = 1
    )
  ),
  remove_constants %>>% encode %>>% po("learner",
    id = "CoxBoost",
    learner = lrn("surv.cv_coxboost_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      favor_clinical = FALSE,
      penalty = "optimCoxBoostPenalty",
      K = 5
    )
  ),
  remove_constants %>>% po("learner",
    id = "ranger",
    learner = lrn("surv.ranger_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      favor_clinical = FALSE,
      num.trees = 2000
    )
  ),
  remove_constants %>>% po("learner",
    id = "ranger_favoring",
    learner = lrn("surv.ranger_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      favor_clinical = TRUE,
      num.trees = 2000
    )
  ),
  remove_constants %>>% encode %>>% po("learner",
    id = "CoxBoost_favoring",
    learner = lrn("surv.cv_coxboost_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      favor_clinical = TRUE,
      penalty = "optimCoxBoostPenalty",
      K = 5
    )
  ),
  remove_constants %>>% encode %>>% po("learner",
    id = "prioritylasso",
    learner = lrn("surv.cv_prioritylasso",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      block1.penalization = TRUE, lambda.type = "lambda.min", standardize = TRUE, nfolds = 5, cvoffset = TRUE, cvoffsetnfolds = 5, favor_clinical = FALSE
    )
  ),
  remove_constants %>>% encode %>>% po("learner",
    id = "prioritylasso_favoring",
    learner = lrn("surv.cv_prioritylasso",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      block1.penalization = FALSE, lambda.type = "lambda.min", standardize = TRUE, nfolds = 5, cvoffset = TRUE, cvoffsetnfolds = 5, favor_clinical = TRUE
    )
  ),
  remove_constants %>>% encode %>>% po("learner",
    id = "Lasso",
    learner = lrn("surv.cv_glmnet_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      s = "lambda.min", standardize = TRUE, favor_clinical = FALSE, nfolds = 5
    )
  ),
  remove_constants %>>% encode %>>% po("learner",
    id = "Lasso_favoring",
    learner = lrn("surv.cv_glmnet_custom",
      s = "lambda.min",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      standardize = TRUE, favor_clinical = TRUE, nfolds = 5
    )
  )
)

format_splits <- function(raw_splits) {
  if (any(is.na(raw_splits))) {
    apply(data.frame(raw_splits), 1, function(x) unname(x[!is.na(x)]) + 1)
  } else {
    x <- unname(as.matrix(raw_splits)) + 1
    split(x, row(x))
  }
}

for (cancer in config$target_cancers) {
  data <- vroom::vroom(
    here::here(
      "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
      "survival_benchmark", "data", "processed", "TARGET",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
    mutate(across(where(is.character), as.factor)) %>%
    mutate(across(where(is.factor), forcats::fct_explicit_na, "NA"))
  tmp <- as_task_surv(data,
    time = "OS_days",
    event = "OS",
    type = "right",
    id = cancer
  )
  tmp$add_strata("OS")
  train_splits <- format_splits(readr::read_csv(here::here(
    "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
    "survival_benchmark", "data", "splits", "TARGET", paste0(cancer, "_train_splits.csv")
  )))
  test_splits <- format_splits(readr::read_csv(here::here(
    "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
    "survival_benchmark", "data", "splits", "TARGET", paste0(cancer, "_test_splits.csv")
  )))
  grid <- benchmark_grid(
    tmp, learners, ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
  )
  bmr <- benchmark(grid)
  saveRDS(
    bmr,
    here::here(
      here::here("data", "results", "TARGET", paste0(cancer, "_results.rds"))
    )
  )
}
