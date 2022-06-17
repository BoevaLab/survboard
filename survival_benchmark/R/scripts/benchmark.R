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
encode <- po("encode", method = "treatment")
fix_factors <- po("fixfactors")
impute_missing_prediction <- po("imputeoor", affect_columns = selector_type("factor"))
remove_column_missing_train_factor <- po("select", selector = selector_invert(selector_grep("DROPME")))
impute <- po("imputeconstant", constant = 0, affect_columns = selector_grep("clinical"))


pipe <- remove_constants %>>% fix_factors
pipe_ohe <- pipe %>>% encode %>>% impute
source(here::here("survival_benchmark", "R", "learners", "blockforest_learners.R"))
source(here::here("survival_benchmark", "R", "learners", "cv_coxboost_learners.R"))
source(here::here("survival_benchmark", "R", "learners", "cv_lasso_learners.R"))
source(here::here("survival_benchmark", "R", "learners", "cv_prioritylasso_learners.R"))
source(here::here("survival_benchmark", "R", "learners", "ranger_learners.R"))
set.seed(42)

learners <- list(
  # BlockForest
  #pipe %>>% impute_missing_prediction %>>% po("learner",
  #                                            id = "BlockForest",
  #                                           learner = lrn("surv.blockforest",
  #                                                          encapsulate = c(train = "evaluate", predict = "evaluate"),
  #  fallback = lrn("surv.kaplan"),
  #  
  #                                      block.method = "BlockForest",
  #  num.trees = 2000, mtry = NULL, nsets = 100, num.trees.pre = 1500, splitrule = "extratrees", always.select.block = 0
   #                                           )
  #),
  # BlockForest favoring clinical data
  #pipe %>>% impute_missing_prediction %>>% po("learner",
  #                                            id = "BlockForest_favoring",
  #                                            learner = lrn("surv.blockforest",
  #                                                          block.method = "BlockForest",
  #                                                          encapsulate = c(train = "evaluate", predict = "evaluate"),
  #                                                          fallback = lrn("surv.kaplan"),
  #                                                          num.trees = 2000, mtry = NULL, nsets = 100, num.trees.pre = 1500, splitrule = "e#xtratrees", always.select.block = 1
  #                                            )
  #),
  #pipe_ohe %>>% po("learner",
  #                 id = "CoxBoost",
  #                 learner = lrn("surv.cv_coxboost_custom",
                                 #encapsulate = c(train = "evaluate", predict = "evaluate"),
                                 #fallback = lrn("surv.kaplan"),
  #                               favor_clinical = FALSE,
  #                               K = 5
  #                 )
  #)
  #pipe %>>% impute_missing_prediction %>>% po("learner",
  #                                            id = "ranger",
  #                                            learner = lrn("surv.ranger_custom",
  #                                                          encapsulate = c(train = "evaluate", predict = "evaluate"),
  #                                                          fallback = lrn("surv.kaplan"),
  #                                                          favor_clinical = FALSE,
  #                                                          num.trees = 2000,
  #                                                          splitrule = "extratrees"
  #                                            )
  #),
  #pipe %>>% impute_missing_prediction %>>% po("learner",
  #                                            id = "ranger_favoring",
  #                                            learner = lrn("surv.ranger_custom",
  #                                                          encapsulate = c(train = "evaluate", predict = "evaluate"),
  #                                                          fallback = lrn("surv.kaplan"),
  #                                                          favor_clinical = TRUE,
  #                                                          num.trees = 2000,
  #                                                          splitrule = "extratrees"
  #                                            )
  #),
  #pipe_ohe %>>% po("learner",
   #                id = "CoxBoost_favoring",
   #
  #learner = lrn("surv.cv_coxboost_custom",
  #                               encapsulate = c(train = "evaluate", predict = "evaluate"),
  #                               fallback = lrn("surv.kaplan"),
  #                               favor_clinical = TRUE,
  #                               penalty = "optimCoxBoostPenalty",
  #                               K = 5
  #                 )
  #)
  pipe_ohe %>>% po("learner",
                   id = "prioritylasso",
                  learner = lrn("surv.cv_prioritylasso",
                              encapsulate = c(train = "evaluate", predict = "evaluate"),
                              fallback = lrn("surv.kaplan"),
                               block1.penalization = TRUE, lambda.type = "lambda.min", standardize = TRUE, nfolds = 5, cvoffset = FALSE, cvoffsetnfolds = 5, favor_clinical = TRUE
                   )
  )
  #pipe_ohe %>>% po("learner",
  #                 id = "prioritylasso_favoring",
  #                 learner = lrn("surv.cv_prioritylasso",
   #                            encapsulate = c(train = "evaluate", predict = "evaluate"),
   #                              fallback = lrn("surv.kaplan"),
  #                               block1.penalization = TRUE, lambda.type = "lambda.min", standardize = TRUE, nfolds = 5, cvoffset = FALSE, cvoffsetnfolds = 5, favor_clinical = TRUE
  #                 )
  #)#,
  #pipe_ohe %>>% po("learner", id = "CoxPH", learner = lrn("surv.coxph"))
  #pipe_ohe %>>% po("learner",
  #                 id = "Lasso",
  #                 learner = lrn("surv.cv_glmnet_custom",
  #                               encapsulate = c(train = "evaluate", predict = "evaluate"),
  #                               fallback = lrn("surv.kaplan"),
   #                              s = "lambda.min", standardize = TRUE, favor_clinical = FALSE, nfolds = 5
   #                )
  #)
  #pipe_ohe %>>% po("learner",
  #                 id = "Lasso_favoring",
  #                 learner = lrn("surv.cv_glmnet_custom",
  #                               encapsulate = c(train = "evaluate", predict = "evaluate"),
  #                               fallback = lrn("surv.kaplan"),
  #                               s = "lambda.min", standardize = TRUE, favor_clinical = TRUE, nfolds = 5
  #                 )
  #)
)


#learners <- list(
#  pipe_ohe %>>% po("learner", id = "KM", learner = lrn("surv.kaplan"))
#)

format_splits <- function(raw_splits) {
  if (any(is.na(raw_splits))) {
    apply(data.frame(raw_splits), 1, function(x) unname(x[!is.na(x)]) + 1)
  } else {
    x <- unname(as.matrix(raw_splits)) + 1
    split(x, row(x))
  }
}


for (cancer in c("CLLE-ES")) {
  data <- vroom::vroom(
    here::here(
      "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
      "survival_benchmark", "data", "processed", "ICGC",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
    mutate(across(where(is.character), as.factor)) %>%
    mutate(across(where(is.factor), forcats::fct_explicit_na, "MISSING"))
  data <- data[, 
               c(which(colnames(data) == "OS"), which(colnames(data) == "OS_days"), which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == "gex"))]
  tmp <- as_task_surv(data,
    time = "OS_days",
    event = "OS",
    type = "right",
    id = cancer
  )
  tmp$add_strata("OS")
  train_splits <- format_splits(readr::read_csv(here::here(
    "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
    "survival_benchmark", "data", "splits", "ICGC", paste0(cancer, "_train_splits.csv")
  )))
  test_splits <- format_splits(readr::read_csv(here::here(
    "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
    "survival_benchmark", "data", "splits", "ICGC", paste0(cancer, "_test_splits.csv")
  )))
  grid <- benchmark_grid(
    tmp, learners, ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
  )
  bmr <- benchmark(grid)
  saveRDS(
    bmr,
    here::here(
      here::here("data", "results", "ICGC", paste0(cancer, "_results_prioritylasso_favoring_emergency.rds"))
    )
  )
}

