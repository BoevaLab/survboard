library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
library(mlr3proba)
library(rjson)
library(dplyr)
#future::plan("multisession", workers = 4)
future::plan("sequential")
options(future.globals.onReference = "ignore")
config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)

remove_constants <- po("removeconstants")
encode <- po("encode", method = "treatment")
fix_factors <- po("fixfactors")
impute_missing_prediction <- po("imputeoor", affect_columns=selector_type("factor"))
remove_column_missing_train_factor <- po("select", selector=selector_invert(selector_grep("DROPME")))
impute <- po("imputeconstant", constant = 0, affect_columns=selector_grep("clinical"))



pipe <- remove_constants %>>% fix_factors %>>% encode %>>% impute
source(here::here("survival_benchmark", "R", "learners", "blockforest_learners.R"))
source(here::here("survival_benchmark", "R", "learners", "cv_coxboost_learners.R"))
source(here::here("survival_benchmark", "R", "learners", "cv_lasso_learners.R"))
source(here::here("survival_benchmark", "R", "learners", "cv_prioritylasso_learners.R"))
source(here::here("survival_benchmark", "R", "learners", "ranger_learners.R"))
set.seed(42)

learners <- list(
  remove_constants %>>% fix_factors %>>% encode %>>% impute %>>% po("learner",
                                       id = "CoxBoost",
                                       learner = lrn("surv.cv_coxboost_custom",
                                                     #encapsulate = c(train = "evaluate", predict = "evaluate"),
                                                     #fallback = lrn("surv.kaplan"),
                                                     favor_clinical = FALSE,
                                                     penalty = "optimCoxBoostPenalty",
                                                     K = 5,
                                                     return.score = FALSE
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

for (cancer in config$target_cancers[1:1]) {
  data <- vroom::vroom(
    here::here(
      "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
      "survival_benchmark", "data", "processed", "TARGET",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
    #replace(is.na(.), 0)
    mutate(across(where(is.character), as.factor)) %>%
    mutate(across(where(is.factor), forcats::fct_explicit_na, "MISSING"))
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
