library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
library(mlr3proba)
library(rjson)
library(dplyr)
library(forcats)
future::plan("sequential")

config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)

remove_constants <- po("removeconstants")
impute <- po("imputeconstant", affect_columns = selector_type(c("factor")), constant = "NA", check_levels = FALSE)
encode <- po("encode", method = "treatment")

source(here::here("survival-benchmark", "R", "learners", "cv_ridge_learners.R"))


set.seed(42)
learners <- list(
  remove_constants %>>% 
    encode %>>%
    #po("fixfactors", droplevels = FALSE) %>>% 
    po("learner",
                                       learner = lrn("surv.cv_ridge", s ="lambda.min", standardize = TRUE, nfolds = 5),
                                        #learner = lrn("surv.cv_ridge", nfolds ),
                                       id = "clinical_only_cox"
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


grid <- data.frame()
for (cancer in config$icgc_cancers[3:3]) {
  data <- vroom::vroom(
    here::here(
      "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
      "survival_benchmark", "data", "processed", "ICGC",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  
  lel <- data.frame(data[, c(which(substr(colnames(data), 1, 8) == "clinical"), which(colnames(data) == "OS"), which(colnames(data) == "OS_days"))]) %>% mutate(across(where(is.character), as.factor)) %>% mutate_if(is.factor,
                                                                                                                                            fct_explicit_na,
                                                                                                                                            na_level = "NA")
  tmp <- as_task_surv(lel,
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
  grid <- rbind(grid, benchmark_grid(
    tmp, learners,  ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
  ))
}


progressr::handlers(global = TRUE)
lgr::get_logger("mlr3")$set_threshold("warn")
set.seed(42)
bmr <- benchmark(grid)

saveRDS(bmr, here::here("data", "results", "clinical_cox_icgc.rds"))


grid <- data.frame()
for (cancer in config$target_cancers) {
  data <- vroom::vroom(
    here::here(
      "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
      "survival_benchmark", "data", "processed", "TARGET",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  
  lel <- data.frame(data[, c(which(substr(colnames(data), 1, 8) == "clinical"), which(colnames(data) == "OS"), which(colnames(data) == "OS_days"))]) %>% mutate(across(where(is.character), as.factor)) %>% mutate_if(is.factor,
                                                                                                                                            fct_explicit_na,
                                                                                                                                            na_level = "NA")
  tmp <- as_task_surv(lel,
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
  grid <- rbind(grid, benchmark_grid(
    tmp, learners,  ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
  ))
}


progressr::handlers(global = TRUE)
lgr::get_logger("mlr3")$set_threshold("warn")
set.seed(42)
bmr <- benchmark(grid)

saveRDS(bmr, here::here("data", "results", "clinical_cox_target.rds"))

set.seed(42)
grid <- data.frame()
for (cancer in config$tcga_cancers[-grep("SKCM", config$tcga_cancers)]) {
  data <- vroom::vroom(
    here::here(
      "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
      "survival_benchmark", "data", "processed", "TCGA",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  data <- data[, -grep("clinical_patient_id", colnames(data))]
  lel <- data.frame(data[, c(which(substr(colnames(data), 1, 8) == "clinical"), which(colnames(data) == "OS"), which(colnames(data) == "OS_days"))]) %>% mutate(across(where(is.character), as.factor)) %>% mutate_if(is.factor,
                                                                                                                                                                                                                      fct_explicit_na,
                                                                                                                                                                                                                      na_level = "NA")
  tmp <- as_task_surv(lel,
                      time = "OS_days",
                      event = "OS",
                      type = "right",
                      id = cancer
  )
  tmp$add_strata("OS")
  train_splits <- format_splits(readr::read_csv(here::here(
    "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
    "survival_benchmark", "data", "splits", "TCGA", paste0(cancer, "_train_splits.csv")
  )))
  test_splits <- format_splits(readr::read_csv(here::here(
    "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
    "survival_benchmark", "data", "splits", "TCGA", paste0(cancer, "_test_splits.csv")
  )))
  grid <- rbind(grid, benchmark_grid(
    tmp, learners,  ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
  ))
}


progressr::handlers(global = TRUE)
lgr::get_logger("mlr3")$set_threshold("warn")
set.seed(42)
bmr <- benchmark(grid)

saveRDS(bmr, here::here("data", "results", "clinical_cox_tcga.rds"))
