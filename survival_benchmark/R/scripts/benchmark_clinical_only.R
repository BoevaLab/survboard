library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
library(mlr3proba)
library(rjson)
library(dplyr)
library(forcats)
future::plan("multisession", workers = 4)

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

set.seed(42)
learners <- list(
  pipe_ohe %>>% 
    po("learner", learner = lrn("surv.coxph"),
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
for (cancer in config$icgc_cancers) {
  data <- vroom::vroom(
    here::here(
      "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
      "survival_benchmark", "data", "processed", "ICGC",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  data <- data.frame(data[, c(which(colnames(data) == "OS"), which(colnames(data) == "OS_days") , which(sapply(strsplit(colnames(data), "_"), function(x) x[[1]]) == "clinical"))]) %>%
    mutate(across(where(is.character), as.factor)) %>%
    mutate(across(where(is.factor), forcats::fct_explicit_na, "MISSING"))
  tmp <- as_task_surv(data,
                      time = "OS_days",
                      event = "OS",
                      type = "right",
                      id = cancer
  )
  tmp$add_strata("OS")
  rm(data)
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
bmr <- benchmark(grid, store_backends = FALSE, clone = c("learner"))

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
  data <- data.frame(data[, c(which(colnames(data) == "OS"), which(colnames(data) == "OS_days") , which(sapply(strsplit(colnames(data), "_"), function(x) x[[1]]) == "clinical"))]) %>%
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
for (cancer in config$tcga_cancers) {
  data <- vroom::vroom(
    here::here(
      "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark",
      "survival_benchmark", "data", "processed", "TCGA",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  data <- data.frame(data[, c(which(colnames(data) == "OS"), which(colnames(data) == "OS_days") , which(sapply(strsplit(colnames(data), "_"), function(x) x[[1]]) == "clinical"))]) %>%
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
