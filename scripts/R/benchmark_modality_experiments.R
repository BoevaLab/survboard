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
source(here::here("survboard", "R", "learners", "blockforest_learners.R"))
source(here::here("survboard", "R", "learners", "cv_coxboost_learners.R"))
source(here::here("survboard", "R", "learners", "cv_lasso_learners.R"))
source(here::here("survboard", "R", "learners", "cv_prioritylasso_learners.R"))
source(here::here("survboard", "R", "learners", "ranger_learners.R"))
set.seed(42)

model_names <- c(
  "BlockForest",
  "RSF",
  "Lasso"
)

learners <- list(
  pipe %>>% impute_missing_prediction %>>% po("learner",
    id = "BlockForest",
    learner = lrn("surv.blockforest",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      block.method = "BlockForest",
      num.trees = 2000, mtry = NULL, nsets = 100, num.trees.pre = 1500, splitrule = "extratrees", always.select.block = 0
    )
  ),
  pipe %>>% impute_missing_prediction %>>% po("learner",
    id = "ranger",
    learner = lrn("surv.ranger_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      favor_clinical = FALSE,
      num.trees = 2000,
      splitrule = "extratrees"
    )
  ),
  pipe_ohe %>>% po("learner",
    id = "Lasso",
    learner = lrn("surv.cv_glmnet_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      s = "lambda.min", standardize = TRUE, favor_clinical = FALSE, nfolds = 5
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

for (cancer in config$tcga_cancers[c(1:11, 13:17, 19)]) {
  data <- vroom::vroom(
    here::here(
      "data", "processed", "TCGA",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
    mutate(across(where(is.character), as.factor)) %>%
    mutate(across(where(is.factor), forcats::fct_explicit_na, "MISSING"))
  data <- data[
    ,
    which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) != "clinical")
  ]
  tmp <- as_task_surv(data,
    time = "OS_days",
    event = "OS",
    type = "right",
    id = cancer
  )
  tmp$add_strata("OS")
  train_splits <- format_splits(readr::read_csv(here::here(
    "data", "splits", "TCGA", paste0(cancer, "_train_splits.csv")
  )))
  test_splits <- format_splits(readr::read_csv(here::here(
    "data", "splits", "TCGA", paste0(cancer, "_test_splits.csv")
  )))
  grid <- benchmark_grid(
    tmp, learners, ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
  )
  bmr <- benchmark(grid)
  bmr <- bmr$score()
  for (model in 0:length(model_names)) {
    if (!dir.exists(here::here(
      "data", "results_reproduced", "survival_functions", "TCGA", cancer, paste0(model_names[(model + 1)], "_without_clinical")
    ))) {
      dir.create(
        here::here(
          "data", "results_reproduced", "survival_functions", "TCGA", cancer, paste0(model_names[(model + 1)], "_without_clinical")
        ),
        recursive = TRUE
      )
    }
    for (i in 1:25) {
      surv_pred <- data.frame(bmr[(1 + (model * 25):((model + 1) * 25))]$prediction[[i]]$data$distr, check.names = FALSE)
      surv_pred %>% readr::write_csv(
        here::here(
          "data", "results_reproduced", "survival_functions", "TCGA", cancer, paste0(model_names[(model + 1)], "_without_clinical"), paste0("split_", i, ".csv")
        )
      )
    }
  }
}

for (cancer in config$tcga_cancers[c(1:11, 13:17, 19)]) {
  data <- vroom::vroom(
    here::here(
      "data", "processed", "TCGA",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
    mutate(across(where(is.character), as.factor)) %>%
    mutate(across(where(is.factor), forcats::fct_explicit_na, "MISSING"))
  data <- data[
    ,
    which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) %in% c("OS", "clinical", "gex"))
  ]
  tmp <- as_task_surv(data,
                      time = "OS_days",
                      event = "OS",
                      type = "right",
                      id = cancer
  )
  tmp$add_strata("OS")
  train_splits <- format_splits(readr::read_csv(here::here(
    "data", "splits", "TCGA", paste0(cancer, "_train_splits.csv")
  )))
  test_splits <- format_splits(readr::read_csv(here::here(
    "data", "splits", "TCGA", paste0(cancer, "_test_splits.csv")
  )))
  grid <- benchmark_grid(
    tmp, learners, ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
  )
  bmr <- benchmark(grid)
  bmr <- bmr$score()
  for (model in 0:length(model_names)) {
    if (!dir.exists(here::here(
      "data", "results_reproduced", "survival_functions", "TCGA", cancer, paste0(model_names[(model + 1)], "_gex_clinical")
    ))) {
      dir.create(
        here::here(
          "data", "results_reproduced", "survival_functions", "TCGA", cancer, paste0(model_names[(model + 1)], "_gex_clinical")
        ),
        recursive = TRUE
      )
    }
    for (i in 1:25) {
      surv_pred <- data.frame(bmr[(1 + (model * 25):((model + 1) * 25))]$prediction[[i]]$data$distr, check.names = FALSE)
      surv_pred %>% readr::write_csv(
        here::here(
          "data", "results_reproduced", "survival_functions", "TCGA", cancer, paste0(model_names[(model + 1)], "_gex_clinical"), paste0("split_", i, ".csv")
        )
      )
    }
  }
}

model_names <- c(
  "RSF",
  "Lasso"
)

learners <- list(
  pipe %>>% impute_missing_prediction %>>% po("learner",
                                              id = "ranger",
                                              learner = lrn("surv.ranger_custom",
                                                            encapsulate = c(train = "evaluate", predict = "evaluate"),
                                                            fallback = lrn("surv.kaplan"),
                                                            favor_clinical = FALSE,
                                                            num.trees = 2000,
                                                            splitrule = "extratrees"
                                              )
  ),
  pipe_ohe %>>% po("learner",
                   id = "Lasso",
                   learner = lrn("surv.cv_glmnet_custom",
                                 encapsulate = c(train = "evaluate", predict = "evaluate"),
                                 fallback = lrn("surv.kaplan"),
                                 s = "lambda.min", standardize = TRUE, favor_clinical = FALSE, nfolds = 5
                   )
  )
)

for (cancer in config$tcga_cancers[c(1:11, 13:17, 19)]) {
  data <- vroom::vroom(
    here::here(
      "data", "processed", "TCGA",
      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
    mutate(across(where(is.character), as.factor)) %>%
    mutate(across(where(is.factor), forcats::fct_explicit_na, "MISSING"))
  data <- data[
    ,
    which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) %in% c("OS", "gex"))
  ]
  tmp <- as_task_surv(data,
                      time = "OS_days",
                      event = "OS",
                      type = "right",
                      id = cancer
  )
  tmp$add_strata("OS")
  train_splits <- format_splits(readr::read_csv(here::here(
    "data", "splits", "TCGA", paste0(cancer, "_train_splits.csv")
  )))
  test_splits <- format_splits(readr::read_csv(here::here(
    "data", "splits", "TCGA", paste0(cancer, "_test_splits.csv")
  )))
  grid <- benchmark_grid(
    tmp, learners, ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
  )
  bmr <- benchmark(grid)
  bmr <- bmr$score()
  for (model in 0:length(model_names)) {
    if (!dir.exists(here::here(
      "data", "results_reproduced", "survival_functions", "TCGA", cancer, paste0(model_names[(model + 1)], "_gex_only")
    ))) {
      dir.create(
        here::here(
          "data", "results_reproduced", "survival_functions", "TCGA", cancer, paste0(model_names[(model + 1)], "_gex_only")
        ),
        recursive = TRUE
      )
    }
    for (i in 1:25) {
      surv_pred <- data.frame(bmr[(1 + (model * 25):((model + 1) * 25))]$prediction[[i]]$data$distr, check.names = FALSE)
      surv_pred %>% readr::write_csv(
        here::here(
          "data", "results_reproduced", "survival_functions", "TCGA", cancer, paste0(model_names[(model + 1)], "_gex_only"), paste0("split_", i, ".csv")
        )
      )
    }
  }
}