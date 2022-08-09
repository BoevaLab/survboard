suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3pipelines)
  library(mlr3learners)
  library(mlr3tuning)
  library(paradox)
  library(mlr3proba)
  library(rjson)
  library(dplyr)
  source(here::here("survboard", "R", "learners", "blockforest_learners.R"))
  source(here::here("survboard", "R", "learners", "cv_coxboost_learners.R"))
  source(here::here("survboard", "R", "learners", "cv_lasso_learners.R"))
  source(here::here("survboard", "R", "learners", "cv_prioritylasso_learners.R"))
  source(here::here("survboard", "R", "learners", "ranger_learners.R"))
})

# Set up parallelisation option.
future::plan("multisession")
options(future.globals.onReference = "ignore")

config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)

# Set up mlr3 pipelines.
remove_constants <- po("removeconstants")
encode <- po("encode", method = "treatment")
fix_factors <- po("fixfactors")
impute <- po("imputeconstant", constant = 0, affect_columns = selector_grep("clinical"))
pipe <- remove_constants %>>% fix_factors
pipe_ohe <- pipe %>>% encode %>>% impute

# Seeding for reproducibility.
set.seed(42)

# Model names to be reproduced.
model_names <- c(
  "BlockForest",
  "BlockForest_favoring",
  "CoxBoost",
  "RSF",
  "RSF_favoring",
  "prioritylasso",
  "Lasso"
)

# Set up mlr3 models to be reproduced.
# All models use a KM model as a fallback for each split
# in case there are any errors throughout the benchmark
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
    id = "BlockForest_favoring",
    learner = lrn("surv.blockforest",
      block.method = "BlockForest",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      num.trees = 2000, mtry = NULL, nsets = 100, num.trees.pre = 1500, splitrule = "e#xtratrees", always.select.block = 1
    )
  ),
  pipe_ohe %>>% po("learner",
    id = "CoxBoost",
    learner = lrn("surv.cv_coxboost_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      favor_clinical = FALSE,
      K = 5
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
  pipe %>>% impute_missing_prediction %>>% po("learner",
    id = "ranger_favoring",
    learner = lrn("surv.ranger_custom",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      favor_clinical = TRUE,
      num.trees = 2000,
      splitrule = "extratrees"
    )
  ),
  pipe_ohe %>>% po("learner",
    id = "prioritylasso",
    learner = lrn("surv.cv_prioritylasso",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      block1.penalization = TRUE, lambda.type = "lambda.min", standardize = TRUE, nfolds = 5, cvoffset = TRUE, cvoffsetnfolds = 5, favor_clinical = FALSE
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

for (project in c("TCGA", "ICGC", "TARGET")) {
  # Iterate over all cancers in the project.
  for (cancer in config[[paste0(tolower(project))]]) {
    # Read in complete modality sample dataset.
    data <- vroom::vroom(
      here::here(
        "data", "processed", project,
        paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
      )
    )
    # Remove patient_id column and explicitly cast character columns as strings.
    data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
      mutate(across(where(is.character), as.factor))
    # Create mlr3 task dataset - our event time is indicated by `OS_days`
    # and our event by `OS`. All our datasets are right-censored.
    tmp <- as_task_surv(data,
      time = "OS_days",
      event = "OS",
      type = "right",
      id = cancer
    )
    # Add stratification on the event - this is only necessary if you
    # use mlr3 to tune hyperparameters, but it is good to keep in for
    # safety.
    tmp$add_strata("OS")
    # Iterate over `get_splits` to get full train and test splits for usage in mlr3.
    train_splits <- lapply(1:(config$outer_repetitions * config$outer_splits), function(x) get_splits(cancer = cancer, project = project, n_samples = nrow(data), split_number = x, setting = "standard")[["train_ix"]])
    test_splits <- lapply(1:(config$outer_repetitions * config$outer_splits), function(x) get_splits(cancer = cancer, project = project, n_samples = nrow(data), split_number = x, setting = "standard")[["test_ix"]])

    # Run benchmark using mlr3.
    bmr <- benchmark(benchmark_grid(
      tmp, learners, ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
    ))
    # Score benchmark such that we get access to prediction objects.
    bmr <- bmr$score()
    # Loop over and write out predictions for all models (zero-indexed,
    # since we need this for getting the right survival functions).
    for (model in 0:length(model_names)) {
      if (!dir.exists(here::here(
        "data", "results_reproduced", "survival_functions", project, cancer, model_names[(model + 1)]
      ))) {
        dir.create(
          here::here(
            "data", "results_reproduced", "survival_functions", project, cancer, model_names[(model + 1)]
          ),
          recursive = TRUE
        )
      }
      # Write out CSV file of survival function prediction for each split.
      for (i in 1:(config$outer_repetitions * config$outer_splits)) {
        data.frame(bmr[((1 + (model * (config$outer_repetitions * config$outer_splits))):((model + 1) * (config$outer_repetitions * config$outer_splits)))]$prediction[[i]]$data$distr, check.names = FALSE) %>% readr::write_csv(
          here::here(
            "data", "results_reproduced", "survival_functions", project, cancer, model_names[(model + 1)], paste0("split_", i, ".csv")
          )
        )
      }
    }
  }
}
