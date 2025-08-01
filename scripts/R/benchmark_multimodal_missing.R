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
  source(here::here("survboard", "R", "learners", "cv_lasso_learners.R"))
  source(here::here("survboard", "R", "learners", "cv_prioritylasso_learners.R"))
  source(here::here("survboard", "R", "learners", "ranger_learners.R"))
  source(here::here("survboard", "R", "utils", "utils.R"))

})
# Seeding for reproducibility.
set.seed(42)

# Set up parallelisation option.
future::plan("multicore", workers=25L)
options(future.globals.onReference = "ignore")

config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)

# Set up mlr3 pipelines.
remove_constants <- po("removeconstants")
encode <- po("encode", method = "one-hot")
fix_factors <- po("fixfactors")
impute_missing_prediction <- po("imputeoor", affect_columns = selector_type("factor"))
impute <- po("imputeconstant", constant = 0, affect_columns = selector_grep("clinical"))
impute_missing_blocks <- po("imputeconstant", constant = 0, affect_columns = selector_type("numeric"))
pipe <- remove_constants %>>% fix_factors
pipe_ohe <- pipe %>>% encode %>>% impute

# Model names to be reproduced.
model_names <- c(
 # "blockforest",
  "priority_elastic_net"
)

# Set up mlr3 models to be reproduced.
# All models use a KM model as a fallback for each split
# in case there are any errors throughout the benchmark
learners <- list(
  pipe_ohe %>>% po("learner",
    id = "priority_elastic_net",
    learner = lrn("surv.cv_prioritylasso",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan"),
      block1.penalization = TRUE, lambda.type = "lambda.min", 
      standardize = TRUE, nfolds = 5, cvoffset = FALSE, cvoffsetnfolds = 5, favor_clinical = FALSE,
      alpha = 0.9,
      nlambda = 100,
      mcontrol = missing.control(
            handle.missingdata = "impute.offset",
            offset.firstblock = "zero",
            impute.offset.cases = "available.cases",
            nfolds.imputation = 5,
            lambda.imputation = "lambda.min",
            perc.comp.cases.warning = 0.3,
            threshold.available.cases = 30,
            select.available.cases = "max"
        )
    )
  )
)

for (project in c("METABRIC", "TCGA", "ICGC", "TARGET")) {
  # Iterate over all cancers in the project.
  for (cancer in config[[paste0(tolower(project), "_cancers")]]) {
    # Read in complete modality sample dataset.
    data <- vroom::vroom(
      here::here(
        "data_reproduced", project,
        paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
      )
    )
    data_missing <- vroom::vroom(
      here::here(
        "data_reproduced", project,
        paste0(cancer, "_data_incomplete_modalities_preprocessed.csv", collapse = "")
      )
    )
    # Remove patient_id column and explicitly cast character columns as strings.
    data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
      mutate(across(where(is.character), as.factor))
    data_combined <- rbind(data, data.frame(data_missing[, -which("patient_id" == colnames(data_missing))]) %>%
      mutate(across(where(is.character), as.factor)))
    # Create mlr3 task dataset - our event time is indicated by `OS_days`
    # and our event by `OS`. All our datasets are right-censored.
    tmp <- as_task_surv(data_combined,
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
    train_splits <- lapply(train_splits, function(x) c(x, ((nrow(data) + 1):(nrow(data)+nrow(data_missing)))))

    # Run benchmark using mlr3.
    bmr <- benchmark(benchmark_grid(
      tmp, learners, ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
    ))
    # Score benchmark such that we get access to prediction objects.
    bmr <- bmr$score()
    # Loop over and write out predictions for all models (zero-indexed,
    # since we need this for getting the right survival functions).
    for (model in 0:(length(model_names)-1)) {
      if (!dir.exists(here::here(
        "results_reproduced", "survival_functions", "full_missing", project, cancer, model_names[(model + 1)]
      ))) {
        dir.create(
          here::here(
            "results_reproduced", "survival_functions", "full_missing", project, cancer, model_names[(model + 1)]
          ),
          recursive = TRUE
        )
      }
      # Write out CSV file of survival function prediction for each split.
      for (i in 1:(config$outer_repetitions * config$outer_splits)) {
        data.frame(bmr[((1 + (model * (config$outer_repetitions * config$outer_splits))):((model + 1) * (config$outer_repetitions * config$outer_splits)))]$prediction[[i]]$data$distr, check.names = FALSE) %>% readr::write_csv(
          here::here(
            "results_reproduced", "survival_functions", "full_missing", project, cancer, model_names[(model + 1)], paste0("split_", i, ".csv")
          )
        )
      }
    }
  }
}

sessionInfo()
