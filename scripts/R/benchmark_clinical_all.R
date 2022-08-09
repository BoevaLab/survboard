suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3pipelines)
  library(mlr3learners)
  library(mlr3tuning)
  library(paradox)
  library(mlr3proba)
  library(rjson)
  library(dplyr)
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

model_names <- c(
  "Cox"
)

learners <- list(
  remove_constants %>>% encode %>>% po("learner",
    id = "Cox",
    learner = lrn("surv.coxph")
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

    # Keep only survival information (OS and OS_days) and clinical variables.
    data <- data[
      ,
      which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) %in% c("clinical", "OS"))
    ]

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
