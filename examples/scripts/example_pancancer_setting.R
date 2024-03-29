run_pancancer_setting_example_R <- function(project, cancer, n_cores) {
  # Load base libraries.
  suppressPackageStartupMessages({
    library(mlr3)
    library(mlr3pipelines)
    library(mlr3learners)
    library(mlr3tuning)
    library(paradox)
    library(mlr3proba)
    library(rjson)
    library(dplyr)
    source(here::here("survboard", "R", "utils", "prod", "get_splits.R"))
  })

  if (project != "TCGA" || cancer != "pancancer") {
    stop("Only call this function for the TCGA project (since otherwise, pancancer data is not available) and with cancer == 'pancancer'.")
  }

  # Turn on parallelisation with the number of cores determined by the user.
  future::plan("multisession", workers = n_cores)

  # Read in config file.
  config <- rjson::fromJSON(
    file = here::here("config", "config.json")
  )

  # Remove zero variance features.
  remove_constants <- po("removeconstants")
  # One-hot encode categorical variables.
  encode <- po("encode", method = "treatment")
  fix_factors <- po("fixfactors")
  # Zero impute new factors in the test set (i.e., categories which were)
  # never seen in the test set, but appear in the train set are considered to
  # have no category at all.
  impute <- po("imputeconstant", constant = 0, affect_columns = selector_grep("clinical"))
  # Build pipeline for mlr3.
  pipe_ohe <- remove_constants %>>% fix_factors %>>% encode %>>% impute

  # Set seed for reproducibility. mlr3 handles parallelisation with
  # the future package.
  set.seed(42)

  # TODO: Implement your model as a mlr3 learner here and use it instead
  # to submit to our webservice.
  # NOTE: Your model is expected to handle the multi-modality
  # structure of the data if you would like it to.
  # The columns encode which modality each feature belongs to
  # within their column structure (i.e., `modality1_feature_1`, `modality2_feature_1`, etc).
  learners <- list()
  if (length(learners) == 0) {
    stop("TODO: Please implement a model which can handle missing modalities and use it within mlr3. Since we are not aware of such models in R, we provide this code only as an example.")
  }

  # TODO: Give your model a name - predictions will be written
  # out using this name.
  model_names <- ""
  if (model_names == "") {
    stop("TODO: Please give your model a descriptive name.")
  }



  # Read in complete modality sample dataset.
  data <- data.frame(vroom::vroom(
    here::here(
      "data", "processed", project,
      paste0(cancer, "_complete.csv", collapse = "")
    )
  ), check.names = FALSE)
  data_incomplete <- data.frame(vroom::vroom(
    here::here(
      "data", "processed", project,
      paste0(cancer, "_incomplete.csv", collapse = "")
    )
  ), check.names = FALSE)

  # Concatenate complete and incomplete_samples - IMPORTANT: Do not change
  # the order here, as the splits assume that the incomplete samples follow
  # the complete_samples.
  data <- rbind(data, data_incomplete)

  # Remove patient_id column and explicitly cast character columns as strings.
  data <- data[, -which("patient_id" == colnames(data))] %>%
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
  train_splits <- lapply(1:(config$outer_repetitions * config$outer_splits), function(x) get_splits(cancer = cancer, project = project, n_samples = nrow(data) + 5, split_number = x, setting = "pancancer")[["train_ix"]])
  test_splits <- lapply(1:(config$outer_repetitions * config$outer_splits), function(x) get_splits(cancer = cancer, project = project, n_samples = nrow(data) + 5, split_number = x, setting = "pancancer")[["test_ix"]])

  # Run benchmark using mlr3.
  bmr <- benchmark(benchmark_grid(
    tmp, learners, ResamplingCustom$new()$instantiate(tmp, train_splits, test_splits)
  ))
  # Score benchmark such that we get access to prediction objects.
  bmr <- bmr$score()

  # This loop is only necessary if you train multiple models at the same time
  # but it will still work just fine with a single model.
  for (cancer in config$tcga_cancers) {
    for (model in 0:length(model_names)) {
      # Make sure the requisite folders exist to write out predictions -
      # otherwise, create them.
      if (!dir.exists(here::here(
        "data", "results_submission", "survival_functions", project, cancer, model_names[(model + 1)]
      ))) {
        dir.create(
          here::here(
            "data", "results_submission", "survival_functions", project, cancer, model_names[(model + 1)]
          ),
          recursive = TRUE
        )
      }
      for (i in 1:25) {
        # Write out each survival function to a CSV file, as required by our
        # webservice.
        # NOTE: Our survival functions are output with each column being a 
        # time and each row being a sample. Thus, for the pancancer
        # setting we pick out the test splits belonging to each cancer 
        # in turn.
        data.frame(bmr[((1 + (model * 25)):((model + 1) * 25))]$prediction[[i]]$data$distr[which(data$clinical_cancer_type[test_splits[[i]]] == cancer), ], check.names = FALSE) %>% readr::write_csv(
          here::here(
            "data", "results_submission", "survival_functions", project, cancer, model_names[(model + 1)], paste0("split_", i, ".csv")
          )
        )
      }
    }
  }
}

suppressPackageStartupMessages({
  library(here)
  library(argparse)
})


parser <- ArgumentParser()
parser$add_argument("--project", help = "Which project is being used. Must be one of TCGA, ICGC or TARGET.")
parser$add_argument("--cancer", help = "Which cancer is being used. Must be one of the valid cancer datasets for TCGA, ICGC, or TARGET. See our webservice or paper for details.")
parser$add_argument("--n_cores", help = "How many cores you would like to use.")
args <- parser$parse_args()

run_pancancer_setting_example_R(
  project = args$project,
  cancer = args$cancer,
  n_cores = args$n_cores
)
