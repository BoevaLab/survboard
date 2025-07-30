log <- file(snakemake@log[[1]], open = "wt")
sink(log, type = "output")
sink(log, type = "message")

.libPaths(c("/cluster/customapps/biomed/boeva/dwissel/4.2", .libPaths()))
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
  source(here::here("survboard", "R", "utils", "utils.R"))
})

# Seeding for reproducibility.
set.seed(42)


# Set up parallelisation option.
options(future.globals.onReference = "ignore")
future::plan("sequential")
config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)

# Set up mlr3 pipelines.
remove_constants <- po("removeconstants")
encode <- po("encode", method = "one-hot")
fix_factors <- po("fixfactors")
impute_missing_prediction <- po("imputeoor", affect_columns = selector_type("factor"))
impute <- po("imputeconstant", constant = 0, affect_columns = selector_grep("clinical"))
pipe <- remove_constants %>>% fix_factors
pipe_ohe <- pipe %>>% encode %>>% impute

# Set up mlr3 models to be reproduced.
# All models use a KM model as a fallback for each split
# in case there are any errors throughout the benchmark
learners <- pipe %>>% impute_missing_prediction %>>% po("learner",
  id = "blockforest",
  learner = lrn("surv.blockforest",
    encapsulate = c(train = "evaluate", predict = "evaluate"),
    # fallback = lrn("surv.kaplan"),
    block.method = "BlockForest",
    num.trees = 2000, mtry = NULL, nsets = 300, num.trees.pre = 100, splitrule = "extratrees", always.select.block = 0,
    respect.unordered.factors = "order"
  )
)

for (project in c("TCGA")) {
  # Iterate over all cancers in the project.
  for (cancer in c(snakemake@wildcards[["cancer"]])) {
    set.seed(42)
    # Read in complete modality sample dataset.
    data <- vroom::vroom(
      here::here(
        "data_reproduced", project,
        paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
      )
    )
    # Remove patient_id column and explicitly cast character columns as strings.
    data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
      mutate(across(where(is.character), as.factor))

    data <- data[
      ,
      which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) %in% c("OS", "clinical", "gex"))
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
    # Run training using mlr3.
    learners$train(tmp)
    write.table(data.frame(), file = here::here(
      "results_reproduced", "timings", paste0("blockforest_", snakemake@wildcards[["cancer"]])
    ), col.names = FALSE)
  }
}

sessionInfo()

sink()
sink()
