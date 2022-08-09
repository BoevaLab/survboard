#' Get splits for SurvBoard benchmark
#'
#' @description
#' Since manually reading in splits can become complicated for the SurvBoard
#' benchmark, we provide a utility function allowing users to easily
#' get splits in their scripts. Please see the `examples` folder for full
#' example usage of this function.
#'
#' @param cancer character. Which cancer dataset splits are desired for.
#'               Must be one of the cancers detailed in Table S2.
#'               Can be "" if setting == "pancancer".
#'
#' @param project character. Which project splits are desired for.
#'                Must be in c("TCGA", "ICGC", "TARGET").

#' @param n_samples integer. The number of samples in the dataset in question.
#'                  We need this for datasets including some or all missing
#'                  modality samples in order to properly include all missing
#'                  modality samples in the training set.
#'
#' @param split_number integer. Which (outer) split is desired. Must be in 1:25.
#'
#' @param setting character. Which setting the model is being trained in.
#'                Must be in c("standard", "missing", "pancancer").
#'
#' @returns list. List element containing two vectors, where the first
#'          vector indicates the indices for the training set and the second
#'          indicates the indices for the test set.
get_splits <- function(cancer,
                       project,
                       n_samples,
                       split_number,
                       setting = "standard") {
  library(here)
  library(rjson)
  source(here::here("survboard", "R", "utils", "prod", "utils.R"))
  config <- rjson::fromJSON(
    file = here::here("config", "config.json")
  )
  if (!project %in% c("TCGA", "ICGC", "TARGET")) {
    stopf("Please make sure you select one of the adequate projects.")
  }
  if (!cancer %in% c(config$icgc_cancers, config$tcga_cancers, config$target_cancers) & setting != "pancancer") {
    stopf("Please make sure you select one of the adequate cancers.")
  }
  if (setting == "pancancer") {
    if (project != "TCGA") {
      stopf("Pancancer data is only available for TCGA.")
    }
    train_splits <- format_splits(readr::read_csv(here::here(
      "data", "splits", "TCGA", "pancancer_train_splits.csv"
    )))[[split_number]]
    test_splits <- format_splits(readr::read_csv(here::here(
      "data", "splits", "TCGA", "pancancer_test_splits.csv"
    )))[[split_number]]
    train_splits <- c(train_splits, (max(max(train_splits), max(test_splits)) + 1):n_samples)
  } else if (setting == "missing") {
    train_splits <- format_splits(readr::read_csv(here::here(
      "data", "splits", "TCGA", paste0(cancer, "_train_splits.csv")
    )))[[split_number]]
    test_splits <- format_splits(readr::read_csv(here::here(
      "data", "splits", "TCGA", paste0(cancer, "_test_splits.csv")
    )))[[split_number]]
    train_splits <- c(train_splits, (max(max(train_splits), max(test_splits)) + 1):n_samples)
  } else {
    train_splits <- format_splits(readr::read_csv(here::here(
      "data", "splits", "TCGA", paste0(cancer, "_train_splits.csv")
    )))[[split_number]]
    test_splits <- format_splits(readr::read_csv(here::here(
      "data", "splits", "TCGA", paste0(cancer, "_test_splits.csv")
    )))[[split_number]]
  }
  return(list(train_ix = train_splits, test_ix = test_splits))
}
