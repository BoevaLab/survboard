#' Get stratified cross-validation indices.
#'
#' @param event integer. Vector containing event indicator to be stratified on.
#' @param n_folds integer. How many folds should be used within the CV.
#' @param n_samples integer. How many samples are within the set to be used CV on.
#'
#' @returns foldids_formatted. Integer. Vector containing the assignment of
#'                                      each sample to its respective CV fold.
get_folds <- function(event, n_folds, n_samples) {
  # Load package within function for parallelisation.
  suppressPackageStartupMessages(library(splitTools))
  foldids <- create_folds(event, k = n_folds, invert = TRUE, type = "stratified")
  foldids_formatted <- rep(1, n_samples)
  for (i in 2:length(foldids)) {
    foldids_formatted[foldids[[i]]] <- i
  }
  return(foldids_formatted)
}


#' Gets indices of blocks for usage with multi-modal models.
#'
#' @param block_order character. Vector containing the (unique) vector of modality names.
#' @param feature_names data.frame. Complete data.frame of the training data.
#'
#' @returns blocks character. Vector containing a mapping from feature to
#'                            which modality it belongs, prefaced with "bp".
#' @example
#' get_block_assignment(c("apple", "pear"), c("apple_feature_1", "pear_feature_1", "apple_feature_2")).
#' Returns: c("bp1", "bp2", "bp1").
get_block_assignment <- function(block_order, feature_names) {
  blocks <- sapply(block_order, function(x) grep(x, feature_names))
  blocks <- blocks[sapply(blocks, length) > 1]
  names(blocks) <- paste0("bp", 1:length(blocks))
  return(blocks)
}

#' Transforms coefficients of a Coxian-based model into a Cox model
#' that can be used to produce survival function estimates.
#' Inspired by the implementation of Herrman et al. (2021).
#'
#' @param coefficients numeric. Named vector containing coefficient estimates
#'                              for each non-zero coefficients.
#' @param data data.frame. Complete data.frame of the training data.
#' @param target Surv. Survival information of the training data.
#'
#' @returns cox_helper coxph.
transform_cox_model <- function(coefficients, data, target) {
  # Package is loaded within function to enable parallelisation.
  suppressPackageStartupMessages(library(survival))

  # Keep only features selected. `drop = FALSE` in case only one feature
  # has been selected.
  train_data <- data[, which(colnames(data) %in% names(coefficients)), drop = FALSE]
  # Combine target and training matrix for usage with `survival::coxph`.
  train_matrix <- cbind(target, train_data)
  colnames(train_matrix)[1:2] <- c("OS_days", "OS")
  # Initialize Cox model at the coefficient estimates - we don't actually
  # fit the model here, since we are only interested in using `coxph` with
  # `pec` to easily produce survival function estimates.
  cox_helper <- coxph(Surv(OS_days, OS) ~ ., x = TRUE, init = coefficients, iter.max = 0, data = data.frame(train_matrix))
  return(cox_helper)
}

#' Calculate priority order based on an adaptive step as described in
#' Herrman et al. (2021). Concretely, we fit a Ridge on each modality
#' individually and then order the modalities based on their mean
#' absolute coefficients of the first step.
#'
#' @param target Surv. Survival information of the training data.
#' @param data data.frame. Complete data.frame of the training data.
#' @param blocks character. Vector containing unique names of each modality name.
#' @param foldid numeric. Vector containing the fold assignments for reproducibility.
#' @param lambda.type character. Whether to use the `lambda.min` or `lambda.1se` rule from `glmnet`.
#' @param favor_clinical logical. Whether clinical should be favored by
#'                                always giving it the highest priority.
#'
#' @returns block_order character. Reordered `blocks` vector, where
#'                                 earlier modalities receive higher priority.
get_prioritylasso_block_order <- function(target, data, blocks, foldid, lambda.type, favor_clinical) {
  suppressPackageStartupMessages({
    library(glmnet)
    library(coefplot)
  })
  mean_absolute_coefficients <- c()
  for (i in 1:length(blocks)) {
    # Run initial Ridge step.
    tmp <- glmnet::cv.glmnet(
      data[, which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == blocks[i])],
      y = target,
      foldid = foldid,
      type.measure = "deviance",
      family = "cox",
      alpha = 0
    )
    mean_absolute_coefficients <- c(mean_absolute_coefficients, mean(abs(coefplot::extract.coef(tmp, lambda.type)[, 1])))
  }
  # Reorder blocks based on mean absolute coefficients of unimodal Ridge fits.
  block_order <- blocks[sort(mean_absolute_coefficients, index.return = TRUE, decreasing = TRUE)$ix]
  # Put clinical first in case it is favored.
  if (favor_clinical) {
    block_order <- c("clinical", block_order[-which(block_order == "clinical")])
  }
  return(block_order)
}



#' Format splits into a usable format for R. Excludes NAs and increments
#' all indices by one to account for the indexing difference between
#' R and Python.
#'
#' @param raw_splits data.frame. Raw data.frame of splits as read from the 
#'                   corresponding CSV files.
#'
#' @returns list. List of length 25 (one element for each outer split),
#' each containing the indices corresponding to the split in question.
format_splits <- function(raw_splits) {
  if (any(is.na(raw_splits))) {
    apply(data.frame(raw_splits), 1, function(x) unname(x[!is.na(x)]) + 1)
  } else {
    x <- unname(as.matrix(raw_splits)) + 1
    split(x, row(x))
  }
}

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
  suppressPackageStartupMessages({
    library(here)
    library(rjson)
    #source(here::here("survboard", "R", "utils", "prod", "utils.R"))
  })
  
  config <- rjson::fromJSON(
    file = here::here("config", "config.json")
  )
  # Various checks to make sure reasonable values are being passed
  # to the get_splits function.
  if (setting != "pancancer") {
    train_splits <- format_splits(readr::read_csv(here::here(
      "data_reproduced", "splits", project, paste0(cancer, "_train_splits.csv")
    )))[[split_number]]
    test_splits <- format_splits(readr::read_csv(here::here(
      "data_reproduced", "splits", project, paste0(cancer, "_test_splits.csv")
    )))[[split_number]]
  }
  else {
      train_splits <- format_splits(readr::read_csv(here::here(
      "data_reproduced", "splits", "TCGA", paste0("pancancer_train_splits.csv")
    )))[[split_number]]
    test_splits <- format_splits(readr::read_csv(here::here(
      "data_reproduced", "splits", "TCGA", paste0("pancancer_test_splits.csv")
    )))[[split_number]]
  }
  
  return(list(train_ix = train_splits, test_ix = test_splits))
}
