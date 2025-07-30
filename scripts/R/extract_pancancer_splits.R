suppressPackageStartupMessages({
  library(readr)
  library(vroom)
  library(dplyr)
})

# https://stackoverflow.com/questions/7196450/create-a-dataframe-of-unequal-lengths
na.pad <- function(x, len) {
  x[1:len]
}

makePaddedDataFrame <- function(l, ...) {
  maxlen <- max(sapply(l, length))
  data.frame(lapply(l, na.pad, len = maxlen), ...)
}

master_pancan <- vroom::vroom("data_reproduced/TCGA/pancancer_complete_master.csv")

cancer_types_pancan <- unique(master_pancan$clinical_cancer_type)

cancer_offsets <- sapply(cancer_types_pancan, function(cancer) {
  which(master_pancan$clinical_cancer_type == cancer)[1] - 1
})

pancancer_train_splits <- lapply(1:25, function(split_ix) {
  unlist(sapply(1:length(cancer_types_pancan), function(cancer_ix) {
    existing_splits <- unname(unlist(vroom::vroom(paste0(
      "data_reproduced/splits/TCGA/", cancer_types_pancan[cancer_ix], "_train_splits.csv"
    ))[split_ix, ]))
    existing_splits <- existing_splits[!is.na(existing_splits)]
    existing_splits + cancer_offsets[cancer_ix]
  }))
})

pancancer_test_splits <- lapply(1:25, function(split_ix) {
  unlist(sapply(1:length(cancer_types_pancan), function(cancer_ix) {
    existing_splits <- unname(unlist(vroom::vroom(paste0(
      "data_reproduced/splits/TCGA/", cancer_types_pancan[cancer_ix], "_test_splits.csv"
    ))[split_ix, ]))
    existing_splits <- existing_splits[!is.na(existing_splits)]
    existing_splits + cancer_offsets[cancer_ix]
  }))
})

pancancer_train_splits_finalized <- data.frame(t(makePaddedDataFrame(pancancer_train_splits)))
pancancer_test_splits_finalized <- data.frame(t(makePaddedDataFrame(pancancer_test_splits)))

pancancer_train_splits_finalized %>% write_csv(
  "data_reproduced/splits/TCGA/pancancer_train_splits.csv"
)

pancancer_test_splits_finalized %>% write_csv(
  "data_reproduced/splits/TCGA/pancancer_test_splits.csv"
)
