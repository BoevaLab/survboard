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