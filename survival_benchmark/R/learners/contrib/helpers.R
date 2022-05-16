# Source: https://github.com/mlr-org/mlr3learners/blob/13a1d0c23ae7d2f6024adffe856cd28973a45a1d/R/helpers.R

# p = probability for levs[2] => matrix with probs for levs[1] and levs[2]
pvec2mat <- function(p, levs) {
  stopifnot(is.numeric(p))
  y <- matrix(c(1 - p, p), ncol = 2L, nrow = length(p))
  colnames(y) <- levs
  y
}


ordered_features <- function(task, learner) {
  cols <- names(learner$state$data_prototype)
  task$data(cols = intersect(cols, task$feature_names))
}


as_numeric_matrix <- function(x) { # for svm / #181
  x <- as.matrix(x)
  if (is.logical(x)) {
    storage.mode(x) <- "double"
  }
  x
}


swap_levels <- function(x) {
  factor(x, levels = rev(levels(x)))
}


rename <- function(x, old, new) {
  if (length(x)) {
    ii <- match(names(x), old, nomatch = 0L)
    names(x)[ii > 0L] <- new[ii]
  }
  x
}


extract_loglik <- function(self) {
  require_namespaces(self$packages)
  if (is.null(self$model)) {
    stopf("Learner '%s' has no model stored", self$id)
  }
  stats::logLik(self$model)
}


opts_default_contrasts <- list(contrasts = c("contr.treatment", "contr.poly"))

# Adapted from GBM package
# https://github.com/cran/gbm/blob/master/R/basehaz.gbm.R
baseline_hazard <- function(t, delta, f.x, t.eval = NULL, smooth = FALSE,
                            cumulative = TRUE, rule = 2) {
  t.unique <- sort(unique(t[delta == 1]))
  alpha <- length(t.unique)
  for (i in 1:length(t.unique)) {
    alpha[i] <- sum(t[delta == 1] == t.unique[i]) /
      sum(exp(f.x[t >= t.unique[i]]))
  }

  if (!smooth && !cumulative) {
    if (!is.null(t.eval)) {
      stop("Cannot evaluate unsmoothed baseline hazard at t.eval.")
    }
  } else {
    if (smooth && !cumulative) {
      lambda.smooth <- supsmu(t.unique, alpha)
    } else {
      if (smooth && cumulative) {
        lambda.smooth <- supsmu(t.unique, cumsum(alpha))
      } else { # (!smooth && cumulative) - THE DEFAULT
        lambda.smooth <- list(x = t.unique, y = cumsum(alpha))
      }
    }
  }


  obj <- if (!is.null(t.eval)) {
    approx(lambda.smooth$x, lambda.smooth$y, xout = t.eval, rule = rule)$y
  } else {
    approx(lambda.smooth$x, lambda.smooth$y, xout = t, rule = rule)$y
  }

  return(obj)
}
