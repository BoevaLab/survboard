# All of the following adapted from pycox: https://github.com/havakv/pycox/blob/master/pycox/evaluation/concordance.py
is_comparable <- function(t_i, t_j, d_i, d_j) {
  return((t_i < t_j) & d_i) | ((t_i == t_j) & (d_i | d_j))
}


is_comparable_antolini <- function(t_i, t_j, d_i, d_j) {
  return((t_i < t_j) & d_i) | ((t_i == t_j) & d_i & (d_j == 0))
}

is_concordant <- function(s_i, s_j, t_i, t_j, d_i, d_j) {
  conc <- 0
  if (t_i < t_j) {
    conc <- (s_i < s_j) + (s_i == s_j) * 0.5
  } else if (t_i == t_j) {
    if (d_i & d_j) {
      conc <- 1. - (s_i != s_j) * 0.5
    } else if (d_i) {
      conc <- (s_i < s_j) + (s_i == s_j) * 0.5 # different from RSF paper.
    } else if (d_j) {
      conc <- (s_i > s_j) + (s_i == s_j) * 0.5 # different from RSF paper.
    }
  }
  return(conc * is_comparable(t_i, t_j, d_i, d_j))
}

is_concordant_antolini <- function(s_i, s_j, t_i, t_j, d_i, d_j) {
  return((s_i < s_j) & is_comparable_antolini(t_i, t_j, d_i, d_j))
}


sum_comparable <- function(t, d, is_comparable_func) {
  n <- length(t)
  count <- 0
  for (i in 1:n) {
    for (j in 1:n) {
      if (j != i) {
        count <- count + is_comparable_func(t[i], t[j], d[i], d[j])
      }
    }
  }
  return(count)
}

sum_concordant <- function(s, t, d) {
  n <- length(t)
  count <- 0
  for (i in 1:n) {
    for (j in 1:n) {
      if (j != i) {
        count <- count + is_concordant(s[i, i], s[i, j], t[i], t[j], d[i], d[j])
      }
    }
  }
  return(count)
}

sum_concordant_disc <- function(s, t, d, s_idx, is_concordant_func) {
  n <- length(t)
  count <- 0
  for (i in 1:n) {
    idx <- s_idx[i]
    for (j in 1:n) {
      if (j != i) {
        count <- count + is_concordant_func(s[idx, i], s[idx, j], t[i], t[j], d[i], d[j])
      }
    }
  }
  return(count)
}
