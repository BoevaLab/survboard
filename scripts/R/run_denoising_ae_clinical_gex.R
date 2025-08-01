suppressPackageStartupMessages({
  library(compound.Cox)
  library(keras)
  library(fastDummies)
  library(dplyr)
  library(survival)
  library(purrr)
  library(glmnet)
  library(tibble)
  library(coefplot)
  library(vroom)
  library(readr)
  source(here::here("survboard", "R", "utils", "utils.R"))
})

set.seed(42)

clamp <- function(x, a, b) pmax(a, pmin(x, b))

config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)

early_stop <- keras::callback_early_stopping(
  monitor = "val_loss",
  min_delta = 0.001, patience = 5, restore_best_weights = TRUE,
  verbose = 1
)

uni.selection_fixed <- function(t.vec, d.vec, X.mat, P.value = 0.001, K = 10, score = TRUE,
                                d0 = 0, randomize = FALSE, CC.plot = FALSE, permutation = FALSE,
                                M = 200) {
  n <- length(t.vec)
  if (randomize == TRUE) {
    rand <- sample(1:n)
    t.vec <- t.vec[rand]
    d.vec <- d.vec[rand]
    X.mat <- X.mat[rand, ]
  }
  p <- ncol(X.mat)
  if (score == TRUE) {
    res <- uni.score(t.vec, d.vec, X.mat, d0)
  } else {
    res <- uni.Wald(t.vec, d.vec, X.mat)
  }
  mask <- !is.na(res$P)
  res$P <- res$P[mask]
  res$beta <- res$beta[mask]
  res$Z <- res$Z[mask]

  temp <- res$P < P.value
  if (sum(temp) == 0) {
    stop("no feature is selected; please increase P.value")
  }
  Beta <- res$beta[temp]
  Z <- res$Z[temp]
  P <- res$P[temp]
  X.cut <- as.matrix(X.mat[, colnames(X.mat) %in% names(temp[which(temp)])])

  q <- ncol(X.cut)
  if (score == TRUE) {
    w <- Z
  } else {
    w <- Beta
  }
  CC <- X.cut %*% w
  c.index0 <- 1 - concordance(Surv(t.vec, d.vec) ~ CC)$concordance
  atr_t <- (matrix(t.vec, n, n, byrow = TRUE) >= matrix(
    t.vec,
    n, n
  ))
  l.func <- function(g) {
    l <- sum((d.vec * CC * g)) - sum(d.vec * log(atr_t %*%
      exp(CC * g)))
    as.numeric(l)
  }
  CC.CV <- CC.test <- NULL
  CVL <- RCVL1 <- RCVL2 <- 0
  g_kk_vec <- numeric(K)
  folds <- cut(seq(1, n), breaks = K, labels = FALSE)
  for (k in 1:K) {
    fold_k <- which(folds == k)
    t_k <- t.vec[-fold_k]
    d_k <- d.vec[-fold_k]
    CC_k <- CC[-fold_k]
    X_k <- X.mat[-fold_k, ]
    n_k <- length(t_k)
    if (score == TRUE) {
      res <- uni.score(t_k, d_k, X_k, d0)
    } else {
      res <- uni.Wald(t_k, d_k, X_k)
    }
    mask <- !is.na(res$P)
    res$P <- res$P[mask]
    res$beta <- res$beta[mask]
    res$Z <- res$Z[mask]
    temp_k <- res$P < P.value
    if (sum(temp_k) == 0) {
      CC_kk <- rep(0, n)
      warning("no feature is selected in a cross-validation fold; please increase P.value")
    } else {
      if (score == TRUE) {
        w_k <- setNames(clamp(res$Z[temp_k], -5, 5), names(res$Z[temp_k]))
      } else {
        w_k <- res$beta[temp_k]
      }
      CC_kk <- as.matrix(X.mat[, colnames(X.mat) %in% names(w_k)]) %*% w_k
    }
    CC.test <- c(CC.test, CC_kk[fold_k])
    res_k <- coxph(Surv(t_k, d_k) ~ CC_k)
    RCVL1 <- RCVL1 + l.func(res_k$coef) - res_k$loglik[2]
    if (score == TRUE) {
      w_CV <- setNames(clamp(uni.score(t_k, d_k, X.cut[-fold_k, ], d0)$Z, -5, 5), names(uni.score(t_k, d_k, X.cut[-fold_k, ], d0)$Z))
    } else {
      w_CV <- uni.Wald(t_k, d_k, X.cut[-fold_k, ])$beta
    }
    mask <- !is.na(w_CV)

    CC.CV <- c(CC.CV, X.cut[fold_k, mask] %*% as.matrix(w_CV[mask]))
    CC.CV_k <- X.cut[-fold_k, mask] %*% as.matrix(w_CV[mask])
    res_CV_k <- coxph(Surv(t_k, d_k) ~ CC.CV_k)
    RCVL2 <- RCVL2 + l.func(res_CV_k$coef) - res_CV_k$loglik[2]
    res_kk <- coxph(Surv(t_k, d_k) ~ CC_kk[-fold_k])
    l_kk.func <- function(g) {
      l <- sum((CC_kk * g)[d.vec == 1]) - sum((log(atr_t %*%
        exp(CC_kk * g)))[d.vec == 1])
      as.numeric(l)
    }
    CVL <- CVL + l_kk.func(res_kk$coef) - as.numeric(res_kk$loglik[2])
  }
  c.index1 <- 1 - concordance(Surv(t.vec, d.vec) ~ CC.CV)$concordance
  c.index2 <- 1 - concordance(Surv(t.vec, d.vec) ~ CC.test)$concordance
  c.index <- c(
    `No cross-validation` = c.index0, `Incomplete cross-validation` = c.index1,
    `Full cross-validation` = c.index2
  )
  if (CC.plot == TRUE) {
    par(mfrow = c(1, 2))
    plot(CC, CC.test,
      xlab = "CC (Not cross-validated)",
      ylab = "CC (Fully cross-validated)"
    )
    plot(CC.CV, CC.test,
      xlab = "CC (Only estimation cross-validated)",
      ylab = "CC (Fully cross-validated)"
    )
  }
  FDR.perm <- f.perm <- NULL
  if (permutation == TRUE) {
    q.perm <- numeric(M)
    for (i in 1:M) {
      set.seed(i)
      if (score == TRUE) {
        res <- uni.score(t.vec, d.vec, X.mat[sample(1:n), ], d0)
      } else {
        res <- uni.Wald(t.vec, d.vec, X.mat[sample(1:n), ])
      }
      mask <- !is.na(res$P)
      res$P <- res$P[mask]
      res$beta <- res$beta[mask]
      res$Z <- res$Z[mask]
      q.perm[i] <- sum(res$P < P.value)
    }
    f.perm <- mean(q.perm)
    FDR.perm <- f.perm / q
  }
  list(
    beta = Beta[order(P)], Z = Z[order(P)], P = P[order(P)],
    CVL = c(CVL = CVL, RCVL1 = RCVL1, RCVL2 = RCVL2), Genes = c(
      `No. of genes` = p,
      `No. of selected genes` = q, `No. of falsely selected genes` = f.perm
    ),
    FDR = c(
      `P.value * (No. of genes)` = P.value * p / q,
      Permutation = FDR.perm
    )
  )
}


zeros_percentage <- 0.3
n_mirna <- 300
n_rppa <- 100
n_other <- 500
fullAE_features <- 50
activation_function <- "sigmoid"

corrupt_with_ones <- function(x) {
  n_to_sample <- floor(length(x) * zeros_percentage)
  elements_to_corrupt <- sample(seq_along(x), n_to_sample, replace = FALSE)
  x[elements_to_corrupt] <- 0
  return(x)
}

denoising_zeros_func <- function(train_data, test_data, num_features) {
  inputs_currupted_ones <- train_data %>%
    as.data.frame() %>%
    purrr::map_df(corrupt_with_ones)
  features <- as.matrix(train_data)
  inputs_currupted_ones <- as.matrix(inputs_currupted_ones)
  test_data <- as.matrix(test_data)
  model1 <- keras_model_sequential()
  model1 %>%
    layer_dense(units = num_features, activation = activation_function, input_shape = ncol(inputs_currupted_ones), name = "BottleNeck") %>%
    layer_dense(units = ncol(inputs_currupted_ones))

  model1 %>% keras::compile(
    loss = "mean_squared_error",
    optimizer = optimizer_adam(lr = 0.001)
  )

  history <- model1 %>% keras::fit(
    x = inputs_currupted_ones, y = features,
    epochs = 100, validation_split = 0.1,
    callbacks = list(early_stop)
  )

  intermediate_layer_model1 <- keras_model(inputs = model1$input, outputs = get_layer(model1, "BottleNeck")$output)
  denoising_zeros_list <- list(
    predict(intermediate_layer_model1, inputs_currupted_ones),
    predict(intermediate_layer_model1, test_data)
  )
  train_RMSE <- evaluate(model1, features, inputs_currupted_ones)
  test_RMSE <- evaluate(model1, test_data, test_data)
  return(denoising_zeros_list)
}

elastic_net_func <- function(train_data, test_data, time, status) {
  cv.fit <- cv.glmnet(train_data, Surv(time, status),
    alpha = 0.5,
    family = "cox", type.measure = "C"
  )
  colnames(train_data) <- paste0("V", 1:ncol(train_data))
  colnames(test_data) <- colnames(train_data)

  tmp <- coefplot::extract.coef(cv.fit)
  coefficients <- tmp[, 1]
  names(coefficients) <- rownames(tmp)
  cox_helper <- transform_cox_model(coefficients, train_data, Surv(time, status))
  newdata <- data.frame(test_data)[, colnames(test_data) %in% names(coefficients), drop = FALSE]
  surv <- data.frame(pec::predictSurvProb(cox_helper, newdata, sort(unique(cox_helper$y[, 1]))))
  colnames(surv) <- sort(unique(cox_helper$y[, 1]))
  return(surv)
}


linear_featureselection_func <- function(train_data, test_data, time, survival, num_features) {
  associat <- uni.selection_fixed(
    t.vec = time, d.vec = survival, X.mat = train_data,
    P.value = 0.8, randomize = TRUE, K = 5
  )
  associat$P <- associat$P[1:min(num_features, length(associat$P))]
  col_filtered <- rownames(as.data.frame(associat$P))

  train_data <- data.frame(train_data, check.names = FALSE) %>% dplyr::select(all_of(col_filtered))
  test_data <- data.frame(test_data, check.names = FALSE) %>% dplyr::select(all_of(col_filtered))
  test_data <- as.matrix(test_data)

  lfs_list <- list(train_data, test_data)
  return(lfs_list)
}

options <- commandArgs(trailingOnly = TRUE)

for (project in c(options[1])) {
  for (cancer in c(options[2])) {
    set.seed(42)
    # Read in complete modality sample dataset.
    target_dir <- paste0("results_reproduced/survival_functions/clinical_gex/", project, "/", cancer, "/", "multimodal_nsclc")
    if (!dir.exists(target_dir)) {
      dir.create(target_dir)
    }
    data <- vroom::vroom(
      here::here(
        "data_reproduced", project,
        paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
      )
    )
    # Remove patient_id column and explicitly cast character columns as strings.
    data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
      mutate(across(where(is.character), as.factor))

    # Iterate over `get_splits` to get full train and test splits for usage in mlr3.
    train_splits <- lapply(1:(config$outer_repetitions * config$outer_splits), function(x) get_splits(cancer = cancer, project = project, n_samples = nrow(data), split_number = x, setting = "standard")[["train_ix"]])
    test_splits <- lapply(1:(config$outer_repetitions * config$outer_splits), function(x) get_splits(cancer = cancer, project = project, n_samples = nrow(data), split_number = x, setting = "standard")[["test_ix"]])

    for (split in 1:25) {
      train_ix <- train_splits[[split]]
      test_ix <- test_splits[[split]]

      combined_data <- dummy_cols(data[, -c(which("OS" == colnames(data)), which("OS_days" == colnames(data)))], remove_first_dummy = TRUE, remove_selected_columns = TRUE)
      train_data <- (combined_data[train_ix, ])
      test_data <- (combined_data[test_ix, ])

      zero_variance_mask <- (apply(train_data, 2, var) == 0) | (apply(test_data, 2, var) == 0)
      train_data <- train_data[, !zero_variance_mask]
      test_data <- test_data[, !zero_variance_mask]

      train_label <- data[train_ix, c(which("OS" == colnames(data)), which("OS_days" == colnames(data)))]
      test_label <- data[test_ix, c(which("OS" == colnames(data)), which("OS_days" == colnames(data)))]

      modalities <- unique(sapply(strsplit(colnames(train_data), "\\_"), function(x) x[[1]]))
      for (modality in c("gex", "clinical")) {
        if (modality == "gex" & "gex" %in% modalities) {
          mod_train <- train_data[, which(sapply(strsplit(colnames(train_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_test <- test_data[, which(sapply(strsplit(colnames(test_data), "\\_"), function(x) x[[1]]) == modality)]

          mod_train_vared <- mod_train[, apply(mod_train, 2, var) > median(apply(mod_train, 2, var))]

          tmp <- linear_featureselection_func(
            scale(mod_train_vared), scale(mod_test), train_label$OS_days, train_label$OS, min(n_other, ncol(mod_train))
          )
          transformed_matrix_train <- tmp[[1]]
          transformed_matrix_test <- tmp[[2]]
        }
        if (modality == "clinical" & "clinical" %in% modalities) {
          transformed_matrix_train <- cbind(transformed_matrix_train, train_data[, which(sapply(strsplit(colnames(train_data), "\\_"), function(x) x[[1]]) == modality)])
          transformed_matrix_test <- cbind(transformed_matrix_test, test_data[, which(sapply(strsplit(colnames(test_data), "\\_"), function(x) x[[1]]) == modality)])
        }
        if (modality == "mirna" & "mirna" %in% modalities) {
          mod_train <- train_data[, which(sapply(strsplit(colnames(train_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_test <- test_data[, which(sapply(strsplit(colnames(test_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_train_vared <- mod_train[, apply(mod_train, 2, var) > median(apply(mod_train, 2, var))]
          tmp <- linear_featureselection_func(
            scale(mod_train_vared), scale(mod_test), train_label$OS_days, train_label$OS, min(n_mirna, ncol(mod_train))
          )
          transformed_matrix_train <- cbind(transformed_matrix_train, tmp[[1]])
          transformed_matrix_test <- cbind(transformed_matrix_test, tmp[[2]])
        }
        if (modality == "meth" & "meth" %in% modalities) {
          mod_train <- train_data[, which(sapply(strsplit(colnames(train_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_test <- test_data[, which(sapply(strsplit(colnames(test_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_train_vared <- mod_train[, apply(mod_train, 2, var) > median(apply(mod_train, 2, var))]
          tmp <- linear_featureselection_func(
            scale(mod_train_vared), scale(mod_test), train_label$OS_days, train_label$OS, min(n_other, ncol(mod_train))
          )
          transformed_matrix_train <- cbind(transformed_matrix_train, tmp[[1]])
          transformed_matrix_test <- cbind(transformed_matrix_test, tmp[[2]])
        }
        if (modality == "rppa" & "rppa" %in% modalities) {
          mod_train <- train_data[, which(sapply(strsplit(colnames(train_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_test <- test_data[, which(sapply(strsplit(colnames(test_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_train_vared <- mod_train[, apply(mod_train, 2, var) > median(apply(mod_train, 2, var))]
          tmp <- linear_featureselection_func(
            scale(mod_train_vared), scale(mod_test), train_label$OS_days, train_label$OS, min(n_rppa, ncol(mod_train))
          )
          transformed_matrix_train <- cbind(transformed_matrix_train, tmp[[1]])
          transformed_matrix_test <- cbind(transformed_matrix_test, tmp[[2]])
        }
        if (modality == "mut" & "mut" %in% modalities) {
          mod_train <- train_data[, which(sapply(strsplit(colnames(train_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_test <- test_data[, which(sapply(strsplit(colnames(test_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_train_vared <- mod_train[, apply(mod_train, 2, var) > median(apply(mod_train, 2, var))]
          tmp <- linear_featureselection_func(
            scale(mod_train_vared), scale(mod_test), train_label$OS_days, train_label$OS, min(n_other, ncol(mod_train))
          )
          transformed_matrix_train <- cbind(transformed_matrix_train, tmp[[1]])
          transformed_matrix_test <- cbind(transformed_matrix_test, tmp[[2]])
        }
        if (modality == "cnv" & "cnv" %in% modalities) {
          mod_train <- train_data[, which(sapply(strsplit(colnames(train_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_test <- test_data[, which(sapply(strsplit(colnames(test_data), "\\_"), function(x) x[[1]]) == modality)]
          mod_train_vared <- mod_train[, apply(mod_train, 2, var) > median(apply(mod_train, 2, var))]
          tmp <- linear_featureselection_func(
            scale(mod_train_vared), scale(mod_test), train_label$OS_days, train_label$OS, min(n_other, ncol(mod_train))
          )
          transformed_matrix_train <- cbind(transformed_matrix_train, tmp[[1]])
          transformed_matrix_test <- cbind(transformed_matrix_test, tmp[[2]])
        }
      }

      ae_projections <- denoising_zeros_func(transformed_matrix_train, transformed_matrix_test, fullAE_features)
      ae_projections_train <- ae_projections[[1]]
      ae_projections_test <- ae_projections[[2]]
      survival_function <- elastic_net_func(ae_projections_train, ae_projections_test, train_label$OS_days, train_label$OS)
      survival_function %>% write_csv(
        here::here(
          "results_reproduced", "survival_functions", "clinical_gex", project, cancer, "multimodal_nsclc", paste0("split_", split, ".csv")
        )
      )
    }
  }
}

sessionInfo()
