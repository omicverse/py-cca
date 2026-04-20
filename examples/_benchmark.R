## Time Seurat::RunCCA on the PBMC fixture and on three synthetic sizes.
suppressPackageStartupMessages({
  library(Seurat)
  library(jsonlite)
})

DIR <- "/scratch/users/steorra/analysis/omicverse_dev/py-CCA/examples/_pbmc_pair"
RESULTS <- list()

run_one <- function(label, X, Y, num_cc, n_repeats = 3) {
  times <- numeric(n_repeats)
  for (i in seq_len(n_repeats)) {
    t0 <- proc.time()
    res <- RunCCA(object1 = X, object2 = Y,
                  standardize = TRUE, num.cc = num_cc,
                  seed.use = 42, verbose = FALSE)
    t1 <- proc.time()
    times[i] <- (t1 - t0)["elapsed"]
  }
  best <- min(times)
  cat(sprintf("[%s] num_cc=%d  shape=%dx%d+%dx%d  best %.3fs (median %.3fs over %d runs)\n",
              label, num_cc, nrow(X), ncol(X), nrow(Y), ncol(Y),
              best, median(times), n_repeats))
  list(label = label, num_cc = num_cc,
       n1 = ncol(X), n2 = ncol(Y), n_features = nrow(X),
       times_seconds = as.numeric(times),
       best_seconds = best, median_seconds = median(times))
}

set.seed(42)

# 1) PBMC fixture (real data)
A <- as.matrix(read.csv(file.path(DIR, "X_A.csv"), row.names = 1, check.names = FALSE))
B <- as.matrix(read.csv(file.path(DIR, "X_B.csv"), row.names = 1, check.names = FALSE))
RESULTS[[length(RESULTS) + 1]] <- run_one("pbmc", A, B, 30)

# 2) Synthetic, scaled
make_pair <- function(n_features, n1, n2, n_shared = 20, noise = 0.1) {
  Z1 <- matrix(rnorm(n_shared * n1), n_shared, n1)
  Z2 <- matrix(rnorm(n_shared * n2), n_shared, n2)
  W  <- matrix(rnorm(n_features * n_shared), n_features, n_shared)
  X  <- W %*% Z1 + noise * matrix(rnorm(n_features * n1), n_features)
  Y  <- W %*% Z2 + noise * matrix(rnorm(n_features * n2), n_features)
  rownames(X) <- paste0("g", 1:n_features); rownames(Y) <- paste0("g", 1:n_features)
  colnames(X) <- paste0("c1_", 1:n1);       colnames(Y) <- paste0("c2_", 1:n2)
  list(X = X, Y = Y)
}

for (cfg in list(
  list(label = "small_2k_500", n_features = 2000, n1 = 250, n2 = 250),
  list(label = "med_2k_2k",    n_features = 2000, n1 = 1000, n2 = 1000),
  list(label = "large_2k_10k", n_features = 2000, n1 = 5000, n2 = 5000)
)) {
  d <- make_pair(cfg$n_features, cfg$n1, cfg$n2)
  RESULTS[[length(RESULTS) + 1]] <- run_one(cfg$label, d$X, d$Y, 30, 3)
}

writeLines(toJSON(RESULTS, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           file.path(DIR, "bench_seurat.json"))
cat(sprintf("\nwrote %s\n", file.path(DIR, "bench_seurat.json")))
