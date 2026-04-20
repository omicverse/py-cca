## Run Seurat::RunCCA on the same synthetic + real data the Python tests use,
## and dump everything py-CCA's parity test needs to compare against.
##
## Usage:
##   Rscript tests/r_parity_dump.R
## Requires: Seurat (>= 4), SeuratObject, irlba, jsonlite

suppressPackageStartupMessages({
  library(Seurat)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "/scratch/users/steorra/analysis/omicverse_dev/py-CCA/tests/_rparity"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(20260418)
make_pair <- function(n_features = 200, n1 = 80, n2 = 120, n_shared = 10, noise = 0.1) {
  Z1 <- matrix(rnorm(n_shared * n1), nrow = n_shared, ncol = n1)
  Z2 <- matrix(rnorm(n_shared * n2), nrow = n_shared, ncol = n2)
  W  <- matrix(rnorm(n_features * n_shared), nrow = n_features, ncol = n_shared)
  X  <- W %*% Z1 + noise * matrix(rnorm(n_features * n1), nrow = n_features)
  Y  <- W %*% Z2 + noise * matrix(rnorm(n_features * n2), nrow = n_features)
  rownames(X) <- paste0("g", 1:n_features); rownames(Y) <- paste0("g", 1:n_features)
  colnames(X) <- paste0("c1_", 1:n1);       colnames(Y) <- paste0("c2_", 1:n2)
  list(X = X, Y = Y)
}

datasets <- list(
  small  = make_pair(100, 30, 40, 5, 0.05),
  medium = make_pair(300, 80, 120, 10, 0.1),
  large  = make_pair(500, 150, 200, 15, 0.15)
)

manifest <- list()

for (dname in names(datasets)) {
  ds <- datasets[[dname]]
  cat(sprintf("[%s] features=%d  n1=%d  n2=%d\n",
              dname, nrow(ds$X), ncol(ds$X), ncol(ds$Y)))
  # Save inputs as flat CSV
  write.table(ds$X, file.path(out_dir, paste0("X_", dname, ".csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(ds$Y, file.path(out_dir, paste0("Y_", dname, ".csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)

  for (num_cc in c(5, 10, 20)) {
    if (num_cc >= min(ncol(ds$X), ncol(ds$Y))) next
    set.seed(42)
    res <- RunCCA(object1 = ds$X, object2 = ds$Y, standardize = TRUE,
                  num.cc = num_cc, seed.use = 42, verbose = FALSE)
    payload <- list(
      dataset = dname, num_cc = num_cc,
      n1 = ncol(ds$X), n2 = ncol(ds$Y), n_features = nrow(ds$X),
      ccv = as.numeric(res$ccv),     # column-major
      ccv_dim = dim(res$ccv),
      d   = as.numeric(res$d)
    )
    fname <- sprintf("%s_cc%d.json", dname, num_cc)
    writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA, na = "string"),
               file.path(out_dir, fname))
    manifest[[length(manifest) + 1]] <- list(
      dataset = dname, num_cc = num_cc, file = fname,
      first_d = res$d[1], last_d = res$d[num_cc]
    )
  }
}

writeLines(toJSON(manifest, pretty = TRUE, auto_unbox = TRUE, digits = NA, na = "string"),
           file.path(out_dir, "manifest.json"))
cat(sprintf("DONE — %d records in %s\n", length(manifest), out_dir))
