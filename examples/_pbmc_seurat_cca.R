## Seurat side of the PBMC two-batch comparison.
## Reads the gene×cell CSVs written by _pbmc_make_two_batches.py,
## runs Seurat::RunCCA, and dumps the embedding for Python to compare.

suppressPackageStartupMessages({
  library(Seurat)
  library(jsonlite)
})

DIR <- "/scratch/users/steorra/analysis/omicverse_dev/py-CCA/examples/_pbmc_pair"

cat("loading...\n")
A <- as.matrix(read.csv(file.path(DIR, "X_A.csv"), row.names = 1, check.names = FALSE))
B <- as.matrix(read.csv(file.path(DIR, "X_B.csv"), row.names = 1, check.names = FALSE))
cat(sprintf("A: %d x %d   B: %d x %d   shared rows: %d\n",
            nrow(A), ncol(A), nrow(B), ncol(B), length(intersect(rownames(A), rownames(B)))))

set.seed(42)
res <- RunCCA(object1 = A, object2 = B,
              standardize = TRUE, num.cc = 30,
              seed.use = 42, verbose = TRUE)

cat(sprintf("ccv shape: %d x %d   d length: %d\n",
            nrow(res$ccv), ncol(res$ccv), length(res$d)))

payload <- list(
  n1 = ncol(A), n2 = ncol(B), num_cc = 30,
  cells_A = colnames(A),
  cells_B = colnames(B),
  ccv = as.numeric(res$ccv),     # column-major
  ccv_dim = dim(res$ccv),
  d   = as.numeric(res$d)
)
writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA, na = "string"),
           file.path(DIR, "seurat_cca.json"))
cat(sprintf("wrote %s\n", file.path(DIR, "seurat_cca.json")))
