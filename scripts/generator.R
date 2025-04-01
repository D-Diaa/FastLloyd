# Load Required Libraries
required_packages <- c("clusterGeneration", "doParallel", "foreach", "logging")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages)
}

library(clusterGeneration)
library(doParallel)
library(foreach)
library(logging)

# Initialize Logging
basicConfig()
addHandler(writeToFile, file = "cluster_generation.log")

# Set Fixed Parameters
n <- 10000
numClust_ls <- c(2, 4, 8, 16, 32)
numNonNoisy_ls <- c(2, 4, 8, 16, 32, 64, 128, 256, 512)
numReplicate <- 3
outputLogFlag <- FALSE
outputEmpirical <- FALSE
outputInfo <- FALSE

# Directory to Save Generated Files
output_dir <- "."

# Function to calculate cluster sizes based on ratio 1:2:...:k
calculate_cluster_sizes <- function(k, total_size) {
  ratios <- 1:k
  proportions <- ratios / sum(ratios)
  sizes <- round(proportions * total_size)
  # Adjust for rounding errors to ensure total is exactly total_size
  sizes[k] <- sizes[k] + (total_size - sum(sizes))
  return(sizes)
}

# Generate All Parameter Combinations
param_grid <- expand.grid(
  numClust = numClust_ls,
  numNonNoisy = numNonNoisy_ls,
  stringsAsFactors = FALSE
)

# Precompute Existing File Prefixes
loginfo("Scanning existing files to avoid regeneration.")
existing_files <- list.files(path = output_dir, pattern = "^SynthNew_\\d+_\\d+", full.names = FALSE)
existing_prefixes <- unique(sub("_\\d+\\.txt$", "", existing_files))

# Add file prefix to param_grid
param_grid$file_prefix <- paste0("SynthNew_", param_grid$numClust, "_", param_grid$numNonNoisy)

# Filter out parameter sets that have already been generated
to_generate <- !(param_grid$file_prefix %in% existing_prefixes)
filtered_param_grid <- param_grid[to_generate, ]

# Function to Generate and Save Clusters
generate_cluster <- function(params) {
  # Calculate cluster sizes based on ratio 1:2:...:k
  cluster_sizes <- calculate_cluster_sizes(params$numClust, n)

  # Generate random number of outliers [0, 100]
  num_outliers <- sample(0:100, 1)

  # Adjust cluster sizes to account for outliers
  if (num_outliers > 0) {
    reduction_per_cluster <- round(num_outliers * (cluster_sizes / n))
    cluster_sizes <- cluster_sizes - reduction_per_cluster
  }

  # Random separation value from [0.16, 0.26]
  sepVal <- runif(1, 0.16, 0.26)

  file_prefix <- params$file_prefix

  # Generate Cluster
  tryCatch({
    loginfo(paste("Generating:", file_prefix, "with separation", sepVal, "and", num_outliers, "outliers"))
    genRandomClust(
      numClust = params$numClust,
      sepVal = sepVal,
      numNonNoisy = params$numNonNoisy,
      numReplicate = numReplicate,
      fileName = file_prefix,
      clustszind = 3,
      clustSizes = cluster_sizes,
      numOutlier = num_outliers,
      outputLogFlag = outputLogFlag,
      outputEmpirical = outputEmpirical,
      outputInfo = outputInfo
    )
    loginfo(paste("Generated:", file_prefix, "with separation", sepVal, "and", num_outliers, "outliers"))
    return(TRUE)
  }, error = function(e) {
    logerror(paste("Failed to generate:", file_prefix, "\nError:", e$message))
    return(FALSE)
  })
}

# Determine the Number of Cores to Use
available_cores <- parallel::detectCores()
needed_cores <- nrow(filtered_param_grid)
num_cores <- max(1, min(available_cores - 1, needed_cores))  # Force at least 1 core

loginfo(paste("Available cores detected:", available_cores))
loginfo(paste("Cores to be used for parallel processing:", num_cores))

# Handle Cases Where No Parameter Sets Need to be Generated
if (nrow(filtered_param_grid) == 0) {
  loginfo("No new parameter sets to generate. Exiting script.")
  quit(save="no")
}

# Set Up Parallel Backend
cl <- makeCluster(num_cores)
registerDoParallel(cl)
loginfo(paste("Parallel backend registered with", num_cores, "cores."))

# Ensure Libraries are Available on Workers
clusterEvalQ(cl, {
  library(clusterGeneration)
  library(logging)

  # Minimal Logging Setup on Each Worker
  basicConfig()
  addHandler(writeToFile, file = "cluster_generation.log", level = 'INFO', append = TRUE)
})

# Execute Cluster Generation in Parallel
results <- foreach(i = 1:nrow(filtered_param_grid)) %dopar% {
  params <- filtered_param_grid[i, ]
  generate_cluster(params)
}

# Stop the Parallel Cluster
stopCluster(cl)
loginfo("Parallel cluster stopped.")

# Summary of Results
success_count <- sum(unlist(results))
failure_count <- length(results) - success_count
loginfo(paste("Generation completed. Success:", success_count, "Failure:", failure_count))
message("All parameter sets processed. Check 'cluster_generation.log' for details.")