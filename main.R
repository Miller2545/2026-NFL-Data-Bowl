# Load packages below
packages <- c(
  "tidyverse",
  "dplyr",
  "xgboost",
  "caret",
  "ggplot2",
  "readxl"
)

# Installs packages if they aren't already downloaded
for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        message(paste("Installing package:", pkg))
        install.packages(pkg, dependencies = TRUE)
    }
}

# Loads installed packages
lapply(packages, library, character.only = TRUE)
