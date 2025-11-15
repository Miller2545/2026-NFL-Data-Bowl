packages <- c(
  "dplyr",
  "xgboost",
  "caret",
  "ggplot2",
  "vroom",
  "stringr",
  "lubridate",
  "purrr",
  "keras",
  "tensorflow",
  "abind"
)

# Install packages if they aren't already downloaded
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Installing package:", pkg))
    install.packages(pkg, dependencies = TRUE)
  }
}

# Load installed packages
lapply(packages, library, character.only = TRUE)


