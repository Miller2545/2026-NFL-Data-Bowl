# Load packages below
packages <- c(
  "dplyr",
  "xgboost",
  "caret",
  "ggplot2",
  "vroom",
  "stringr",
  "lubridate",
  "purrr"
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

input_dir  <- "data/train"
output_dir <- "data/train"

weeks <- sprintf("%02d", 1:18)

input_files  <- file.path(input_dir,  paste0("input_2023_w", weeks, ".csv"))
output_files <- file.path(output_dir, paste0("output_2023_w", weeks, ".csv"))

read_input_week <- function(path) {
  week <- str_extract(path, "w[0-9]{2}") |> str_remove("^w") |> as.integer()
  
  vroom::vroom(
    path,
    col_types = vroom::cols(
      game_id                 = vroom::col_double(),
      play_id                 = vroom::col_double(),
      player_to_predict       = vroom::col_logical(),
      nfl_id                  = vroom::col_double(),
      frame_id                = vroom::col_double(),
      play_direction          = vroom::col_character(),
      absolute_yardline_number= vroom::col_double(),
      player_name             = vroom::col_character(),
      player_height           = vroom::col_character(),
      player_weight           = vroom::col_double(),
      player_birth_date       = vroom::col_character(),
      player_position         = vroom::col_character(),
      player_side             = vroom::col_character(),
      player_role             = vroom::col_character(),
      x                       = vroom::col_double(),
      y                       = vroom::col_double(),
      s                       = vroom::col_double(),
      a                       = vroom::col_double(),
      o                       = vroom::col_double(),
      dir                     = vroom::col_double(),
      num_frames_output       = vroom::col_double(),
      ball_land_x             = vroom::col_double(),
      ball_land_y             = vroom::col_double()
    )
  ) |>
    mutate(week = week)
}

input_all <- map_dfr(input_files, read_input_week)



