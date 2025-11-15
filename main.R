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

# Building the input and output file pathings for loading
input_files  <- file.path(input_dir,  paste0("input_2023_w", weeks, ".csv"))
output_files <- file.path(output_dir, paste0("output_2023_w", weeks, ".csv"))

# Loads each of the input csv files
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

# Loads each of the output files
read_output_week <- function(path) {
  week <- str_extract(path, "w[0-9]{2}") |> str_remove("^w") |> as.integer()
  
  vroom::vroom(
    path,
    col_types = vroom::cols(
      game_id   = vroom::col_double(),
      play_id   = vroom::col_double(),
      nfl_id    = vroom::col_double(),
      frame_id  = vroom::col_double(),
      x         = vroom::col_double(),
      y         = vroom::col_double()
    )
  ) |>
    mutate(week = week)
}

input_all <- map_dfr(input_files, read_input_week)

output_all <- map_dfr(output_files, read_output_week)

# Normalize the directions of the x and y coordinates, right is positive and left is negative movement
normalize_xy <- function(df) {
  df |>
    mutate(
      x = if_else(play_direction == "right", x, 120 - x),
      y = if_else(play_direction == "right", y, 53.3 - y),
      ball_land_x = if_else(play_direction == "right", ball_land_x, 120 - ball_land_x),
      ball_land_y = if_else(play_direction == "right", ball_land_y, 53.3 - ball_land_y)
    )
}

input_all <- normalize_xy(input_all)

# Release frame context
release_ctx <- input_all |>
  group_by(game_id, play_id, nfl_id) |>
  filter(frame_id == max(frame_id)) |>
  ungroup()

# Players to predict at point of release
release_pois <- release_ctx |>
  filter(player_to_predict == TRUE)

output_final <- output_all |>
  group_by(game_id, play_id, nfl_id) |>
  filter(frame_id == max(frame_id)) |>
  ungroup() |>
  rename(
    x_final = x,
    y_final = y
  )

# Joining input and output contexts
train_final <- release_pois |>
  select(
    week, game_id, play_id, nfl_id,
    x, y, s, a, o, dir,
    absolute_yardline_number,
    ball_land_x, ball_land_y,
    play_direction, player_position, player_role, player_side,
    num_frames_output
  ) |>
  inner_join(output_final, by = c("week", "game_id", "play_id", "nfl_id"))

# Save as RDS for better efficiency
saveRDS(input_all,     "data/input_all_processed.rds")
saveRDS(release_ctx,   "data/release_context_all.rds")
saveRDS(release_pois,  "data/release_context_pois.rds")
saveRDS(train_final,   "data/train_final_endpos.rds") 

rm(input_all, output_all, release_ctx, release_pois, train_final, output_final)
gc()

