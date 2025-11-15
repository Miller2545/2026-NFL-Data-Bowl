# Load packages below
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
  "tensorflow"
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

input_all <- readRDS("data/input_all_processed.rds")
release_ctx <- readRDS("data/release_context_all.rds")
release_pois <- readRDS("data/release_context_pois.rds")
final_endpos <- readRDS("data/train_final_endpos.rds")

input_joined <- input_all %>%
  left_join(
    release_ctx %>%
      select(game_id, play_id, nfl_id, release_frame = frame_id),
    by = c("game_id", "play_id", "nfl_id")
  )

# Number of frames to keep starting from last frame (release)
# If there are less than 10 frames padding with 0's
n_frames <- 10

# Features to feed into the model per frame
input_feature_cols <- c(
  "x", "y",
  "s", "a",
  "o", "dir",
  "ball_land_x", "ball_land_y",
  "absolute_yardline_number"
)

# 1) Attach release_frame to every row
input_joined <- input_all %>%
  left_join(
    release_ctx %>%
      select(game_id, play_id, nfl_id, release_frame = frame_id),
    by = c("game_id", "play_id", "nfl_id")
  )

# 2) For each (game, play, player):
#    - sort frames by time
#    - take the last up to n_frames
#    - pad with zeros on top if fewer than n_frames
padded_sequences <- input_joined %>%
  filter(!is.na(release_frame)) %>%   # just in case of bad joins
  group_by(game_id, play_id, nfl_id, release_frame) %>%
  group_modify(~ {
    df <- .x %>%
      arrange(frame_id) %>%
      select(all_of(input_feature_cols))
    
    # last up to n_frames
    if (nrow(df) > n_frames) {
      df <- df[(nrow(df) - n_frames + 1):nrow(df), , drop = FALSE]
    }
    
    real_len <- nrow(df)
    
    # pad with zeros at the TOP if fewer than n_frames
    if (real_len < n_frames) {
      pad_rows <- as_tibble(
        matrix(0, nrow = n_frames - real_len, ncol = ncol(df))
      )
      names(pad_rows) <- names(df)
      df_padded <- bind_rows(pad_rows, df)
    } else {
      df_padded <- df
    }
    
    tibble(
      real_len   = real_len,
      padded_mat = list(as.matrix(df_padded))
    )
  }) %>%
  ungroup()