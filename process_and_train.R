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

input_dir  <- "data/train"
output_dir <- "data/train"

weeks <- sprintf("%02d", 1:18)

# Building the input and output file paths for loading
input_files  <- file.path(input_dir,  paste0("input_2023_w", weeks, ".csv"))
output_files <- file.path(output_dir, paste0("output_2023_w", weeks, ".csv"))

# Loads each of the input csv files
read_input_week <- function(path) {
  week <- str_extract(path, "w[0-9]{2}") |> str_remove("^w") |> as.integer()
  
  vroom::vroom(
    path,
    col_types = vroom::cols(
      game_id                  = vroom::col_double(),
      play_id                  = vroom::col_double(),
      player_to_predict        = vroom::col_logical(),
      nfl_id                   = vroom::col_double(),
      frame_id                 = vroom::col_double(),
      play_direction           = vroom::col_character(),
      absolute_yardline_number = vroom::col_double(),
      player_name              = vroom::col_character(),
      player_height            = vroom::col_character(),
      player_weight            = vroom::col_double(),
      player_birth_date        = vroom::col_character(),
      player_position          = vroom::col_character(),
      player_side              = vroom::col_character(),
      player_role              = vroom::col_character(),
      x                        = vroom::col_double(),
      y                        = vroom::col_double(),
      s                        = vroom::col_double(),
      a                        = vroom::col_double(),
      o                        = vroom::col_double(),
      dir                      = vroom::col_double(),
      num_frames_output        = vroom::col_double(),
      ball_land_x              = vroom::col_double(),
      ball_land_y              = vroom::col_double()
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

input_all  <- purrr::map_dfr(input_files,  read_input_week)
output_all <- purrr::map_dfr(output_files, read_output_week)

# Normalize coordinates (right = positive)
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

# Release frame per player
release_ctx <- input_all |>
  group_by(game_id, play_id, nfl_id) |>
  filter(frame_id == max(frame_id)) |>
  ungroup()

# Players to predict at point of release
release_pois <- release_ctx |>
  filter(player_to_predict == TRUE)

# Final output position (last frame after throw)
output_final <- output_all |>
  group_by(game_id, play_id, nfl_id) |>
  filter(frame_id == max(frame_id)) |>
  ungroup() |>
  rename(
    x_final = x,
    y_final = y
  )

# Join input context at release with final output position
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

# Save the processed frames for offloading
saveRDS(input_all,     "data/input_all_processed.rds")
saveRDS(release_ctx,   "data/release_context_all.rds")
saveRDS(release_pois,  "data/release_context_pois.rds")
saveRDS(train_final,   "data/train_final_endpos.rds") 

# Memory cleanup
rm(input_all, output_all, release_ctx, release_pois, train_final, output_final)
gc()

# Reloading necessary data
input_all     <- readRDS("data/input_all_processed.rds")
release_ctx   <- readRDS("data/release_context_all.rds")
release_pois  <- readRDS("data/release_context_pois.rds")
final_endpos  <- readRDS("data/train_final_endpos.rds")

# Now to build the input data
# The idea behind this is to only include 10 frames prior to ball release for
# training to standardize input

# Attach release_frame to every row
input_joined <- input_all %>%
  left_join(
    release_ctx %>%
      select(game_id, play_id, nfl_id, release_frame = frame_id),
    by = c("game_id", "play_id", "nfl_id")
  )

# Number of frames to keep starting from last frame (release).
# If there are less than 10 frames, we'll pad with 0's.
n_frames <- 10

# Features to feed into the model per frame
input_feature_cols <- c(
  "x", "y",
  "s", "a",
  "o", "dir",
  "ball_land_x", "ball_land_y",
  "absolute_yardline_number"
)

# For each (game, play, player):
#  - sort frames by time
#  - take the last up to n_frames
#  - pad with zeros on top if fewer than n_frames
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

saveRDS(padded_sequences, "data/padded.rds")

# Reload
padded_sequences <- readRDS("data/padded.rds")

output_all <- purrr::map_dfr(output_files, read_output_week)

# Join POIs with their padded 10-frame sequences
pois_with_seq <- release_pois %>%
  select(game_id, play_id, nfl_id, num_frames_output) %>%
  inner_join(
    padded_sequences,
    by = c("game_id", "play_id", "nfl_id")
  )

# Filter output to only POI players
output_pois <- output_all %>%
  inner_join(
    release_pois %>%
      select(game_id, play_id, nfl_id, num_frames_output),
    by = c("game_id", "play_id", "nfl_id")
  )

# Build training rows = one row per (game, play, player, future frame)
train_frames <- output_pois %>%
  inner_join(
    pois_with_seq %>%
      select(game_id, play_id, nfl_id, num_frames_output, padded_mat),
    by = c("game_id", "play_id", "nfl_id")
  )

# Normalized time index for each future frame
train_frames <- train_frames %>%
  mutate(
    t_input = frame_id
  )

# Model building time, yippee

# X: sequence input from padded_mat (list of 10x9 matrices)
X_list <- train_frames$padded_mat

# Sanity checks
stopifnot(length(unique(purrr::map_int(X_list, nrow))) == 1)  # all 10
stopifnot(length(unique(purrr::map_int(X_list, ncol))) == 1)  # all 9

# Combine into [time, features, sample]
X_3d <- abind::abind(X_list, along = 3)

# Reorder to [sample, time, features] for Keras
X_array <- aperm(X_3d, c(3, 1, 2))
dim(X_array)
# Expect: N_samples x 10 x 9

# Time input: [N, 1]
time_input <- matrix(train_frames$t_input, ncol = 1)
dim(time_input)
# N_samples x 1

# Targets: [N, 2] (x_t, y_t at that frame)
Y_array <- as.matrix(train_frames[, c("x", "y")])
dim(Y_array)
# N_samples x 2

n_in_steps   <- dim(X_array)[2]  # 10
n_features   <- dim(X_array)[3]  # 9
hidden_units <- 64

# Sequence input (10 x 9)
seq_input <- layer_input(shape = c(n_in_steps, n_features), name = "seq_input")

# Time input (scalar: t_rel)
time_input_layer <- layer_input(shape = c(1), name = "time_input")

# Encode sequence with LSTM
seq_encoded <- seq_input %>%
  layer_masking(mask_value = 0) %>%
  layer_lstm(units = hidden_units, return_sequences = FALSE)

# Simple MLP on time input
time_encoded <- time_input_layer %>%
  layer_dense(units = 8, activation = "relu")

# Concatenate sequence representation and time
merged <- layer_concatenate(list(seq_encoded, time_encoded))

# Predict x, y at that future frame
output <- merged %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 2, name = "xy_output")

model <- keras_model(
  inputs  = list(seq_input, time_input_layer),
  outputs = output
)

model %>% compile(
  optimizer = "adam",
  loss      = "mse"
)

summary(model)

# Model training

history <- model %>% fit(
  x = list(
    seq_input  = X_array,
    time_input = time_input
  ),
  y = Y_array,
  batch_size = 64,
  epochs = 10,
  validation_split = 0.1,
  shuffle = TRUE
)

# Saving the model
save_model_hdf5(model, "models/lstm_frame_conditioned.h5")
