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

# --- 0. SETUP & PACKAGES ---
packages <- c("tidyverse", "dplyr", "xgboost", "caret", "ggplot2", "readxl", "data.table", "rsample")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
lapply(packages, library, character.only = TRUE)

# --- 1. DATA LOADING ---

nfl_input <- read.csv("~/Downloads/2026-NFL-Data-Bowl/nfl_input.csv")

# --- 2. FEATURE ENGINEERING (The Fix) ---
# We process nfl_input to create nfl_clean
nfl_clean <- nfl_input %>%
  arrange(game_id, play_id, nfl_id, frame_id) %>%
  group_by(game_id, play_id, nfl_id) %>%
  # STEP A: Create Lags (Must happen first)
  mutate(
    delta_x = x - lag(x),
    delta_y = y - lag(y),
    s_lag = lag(s),
    a_lag = lag(a),
    dir_lag = lag(dir),
    o_lag = lag(o)
  ) %>%
  # STEP B: Create Sine/Cosine using the lags from Step A
  mutate(
    dir_sin = sin(dir_lag * pi / 180),
    dir_cos = cos(dir_lag * pi / 180)
  ) %>%
  ungroup() %>%
  # STEP C: Filter NAs
  dplyr::filter(!is.na(delta_x), !is.na(s_lag))

# --- 3. SPLIT DATA ---
set.seed(42)
game_ids <- unique(nfl_clean$game_id)
train_games <- sample(game_ids, length(game_ids) * 0.8)

train_data <- nfl_clean %>% dplyr::filter(game_id %in% train_games)
test_data  <- nfl_clean %>% dplyr::filter(!game_id %in% train_games)

# --- 4. CREATE MATRICES ---
features <- c("s_lag", "a_lag", "dir_sin", "dir_cos", "o_lag")
target_x <- "delta_x"

# CRITICAL FIX: We use train_data and test_data here. 
# We do NOT use nfl_output yet because it doesn't have features engineered.
dtrain_x <- xgb.DMatrix(data = as.matrix(train_data[, features]), label = train_data[[target_x]])
dtest_x  <- xgb.DMatrix(data = as.matrix(test_data[, features]), label = test_data[[target_x]])

# --- 5. TRAIN MODEL ---
params <- list(
  objective = "reg:squarederror",
  max_depth = 6,
  eta = 0.1
)

model_delta_x <- xgboost(
  params = params,
  data = dtrain_x,
  nrounds = 100,
  verbose = 0
)

# --- 6. EVALUATE ---
# Predict changes on the TEST set
preds_delta <- predict(model_delta_x, as.matrix(test_data[, features]))

results <- test_data %>%
  select(game_id, play_id, nfl_id, frame_id, x, delta_x) %>%
  mutate(
    actual_delta_x = delta_x,
    predicted_delta_x = preds_delta,
    # Reconstruct: Previous X (calculated by removing actual delta) + Predicted Delta
    prev_x = x - actual_delta_x, 
    predicted_x = prev_x + predicted_delta_x
  )

# Calculate Error
rmse <- sqrt(mean((results$x - results$predicted_x)^2))
print(paste("RMSE:", round(rmse, 4)))


# --- TRAIN Y MODEL ---
target_y <- "delta_y"

dtrain_y <- xgb.DMatrix(data = as.matrix(train_data[, features]), label = train_data[[target_y]])
dtest_y  <- xgb.DMatrix(data = as.matrix(test_data[, features]), label = test_data[[target_y]])

model_delta_y <- xgboost(
  params = params,
  data = dtrain_y,
  nrounds = 100,
  verbose = 0
)

# Predict Y changes
preds_delta_y <- predict(model_delta_y, as.matrix(test_data[, features]))

print("Y-Model Trained!")

# Combine X and Y predictions into one results dataframe
final_results <- test_data %>%
  select(game_id, play_id, nfl_id, frame_id, x, y, delta_x, delta_y) %>%
  mutate(
    # X Calculations
    pred_dx = preds_delta,
    pred_x = (x - delta_x) + pred_dx, # Previous X + Predicted Delta
    
    # Y Calculations
    pred_dy = preds_delta_y,
    pred_y = (y - delta_y) + pred_dy  # Previous Y + Predicted Delta
  )

# --- VISUALIZATION ---
library(ggplot2)

# Pick 1 random play from the TEST set to look at
set.seed(50) 
sample_play <- final_results %>% 
  select(game_id, play_id, nfl_id) %>% 
  distinct() %>% 
  sample_n(1)

# Get the data for that specific play
plot_data <- final_results %>%
  dplyr::filter(game_id == sample_play$game_id, 
                play_id == sample_play$play_id,
                nfl_id == sample_play$nfl_id) %>%
  arrange(frame_id)

# Plot the Trajectory
ggplot(plot_data, aes(x = x, y = y)) +
  # Draw the ACTUAL path (Solid Black)
  geom_path(aes(color = "Actual"), size = 1.2, alpha = 0.6) +
  # Draw the PREDICTED path (Dashed Red)
  geom_path(aes(x = pred_x, y = pred_y, color = "Predicted"), linetype = "dashed", size = 1) +
  # Add start point
  geom_point(aes(x = x[1], y = y[1]), color = "green", size = 3) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(
    title = paste("Player Trajectory: Game", sample_play$game_id),
    subtitle = "Green Dot = Start. Black = Real, Red = Model Prediction",
    x = "X Coordinate (Long field)",
    y = "Y Coordinate (Width)"
  ) +
  theme_minimal() +
  coord_fixed() # Keeps field proportions correct
#TEST MODEL COMPLETE







# --- CONFIGURATION ---
n_models <- 5
ensemble_x <- list()
ensemble_y <- list()

# Define parameters (Adding subsample to force diversity)
params_ensemble <- list(
  objective = "reg:squarederror",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.6,        # Each tree sees only 60% of rows (Creates diversity)
  colsample_bytree = 0.8  # Each tree sees only 80% of features
)


# --- TRAIN LOOP ---
for(i in 1:n_models) {
  
  # We change the seed every time so the model looks at different random samples
  set.seed(42 + i) 
  
  # Train X Model i
  cat("Training X-Model", i, "...\n")
  model_x_i <- xgboost(
    params = params_ensemble,
    data = dtrain_x,
    nrounds = 100,
    verbose = 0
  )
  ensemble_x[[i]] <- model_x_i
  
  # Train Y Model i
  cat("Training Y-Model", i, "...\n")
  model_y_i <- xgboost(
    params = params_ensemble,
    data = dtrain_y,
    nrounds = 100,
    verbose = 0
  )
  ensemble_y[[i]] <- model_y_i
}




predict_ensemble <- function(model_list, dmatrix_input) {
  # 1. Create a matrix to store predictions from all models
  # Rows = number of data points, Cols = number of models
  n_rows <- nrow(dmatrix_input)
  n_mods <- length(model_list)
  pred_matrix <- matrix(0, nrow = n_rows, ncol = n_mods)
  
  # 2. Get predictions from each model
  for(i in 1:n_mods) {
    pred_matrix[, i] <- predict(model_list[[i]], dmatrix_input)
  }
  
  # 3. Return the average (Row Means)
  return(rowMeans(pred_matrix))
}





simulate_ensemble <- function(start_row, models_x, models_y, n_steps = 20) {
  
  sim_results <- list()
  current_state <- start_row
  
  # Initialize physics state
  curr_x <- current_state$x
  curr_y <- current_state$y
  curr_s <- current_state$s_lag
  curr_dir <- current_state$dir_lag
  
  for (i in 1:n_steps) {
    
    # A. Prepare Input
    input_features <- data.frame(
      s_lag = curr_s,
      a_lag = current_state$a_lag, # Keeping 'a' constant for now
      dir_sin = sin(curr_dir * pi / 180),
      dir_cos = cos(curr_dir * pi / 180),
      o_lag = current_state$o_lag
    )
    
    # Ensure column order matches training
    input_features <- input_features[, features] 
    d_input <- xgb.DMatrix(data = as.matrix(input_features))
    
    # B. PREDICT using the ENSEMBLE function
    pred_dx <- predict_ensemble(models_x, d_input)
    pred_dy <- predict_ensemble(models_y, d_input)
    
    # C. Update Physics
    new_x <- curr_x + pred_dx
    new_y <- curr_y + pred_dy
    
    dist <- sqrt(pred_dx^2 + pred_dy^2)
    new_s <- dist / 0.1 
    
    new_dir_rad <- atan2(pred_dx, pred_dy)
    new_dir <- (new_dir_rad * 180 / pi) %% 360
    
    # D. Store
    sim_results[[i]] <- data.frame(step = i, x = new_x, y = new_y, s = new_s)
    
    # E. Reset
    curr_x <- new_x; curr_y <- new_y; curr_s <- new_s; curr_dir <- new_dir
  }
  
  return(bind_rows(sim_results))
}