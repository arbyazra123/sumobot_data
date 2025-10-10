from functools import lru_cache
import glob
import pyarrow as csv
import pyarrow.parquet as pq
import pandas as pd
import os
import re
import math
import gc
from tqdm import tqdm

def process_log_parquet(csv_path, bot_a, bot_b, config_name, chunksize=100000):
    """Optimized version using Parquet for intermediate processing"""
    parsed = parse_config_name_cached(config_name)
    
    # Convert CSV to Parquet first for faster processing
    parquet_path = csv_path.replace('.csv', '.parquet')
    
    if not os.path.exists(parquet_path):
        # Read CSV with pandas and convert to Parquet
        # Read in chunks to handle large files
        first_chunk = True
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if first_chunk:
                chunk.to_parquet(parquet_path, index=False)
                first_chunk = False
            else:
                chunk.to_parquet(parquet_path, index=False, append=True)
    
    return process_log_parquet_existing(parquet_path, bot_a, bot_b, config_name)

def process_log_parquet_existing(parquet_path, bot_a, bot_b, config_name):
    """Process already converted Parquet file"""
    parsed = parse_config_name_cached(config_name)
    
    # Read only needed columns
    columns = ['GameIndex', 'Category', 'State', 'Target', 'Actor', 'Duration', 
               'GameWinner', 'Name']
    
    # Read the entire parquet file (it's more efficient than CSV chunks)
    df = pd.read_parquet(parquet_path, columns=columns)
    
    # Precompute masks for entire dataframe
    is_action = (df["Category"] == "Action") & (df["State"] != 2)
    is_collision = (df["Category"] == "Collision") & (df["Target"].notna()) & (df["State"] != 2)
    
    game_metrics = []
    
    # Use more efficient groupby operations
    for game_id, gdf in df.groupby("GameIndex"):
        winner = gdf["GameWinner"].iloc[0]
        
        L = gdf["Actor"] == 0
        R = gdf["Actor"] == 1
        
        # Vectorized operations
        duration_L = gdf.loc[L & is_action, "Duration"].sum()
        duration_R = gdf.loc[R & is_action, "Duration"].sum()
        
        actionsL_count = (L & is_action).sum()
        actionsR_count = (R & is_action).sum()
        
        collisionsL = (is_collision & L).sum()
        collisionsR = (is_collision & R).sum()
        
        # Optimized action counts
        action_counts = get_action_counts_optimized(gdf, L, R, is_action)
        
        metrics = {
            "GameIndex": game_id,
            "Winner": winner,
            "Duration_L": duration_L,
            "Duration_R": duration_R,
            "ActionCounts_L": actionsL_count,
            "ActionCounts_R": actionsR_count,
            "TotalActions": actionsL_count + actionsR_count,
            "Collisions_L": collisionsL,
            "Collisions_R": collisionsR,
            "Bot_L": bot_a,
            "Bot_R": bot_b,
            "Timer": parsed.get("Timer"),
            "ActInterval": parsed.get("ActInterval"),
            "Round": parsed.get("Round"),
            "SkillLeft": parsed.get("SkillLeft"),
            "SkillRight": parsed.get("SkillRight"),
            **action_counts
        }
        
        game_metrics.append(metrics)
    
    return pd.DataFrame(game_metrics)

def get_action_counts_optimized(gdf, L_mask, R_mask, is_action_mask):
    """Optimized version of get_action_counts"""
    actions = ["Accelerate", "TurnLeft", "TurnRight", "Dash", "SkillBoost", "SkillStone"]
    counts = {}
    
    for action in actions:
        action_mask = gdf["Name"] == action
        left_count = (L_mask & is_action_mask & action_mask).sum()
        right_count = (R_mask & is_action_mask & action_mask).sum()
        counts[f"{action}_Act_L"] = left_count
        counts[f"{action}_Act_R"] = right_count
    
    return counts

def batch_parquet(base_dir, filters=None, batch_size=5, checkpoint_dir="batched"):
    """Optimized batch processing with Parquet support"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    matchup_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    total_batches = math.ceil(len(matchup_folders) / batch_size)
    
    # Check for existing batches (both CSV and Parquet)
    processed_batches = set()
    for f in os.listdir(checkpoint_dir):
        match = re.match(r"batch_(\d+)\.(csv|parquet)", f)
        if match:
            processed_batches.add(int(match.group(1)))
    
    # Pre-compile regex for better performance
    matchup_pattern = re.compile(r"(.+)_vs_(.+)")
    
    for batch_idx in range(total_batches):
        if (batch_idx + 1) in processed_batches:
            print(f"Skipping batch {batch_idx + 1} (already processed)")
            continue

        batch_folders = matchup_folders[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        print(f"\nBatch {batch_idx + 1}/{total_batches} ({len(batch_folders)} matchups)")
        
        batch_data = []
        
        for matchup_folder in tqdm(batch_folders, desc=f"Batch {batch_idx + 1}", unit="matchup", leave=False):
            matchup_path = os.path.join(base_dir, matchup_folder)
            
            match = matchup_pattern.match(matchup_folder)
            if not match:
                continue
            bot_a, bot_b = match.groups()
            
            # Process all configs in this matchup
            config_results = process_matchup_configs(matchup_path, bot_a, bot_b, filters)
            batch_data.extend(config_results)
        
        # Save batch checkpoint
        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            
            # Save as Parquet for better performance
            batch_path_parquet = os.path.join(checkpoint_dir, f"batch_{batch_idx + 1:02d}.parquet")
            batch_df.to_parquet(batch_path_parquet, index=False)
            
            print(f"\nSaved {batch_path_parquet} ({len(batch_df)} rows)")
            batch_data.clear()
            del batch_df
            gc.collect()

def process_matchup_configs(matchup_path, bot_a, bot_b, filters):
    """Process all config folders for a matchup"""
    config_results = []
    
    for config_folder in os.listdir(matchup_path):
        config_path = os.path.join(matchup_path, config_folder)
        if not os.path.isdir(config_path):
            continue
        
        config = parse_config_name(config_folder)
        if not matches_filters(config, filters):
            continue
        
        # Look for both CSV and Parquet files
        csv_files = glob.glob(os.path.join(config_path, "*.csv"))
        parquet_files = glob.glob(os.path.join(config_path, "*.parquet"))
        
        if not csv_files and not parquet_files:
            continue
        
        # Prefer Parquet if available, otherwise use CSV
        if parquet_files:
            log_path = parquet_files[0]
            process_func = process_log_parquet_existing
        else:
            log_path = csv_files[0]
            process_func = process_log_parquet
        
        tqdm.write(f"Processing: {bot_a} vs {bot_b} | {config_folder}")
        
        try:
            df_games = process_func(log_path, bot_a, bot_b, config_folder)
            config_results.append(df_games)
        except Exception as e:
            print(f"Error processing {log_path}: {e}")
            # Continue with other files instead of exiting
            continue
    
    return config_results

def convert_csv_to_parquet_batch(base_dir):
    """Pre-convert all CSVs to Parquet format for maximum performance"""
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                csv_files.append(csv_path)
    
    print(f"Found {len(csv_files)} CSV files to convert...")
    
    for csv_path in tqdm(csv_files, desc="Converting CSVs to Parquet"):
        parquet_path = csv_path.replace('.csv', '.parquet')
        if not os.path.exists(parquet_path):
            try:
                # Read and convert to parquet
                df = pd.read_csv(csv_path)
                df.to_parquet(parquet_path, index=False)
            except Exception as e:
                print(f"Error converting {csv_path}: {e}")

# Keep the original matches_filters function as it's already efficient
def matches_filters(config, filters):
    if not filters:
        return True

    for key, allowed_values in filters.items():
        val = config.get(key)
        val_str = str(val).lower()
        allowed_strs = [str(v).lower() for v in allowed_values]

        if val_str not in allowed_strs:
            return False
    return True

@lru_cache(maxsize=None)
def parse_config_name_cached(name):
    return parse_config_name(name)

def parse_config_name(config_name: str):
    """
    Extract structured info from config folder name like:
    Timer_45__ActInterval_0.1__Round_BestOf1__SkillLeft_Boost__SkillRight_Stone
    """
    # Split by double underscores "__"
    segments = config_name.split("__")
    config = {}

    for seg in segments:
        # Split only on the first underscore "_"
        if "_" in seg:
            key, value = seg.split("_", 1)
            config[key] = value
        else:
            # Fallback if malformed
            config[seg] = True

    # Optional: convert numeric-looking values
    for k, v in config.items():
        # Try float conversion for numeric strings
        if isinstance(v, str) and re.match(r"^-?\d+(\.\d+)?$", v):
            config[k] = float(v)
    return config

def get_action_counts(df, actions):
    result = {}
    for act in actions:
        left = ((df["Actor"] == 0) & (df["Name"] == act) & (df["State"] != 2)).sum()
        right = ((df["Actor"] == 1) & (df["Name"] == act) & (df["State"] != 2)).sum()
        result[act] = (left, right)
    return result