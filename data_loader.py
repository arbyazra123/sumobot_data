"""
Data loading functions for sumobot analyzer
"""
import polars as pl
import os
import glob
from tqdm import tqdm
from analyzer_utils import collect_with_gpu, extract_timer_from_config


def load_data_chunked(csv_path, chunksize=50000, actor_filter=None):
    """
    Load CSV data using Polars with GPU acceleration

    Args:
        csv_path: Path to CSV file
        chunksize: Number of rows per chunk (ignored for Polars, kept for API compatibility)
        actor_filter: Filter for specific actor (0 for left, 1 for right, None for both)
    """
    # Define explicit schema to avoid inference issues with GPU
    schema = {
        "GameIndex": pl.Int64,
        "Actor": pl.Int64,
        "UpdatedAt": pl.Float64,
        "BotPosX": pl.Float64,
        "BotPosY": pl.Float64,
        "BotRot": pl.Float64,
    }

    # Use Polars lazy API with explicit schema
    lf = pl.scan_csv(csv_path, schema=schema)

    # Filter by actor if specified
    if actor_filter is not None:
        # Use numeric comparison to avoid schema issues
        lf = lf.filter(pl.col("Actor") == actor_filter)

    # Drop invalid entries
    lf = lf.drop_nulls(subset=["BotPosX", "BotPosY", "BotRot"])

    # Collect with GPU acceleration
    df = collect_with_gpu(lf)

    return df


def load_bot_data_from_simulation(base_dir, bot_name, actor_position="left", chunksize=50000, max_configs=None, group_by_timer=False, also_load_distance=False):
    """
    Load all CSV data for a specific bot from the simulation directory

    Args:
        base_dir: Base simulation directory
        bot_name: Name of the bot (e.g., "Bot_BT", "Bot_NN", "Bot_Primitive")
        actor_position: "left" (Actor 0) or "right" (Actor 1) or "both"
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of config folders to process (None for all)
        group_by_timer: If True, return dict of {timer_value: DataFrame}, else return combined DataFrame
        also_load_distance: If True, also return timer-grouped distance data

    Returns:
        Combined DataFrame with all bot data, or dict of DataFrames grouped by Timer
        If also_load_distance=True, returns tuple: (bot_data, distance_data)
    """
    from distance_calculations import calculate_distance_between_bots

    all_data = []
    timer_grouped_data = {}  # {timer_value: [dataframes]}
    timer_distance_data = {}  # {timer_value: [distance dataframes]}

    # Find all matchup folders containing this bot
    matchup_folders = [f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f)) and bot_name in f]

    print(f"Found {len(matchup_folders)} matchup folders for {bot_name}")

    total_csvs = 0
    for matchup_folder in matchup_folders:
        matchup_path = os.path.join(base_dir, matchup_folder)

        # Determine actor filter based on bot position in matchup
        # Bot_A_vs_Bot_B: Bot_A is actor 0 (left), Bot_B is actor 1 (right)
        parts = matchup_folder.split("_vs_")
        if len(parts) != 2:
            continue

        # Determine which actor this bot is in this matchup
        if actor_position == "both":
            actor_filter = None
        elif actor_position == "left":
            actor_filter = 0 if bot_name == parts[0] else None
        else:  # right
            actor_filter = 1 if bot_name == parts[1] else None

        # Skip if bot is not in the desired position for this matchup
        if actor_position != "both" and actor_filter is None:
            continue

        # Get all config folders
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        if max_configs and len(config_folders) > max_configs:
            config_folders = config_folders[:max_configs]

        print(f"  {matchup_folder}: {len(config_folders)} configs")

        # Process each config folder
        for config_folder in tqdm(config_folders, desc=f"  Loading {matchup_folder}", leave=False):
            config_path = os.path.join(matchup_path, config_folder)

            # Find CSV file in this config folder
            csv_files = glob.glob(os.path.join(config_path, "*.csv"))

            if csv_files:
                csv_path = csv_files[0]  # Should only be 1 CSV per config
                df = load_data_chunked(csv_path, chunksize, actor_filter=actor_filter)

                if not df.is_empty():
                    # Also load distance data if requested
                    if also_load_distance:
                        df_all_actors = load_data_chunked(csv_path, chunksize, actor_filter=None)
                        if not df_all_actors.is_empty():
                            dist_df = calculate_distance_between_bots(df_all_actors)
                            if not dist_df.is_empty():
                                timer = extract_timer_from_config(config_folder)
                                if timer is not None:
                                    if timer not in timer_distance_data:
                                        timer_distance_data[timer] = []
                                    timer_distance_data[timer].append(dist_df)

                    if group_by_timer:
                        # Extract timer value and group
                        timer = extract_timer_from_config(config_folder)
                        if timer is not None:
                            if timer not in timer_grouped_data:
                                timer_grouped_data[timer] = []
                            timer_grouped_data[timer].append(df)
                    else:
                        all_data.append(df)
                    total_csvs += 1

    if group_by_timer:
        # Return dict of combined DataFrames per timer
        if not timer_grouped_data:
            print("No valid data found.")
            if also_load_distance:
                return {}, {}
            return {}

        print(f"\nLoaded {total_csvs} CSV files")
        result = {}
        for timer, dfs in timer_grouped_data.items():
            print(f"Combining data for Timer={timer}...")
            result[timer] = pl.concat(dfs)
            print(f"  Timer {timer}: {len(result[timer]):,} samples")

        if also_load_distance:
            return result, timer_distance_data
        return result
    else:
        # Return combined DataFrame
        if not all_data:
            print("No valid data found.")
            if also_load_distance:
                return pl.DataFrame(), {}
            return pl.DataFrame()

        print(f"\nLoaded {total_csvs} CSV files")
        print("Combining all data...")
        df_combined = pl.concat(all_data)

        print(f"Total samples: {len(df_combined):,}")

        if also_load_distance:
            return df_combined, timer_distance_data
        return df_combined
