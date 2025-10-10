
import math
import os
import re
import pandas as pd
import glob
from tqdm import tqdm

def check_structure(base_dir):
    if not os.path.exists(base_dir):
        print(f"❌ BASE_DIR does not exist: {base_dir}")
        return

    print(f"✅ Found BASE_DIR: {base_dir}")
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    print(f"📂 Found {len(subfolders)} matchup folders:")
    for folder in subfolders:
        print(f"   - {folder}")

    if subfolders:
        first_matchup = os.path.join(base_dir, subfolders[0])
        configs = [f for f in os.listdir(first_matchup) if os.path.isdir(os.path.join(first_matchup, f))]
        print(f"\n📂 First matchup '{subfolders[0]}' has {len(configs)} config folders:")
        for cfg in configs:
            print(f"   - {cfg}")

        if configs:
            sample_log = os.path.join(first_matchup, configs[0], "log.csv")
            print(f"\n📝 Sample log path: {sample_log} | Exists? {os.path.exists(sample_log)}")

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

def get_action_counts(gdf, actions):
    """Return dict of {action: (count_left, count_right)}."""
    counts = {}
    for name in actions:
        left = gdf.query('Category == "Action" and Actor == 0 and Name == @name and State != 2').shape[0]
        right = gdf.query('Category == "Action" and Actor == 1 and Name == @name and State != 2').shape[0]
        counts[name] = (left, right)
    return counts


def process_log(csv_path, bot_a, bot_b, config_name):
    """Process one log.csv and return per-game metrics DataFrame."""
    df = pd.read_csv(csv_path)

    parsed = parse_config_name(config_name)
    
    # Group by game index
    game_metrics = []
    for game_id, gdf in df.groupby("GameIndex"):
        winner = gdf["GameWinner"].iloc[0]

        # Duration (action)
        duration_L = gdf[(gdf["Actor"] == 0) & (gdf["Category"] == "Action") & (gdf["State"] != 2)]["Duration"].sum()
        duration_R = gdf[(gdf["Actor"] == 1) & (gdf["Category"] == "Action") & (gdf["State"] != 2)]["Duration"].sum()

        # Count actions
        actionsL = gdf[(gdf["Category"] == "Action") & (gdf["Actor"] == 0) & (gdf["State"] != 2)]
        actionsR = gdf[(gdf["Category"] == "Action") & (gdf["Actor"] == 1) & (gdf["State"] != 2)]
        actions = ["Accelerate", "TurnLeft", "TurnRight", "Dash", "SkillBoost", "SkillStone"]
        counts = get_action_counts(gdf, actions)

        # Count collisions
        collisionsL = gdf[
            (gdf["Category"] == "Collision")
            & (gdf["Target"].notna()) & (gdf["Actor"]== 0) & (gdf["State"] != 2)
        ].shape[0]

        collisionsR = gdf[
            (gdf["Category"] == "Collision")
            & (gdf["Target"].notna()) & (gdf["Actor"]== 1) & (gdf["State"] != 2)
        ].shape[0]

        metrics = {
            "GameIndex": game_id,
            "Winner": winner,
            "Duration_L": duration_L,
            "Duration_R": duration_R,
            "ActionCounts_L": actionsL.shape[0],
            "ActionCounts_R": actionsR.shape[0],
            "TotalActions": (actionsL.shape[0] + actionsR.shape[0]),
            "Collisions_L": collisionsL,
            "Collisions_R": collisionsR,
            "Bot_L": bot_a,
            "Bot_R": bot_b,

            ## Config
            "Timer": parsed.get("Timer"),
            "ActInterval": parsed.get("ActInterval"),
            "Round": parsed.get("Round"),
            "SkillLeft": parsed.get("SkillLeft"),
            "SkillRight": parsed.get("SkillRight"),
        }

        # Add all per-action counts (L/R) automatically
        for name, (left_count, right_count) in counts.items():
            metrics[f"{name}_Act_L"] = left_count
            metrics[f"{name}_Act_R"] = right_count

        game_metrics.append(metrics)
    return pd.DataFrame(game_metrics)

def matches_filters(config, filters):
    if not filters:
        return True

    for key, allowed_values in filters.items():
        val = config.get(key)
        # Normalize both to string for consistent comparison
        val_str = str(val).lower()
        allowed_strs = [str(v).lower() for v in allowed_values]

        if val_str not in allowed_strs:
            return False
    return True


def batch(base_dir, filters = None, batch_size = 5):
    matchup_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    total_batches = math.ceil(len(matchup_folders) / batch_size)
    matchup_data = []

    for batch_idx in range(total_batches):
        batch = matchup_folders[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        print(f"\nBatch {batch_idx + 1}/{total_batches} ({len(batch)} matchups)")

        for matchup_folder in tqdm(batch, desc=f"Batch {batch_idx + 1}", unit="matchup", leave=False):
            matchup_path = os.path.join(base_dir, matchup_folder)

            match = re.match(r"(.+)_vs_(.+)", matchup_folder)
            if not match:
                continue
            bot_a, bot_b = match.groups()

            for config_folder in os.listdir(matchup_path):
                config_path = os.path.join(matchup_path, config_folder)
                if not os.path.isdir(config_path):
                    continue

                config = parse_config_name(config_folder)
                if not matches_filters(config, filters):
                    continue

                csv_files = glob.glob(os.path.join(config_path, "*.csv"))
                if not csv_files:
                    continue

                log_path = csv_files[0]
                tqdm.write(f"Processing: {bot_a} vs {bot_b} | {config_folder}")

                df_games = process_log(log_path, bot_a, bot_b, config_folder)
                matchup_data.append(df_games)

        if matchup_data:
            batch_df = pd.concat(matchup_data, ignore_index=True)
            batch_path = f"batched/batch_{batch_idx + 1:02d}.csv"
            batch_df.to_csv(batch_path, index=False)
            print(f"\nSaved {batch_path} ({len(batch_df)} rows)")
            matchup_data.clear()

def generate():
    all_games = pd.concat([pd.read_csv(f) for f in glob.glob("batched/*.csv")], ignore_index=True)

    group_cols = ["Bot_L", "Bot_R", "Timer", "ActInterval", "Round", "SkillLeft", "SkillRight"]

    # Basic aggregations
    agg_dict = {
        "GameIndex": "nunique",
        "Winner": [("L", lambda x: (x == 0).sum()), ("R", lambda x: (x == 1).sum())],
        "ActionCounts_L": "sum",
        "ActionCounts_R": "sum",
        "TotalActions": "sum",
        "Duration_L": "sum",
        "Duration_R": "sum",
        "Collisions_L": "sum",
        "Collisions_R": "sum"
    }    
    action_cols = [col for col in all_games.columns if any(col.endswith(suffix) for suffix in ("Act_L", "Act_R"))]
    agg_dict.update({col: "sum" for col in action_cols})

    matchup_summary = all_games.groupby(group_cols, as_index=False).agg(agg_dict).reset_index()
    # Flatten columns if MultiIndex
    matchup_summary.columns = [
        "_".join(filter(None, map(str, col))).strip("_") for col in matchup_summary.columns.to_flat_index()
    ]

    # Clean
    matchup_summary.rename(columns=lambda x: x.replace("_sum", "").replace("_nunique", ""), inplace=True)
    matchup_summary.rename(columns={"GameIndex": "Games"}, inplace=True)
    matchup_summary.drop(columns="index", inplace=True)

    matchup_summary["WinRate_L"] = (matchup_summary["Winner_L"]) / matchup_summary["Games"]
    matchup_summary["WinRate_R"] = (matchup_summary["Winner_R"]) / matchup_summary["Games"]
    matchup_summary.to_csv("summary_matchup.csv", index=False)

    # --- Summary per bot across all opponents ---
    bot_summary_A = (
        matchup_summary.groupby("Bot_L")
        .agg(
            TotalGames=("Games", "sum"),
            TotalWins=("Winner_L", "sum"),
            Duration=("Duration_L", "sum"),
            TotalActions=("ActionCounts_L", "sum"),
            Collisions=("Collisions_L", "sum"),
        )
        .reset_index()
        .rename(columns={"Bot_L": "Bot"})
    )
    bot_summary_A["WinRate"] = bot_summary_A["TotalWins"] / bot_summary_A["TotalGames"]

    bot_summary_B = (
        matchup_summary.groupby("Bot_R")
        .agg(
            TotalGames=("Games", "sum"),
            TotalWins=("Winner_R", "sum"),
            Duration=("Duration_R", "sum"),
            TotalActions=("ActionCounts_R", "sum"),
            Collisions=("Collisions_R", "sum")
        )
        .reset_index()
        .rename(columns={"Bot_R": "Bot"})
    )
    bot_summary_B["WinRate"] = bot_summary_B["TotalWins"] / bot_summary_B["TotalGames"]

    bot_summary = pd.concat([bot_summary_A, bot_summary_B], ignore_index=True)
    bot_summary = bot_summary.groupby("Bot").agg(
        TotalGames=("TotalGames", "sum"),
        TotalWins=("TotalWins", "sum"),
        Duration=("Duration", "sum"),
        TotalActions=("TotalActions", "sum"),
        Collisions=("Collisions", "sum"),
    ).reset_index()
    bot_summary["WinRate"] = bot_summary["TotalWins"] / bot_summary["TotalGames"]
    bot_summary.to_csv("summary_bot.csv", index=False)

    print("Done! Created summary_matchup.csv and summary_bot.csv")
    return matchup_summary, bot_summary
