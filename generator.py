
import gc
import math
import os
import re
import subprocess
import sys
import pandas as pd
import glob
from tqdm import tqdm
from functools import lru_cache
import numpy as np


def check_game_jsons(base_dir):
    if not os.path.exists(base_dir):
        print(f"❌ BASE_DIR does not exist: {base_dir}")
        return

    print(f"✅ Found BASE_DIR: {base_dir}")
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    print(f"📂 Found {len(subfolders)} matchup folders:")

    for matchup in subfolders:
        matchup_path = os.path.join(base_dir, matchup)
        configs = [f for f in os.listdir(matchup_path) if os.path.isdir(os.path.join(matchup_path, f))]

        all_ok = True
        for cfg in configs:
            cfg_path = os.path.join(matchup_path, cfg)
            count = sum(
                os.path.exists(os.path.join(cfg_path, f"game_{i:03}.json"))
                for i in range(50)
            )
            if count != 50:
                all_ok = False
                break

        if all_ok:
            print(f"✅ {matchup}: has all 50 game JSONs in each config")
        else:
            print(f"❌ {matchup}: missing one or more game JSONs")

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



def process_log(csv_path, bot_a, bot_b, config_name, chunksize, time_bin_size=5):
    parsed = parse_config_name_cached(config_name)
    game_metrics = []
    time_fragment_actions = []

    # Read CSV in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        print(csv_path)
        # Precompute reusable masks per chunk
        is_action = (chunk["Category"] == "Action") & (chunk["State"] != 2)
        is_collision = (chunk["Category"] == "Collision") & (chunk["Target"].notna()) & (chunk["State"] != 2)

        # Iterate by group
        for game_id, gdf in chunk.groupby("GameIndex"):
            winner = gdf["GameWinner"].iloc[0]

            L = gdf["Actor"] == 0
            R = gdf["Actor"] == 1

            duration_L = gdf.loc[L & is_action, "Duration"].sum()
            duration_R = gdf.loc[R & is_action, "Duration"].sum()
            
            match_dur = gdf.loc[is_action, "UpdatedAt"].iloc[-1]

            actionsL = gdf.loc[L & is_action]
            actionsR = gdf.loc[R & is_action]

            counts = get_action_counts(gdf, ["Accelerate", "TurnLeft", "TurnRight", "Dash", "SkillBoost", "SkillStone"])

            collisionsL = (is_collision & L).sum()
            collisionsR = (is_collision & R).sum()

            metrics = {
                "GameIndex": game_id,
                "Winner": winner,
                "Duration_L": duration_L,
                "Duration_R": duration_R,
                "ActionCounts_L": len(actionsL),
                "ActionCounts_R": len(actionsR),
                "TotalActions": len(actionsL) + len(actionsR),
                "Collisions_L": collisionsL,
                "Collisions_R": collisionsR,
                "Bot_L": bot_a,
                "Bot_R": bot_b,
                "Timer": parsed.get("Timer"),
                "ActInterval": parsed.get("ActInterval"),
                "Round": parsed.get("Round"),
                "SkillLeft": parsed.get("SkillLeft"),
                "SkillRight": parsed.get("SkillRight"),
                "MatchDur":match_dur,
            }

            for name, (left_count, right_count) in counts.items():
                metrics[f"{name}_Act_L"] = left_count
                metrics[f"{name}_Act_R"] = right_count

            game_metrics.append(metrics)

            bins = np.arange(0, match_dur + time_bin_size, time_bin_size)

            for side, bot in [(0, bot_a), (1, bot_b)]:
                gdf_side = gdf[(gdf["Actor"] == side) & is_action].copy()
                if gdf_side.empty:
                    continue

                # Assign time bins
                gdf_side["TimeBin"] = pd.cut(gdf_side["UpdatedAt"], bins=bins, labels=bins[:-1], include_lowest=True)

                # Count per time bin and per action
                grouped = gdf_side.groupby(["TimeBin", "Name"]).size().reset_index(name="Count")
                for _, row in grouped.iterrows():
                    time_fragment_actions.append({
                        "GameIndex": game_id,
                        "Bot": bot,
                        "Timer": parsed.get("Timer"),
                        "ActInterval": parsed.get("ActInterval"),
                        "Round": parsed.get("Round"),
                        "SkillLeft": parsed.get("SkillLeft"),
                        "SkillRight": parsed.get("SkillRight"),
                        "TimeBin": row["TimeBin"],
                        "Action": row["Name"],
                        "Count": row["Count"]
                    })

    return pd.DataFrame(game_metrics), pd.DataFrame(time_fragment_actions)


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


def batch(base_dir, filters=None, batch_size=5, checkpoint_dir="batched", chunksize=50_000, time_bin_size=2):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    matchup_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    total_batches = math.ceil(len(matchup_folders) / batch_size)
    
    matchup_data = []
    time_fragment_data = []  # NEW

    # Determine which batches have already been processed
    processed_batches_metrics = set()
    processed_batches_timefrag = set()
    for f in os.listdir(checkpoint_dir):
        match = re.match(r"batch_(\d+)\.csv", f)
        if match:
            processed_batches_metrics.add(int(match.group(1)))
        match_tf = re.match(r"batch_timefrag_(\d+)\.csv", f)
        if match_tf:
            processed_batches_timefrag.add(int(match_tf.group(1)))

    for batch_idx in range(total_batches):
        # Skip batches already processed
        if (batch_idx + 1) in processed_batches_metrics and (batch_idx + 1) in processed_batches_timefrag:
            print(f"Skipping batch {batch_idx + 1} (already processed)")
            continue

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

                with open("processing.log", "a") as f:
                    f.write(f"{bot_a} vs {bot_b} | {config_folder}\n")

                result = subprocess.run(
                        [sys.executable, "process_one.py", log_path, bot_a, bot_b, config_folder, str(chunksize)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                if result.returncode != 0:
                    print(f"[FAIL] {log_path} (code {result.returncode})")
                    print(result.stderr)
                    continue

                processed_path = log_path.replace(".csv", ".processed.csv")
                if os.path.exists(processed_path):
                    df = pd.read_csv(processed_path)
                    matchup_data.append(df)
                    os.remove(processed_path)

                processed_path_time = log_path.replace(".csv", ".time.processed.csv")
                if os.path.exists(processed_path_time):
                    df = pd.read_csv(processed_path_time)
                    time_fragment_data.append(df)
                    os.remove(processed_path_time)

        # Save batch checkpoint
        if matchup_data:
            batch_df = pd.concat(matchup_data, ignore_index=True)
            batch_path = os.path.join(checkpoint_dir, f"batch_{batch_idx + 1:02d}.csv")
            batch_df.to_csv(batch_path, index=False)
            matchup_data.clear()
            del batch_df
            gc.collect()
            print(f"Saved batch metrics: {batch_path}")

        # Save time-fragment batch checkpoint
        if time_fragment_data:
            batch_tf_df = pd.concat(time_fragment_data, ignore_index=True)
            batch_tf_path = os.path.join(checkpoint_dir, f"batch_timefrag_{batch_idx + 1:02d}.csv")
            batch_tf_df.to_csv(batch_tf_path, index=False)
            time_fragment_data.clear()
            del batch_tf_df
            gc.collect()
            print(f"Saved batch time-fragments: {batch_tf_path}")


def generate():
    all_games = pd.concat(
        [pd.read_csv(f) for f in glob.glob("batched/batch_*.csv")],
        ignore_index=True
    )
    all_timebins = pd.concat(
        [pd.read_csv(f) for f in glob.glob("batched/batch_timefrag_*.csv")],
        ignore_index=True
    ) 

    # --- Summary per matchup configuration ---
    matchup_summary = create_summary_matchup(all_games)

    # --- Summary per bot across all opponents ---
    bot_summary = create_summary(matchup_summary)

     # --- Summary per bot & config over timebins ---
    summary_timebins = summarize_timebins(all_timebins)

    print("Done! Created summary_matchup.csv and summary_bot.csv")
    return matchup_summary, bot_summary, summary_timebins

def create_summary(matchup_summary):
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
            Collisions=("Collisions_R", "sum"),
        )
        .reset_index()
        .rename(columns={"Bot_R": "Bot"})
    )
    bot_summary_B["WinRate"] = bot_summary_B["TotalWins"] / bot_summary_B["TotalGames"]

    # Combine A and B
    bot_summary = pd.concat([bot_summary_A, bot_summary_B], ignore_index=True)

    # Aggregate totals per bot
    bot_summary = (
        bot_summary.groupby("Bot")
        .agg(
            TotalGames=("TotalGames", "sum"),
            TotalWins=("TotalWins", "sum"),
            Duration=("Duration", "sum"),
            TotalActions=("TotalActions", "sum"),
            Collisions=("Collisions", "sum"),
        )
        .reset_index()
    )

    # Calculate WinRate
    bot_summary["WinRate"] = bot_summary["TotalWins"] / bot_summary["TotalGames"]

    # Rank by WinRate (1 = highest winrate)
    bot_summary["Rank"] = bot_summary["WinRate"].rank(ascending=False, method="dense").astype(int)

    # Optional: sort by rank for output
    bot_summary = bot_summary.sort_values(by="Rank", ascending=True).reset_index(drop=True)

    # Save to CSV
    bot_summary.to_csv("summary_bot.csv", index=False)
    
    return bot_summary


def create_summary_matchup(all_games: pd.DataFrame):
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
        "Collisions_R": "sum",
        "MatchDur": "mean",
    }

    # Include all *_Act_L/_Act_R columns
    action_cols = [col for col in all_games.columns if any(col.endswith(suffix) for suffix in ("Act_L", "Act_R"))]
    agg_dict.update({col: "sum" for col in action_cols})

    # Aggregate
    matchup_summary = all_games.groupby(group_cols, as_index=False).agg(agg_dict).reset_index()

    # Flatten MultiIndex columns
    matchup_summary.columns = [
        "_".join(filter(None, map(str, col))).strip("_")
        for col in matchup_summary.columns.to_flat_index()
    ]

    # Clean up column names
    matchup_summary.rename(
        columns=lambda x: x.replace("_sum", "").replace("_nunique", "").replace("_mean", ""),
        inplace=True,
    )

    matchup_summary.rename(columns={"GameIndex": "Games"}, inplace=True)
    matchup_summary.drop(columns="index", inplace=True)

    # Compute win rates
    matchup_summary["WinRate_L"] = matchup_summary["Winner_L"] / matchup_summary["Games"]
    matchup_summary["WinRate_R"] = matchup_summary["Winner_R"] / matchup_summary["Games"]

    # Compute rank based only on Bot_L and its WinRate_L
    bot_ranks_L = (
        matchup_summary.groupby("Bot_L", as_index=False)["WinRate_L"]
        .mean()
        .sort_values("WinRate_L", ascending=False)
        .reset_index(drop=True)
    )

    bot_ranks_R = (
        matchup_summary.groupby("Bot_R", as_index=False)["WinRate_R"]
        .mean()
        .sort_values("WinRate_R", ascending=False)
        .reset_index(drop=True)
    )

    # Assign dense rank (1 = best)
    bot_ranks_L["Rank_L"] = bot_ranks_L["WinRate_L"].rank(ascending=False, method="dense").astype(int)
    bot_ranks_R["Rank_R"] = bot_ranks_R["WinRate_R"].rank(ascending=False, method="dense").astype(int)

    # Merge back to the main summary
    matchup_summary = matchup_summary.merge(bot_ranks_L[["Bot_L", "Rank_L"]], on="Bot_L", how="left")
    matchup_summary = matchup_summary.merge(bot_ranks_R[["Bot_R", "Rank_R"]], on="Bot_R", how="left")

    matchup_summary = matchup_summary.sort_values(by="Rank_L", ascending=True).reset_index(drop=True)

    # Save to CSV
    matchup_summary.to_csv("summary_matchup.csv", index=False)

    return matchup_summary

def summarize_timebins(df_time: pd.DataFrame):
    """
    Summarize time fragment (timebin) actions across all games.
    
    Parameters:
        df_time: pd.DataFrame
            Must contain at least:
            ['Bot', 'Timer', 'ActInterval', 'Round', 'TimeBin', 'Action', 'Count']
        output_path: str
            Where to save the summarized CSV.
    """
    # Validate columns
    required = {'Bot', 'Timer', 'ActInterval', 'Round', 'TimeBin', 'Action', 'Count'}
    if not required.issubset(df_time.columns):
        missing = required - set(df_time.columns)
        raise ValueError(f"Missing columns in df_time: {missing}")

    # Group and compute mean
    summary = (
        df_time
        .groupby(['Bot', 'Timer', 'ActInterval', 'Round', 'TimeBin', 'Action'], as_index=False)
        .agg({'Count': 'mean'})
        .rename(columns={'Count': 'MeanCount'})
    )

    # Sort for readability
    summary = summary.sort_values(['Bot', 'Timer', 'ActInterval', 'Round', 'TimeBin', 'Action'])

    # Save CSV
    summary.to_csv("summary_timebins.csv", index=False)
    print(f"✅ Saved summary to {output_path}")

    return summary
