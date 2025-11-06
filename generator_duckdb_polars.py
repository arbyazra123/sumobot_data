"""
DuckDB + Polars: Maximum performance version
Uses DuckDB for reading and Polars for fast DataFrame operations
"""

import duckdb
import os
import re
from tqdm import tqdm
from functools import lru_cache

try:
    import polars as pl
    print("✅ Using Polars")
except ImportError:
    print("❌ Polars not installed. Run: pip install polars")
    exit(1)

import numpy as np
import pandas as pd  # Need pandas for pd.cut in time bins


@lru_cache(maxsize=None)
def parse_config_name_cached(name):
    return parse_config_name(name)


def parse_config_name(config_name: str):
    """Extract structured info from config folder name"""
    segments = config_name.split("__")
    config = {}

    for seg in segments:
        if "_" in seg:
            key, value = seg.split("_", 1)
            config[key] = value
        else:
            config[seg] = True

    for k, v in config.items():
        if isinstance(v, str) and re.match(r"^-?\d+(\.\d+)?$", v):
            config[k] = float(v)

    return config


def process_all_games_sql(db_path="sumobot_data.duckdb"):
    """
    Use DuckDB SQL to do heavy lifting, return structured data per game.
    This is MUCH faster than Python loops.
    """

    con = duckdb.connect(db_path, read_only=True)

    print("🔍 Computing game metrics in DuckDB SQL...")

    # Complex SQL that computes everything we need per game
    query = """
    WITH action_data AS (
        SELECT
            GameIndex,
            Actor,
            Name,
            State,
            StartedAt,
            UpdatedAt,
            bot_a,
            bot_b,
            config_name,
            GameWinner,
            Category
        FROM game_logs
        WHERE Category = 'Action'
    ),

    -- Compute LAG for durations
    action_with_lag AS (
        SELECT
            GameIndex,
            Actor,
            Name,
            State,
            StartedAt,
            UpdatedAt,
            bot_a,
            bot_b,
            config_name,
            GameWinner,
            LAG(StartedAt) OVER (
                PARTITION BY GameIndex, Actor, Name
                ORDER BY UpdatedAt
            ) as prev_started_at
        FROM action_data
    ),

    -- Compute durations per game/actor/action
    action_durations AS (
        SELECT
            GameIndex,
            bot_a,
            bot_b,
            config_name,
            Actor,
            Name,
            SUM(
                CASE
                    WHEN State = 2 AND prev_started_at IS NOT NULL THEN
                        UpdatedAt - prev_started_at
                    ELSE 0
                END
            ) as ActualDuration
        FROM action_with_lag
        GROUP BY GameIndex, bot_a, bot_b, config_name, Actor, Name
    ),

    -- Action counts per game
    action_counts AS (
        SELECT
            GameIndex,
            bot_a,
            bot_b,
            config_name,
            Actor,
            Name,
            COUNT(*) as action_count
        FROM action_data
        WHERE State != 2
        GROUP BY GameIndex, bot_a, bot_b, config_name, Actor, Name
    ),

    -- Collision counts per game (separated by type)
    collision_data AS (
        SELECT
            GameIndex,
            bot_a,
            bot_b,
            config_name,
            SUM(CASE WHEN Actor = 0 AND ColTieBreaker = 0 AND ColActor = 1 THEN 1 ELSE 0 END) as collision_L,
            SUM(CASE WHEN Actor = 1 AND ColTieBreaker = 0 AND ColActor = 1 THEN 1 ELSE 0 END) as collision_R,
            SUM(COALESCE(ColTieBreaker, 0)) as collision_tie
        FROM game_logs
        WHERE Category = 'Collision'
            AND State != 2
        GROUP BY GameIndex, bot_a, bot_b, config_name
    ),

    -- Game metadata
    game_meta AS (
        SELECT DISTINCT
            GameIndex,
            bot_a,
            bot_b,
            config_name,
            GameWinner,
            MAX(UpdatedAt) as match_duration
        FROM game_logs
        GROUP BY GameIndex, bot_a, bot_b, config_name, GameWinner
    )

    -- Final aggregation per game
    SELECT
        gm.GameIndex,
        gm.GameWinner as Winner,
        gm.bot_a as Bot_L,
        gm.bot_b as Bot_R,
        gm.config_name,
        gm.match_duration as MatchDur,

        -- Durations
        COALESCE(SUM(CASE WHEN ad.Actor = 0 THEN ad.ActualDuration END), 0) as Duration_L,
        COALESCE(SUM(CASE WHEN ad.Actor = 1 THEN ad.ActualDuration END), 0) as Duration_R,

        -- Action counts
        COALESCE(SUM(CASE WHEN ac.Actor = 0 THEN ac.action_count END), 0) as ActionCounts_L,
        COALESCE(SUM(CASE WHEN ac.Actor = 1 THEN ac.action_count END), 0) as ActionCounts_R,

        -- Total actions
        COALESCE(SUM(ac.action_count), 0) as TotalActions,

        -- Collisions
        COALESCE(MAX(cd.collision_L), 0) as Collisions_L,
        COALESCE(MAX(cd.collision_R), 0) as Collisions_R,
        COALESCE(MAX(cd.collision_tie), 0) as Collisions_Tie,

        -- Action-specific counts
        COALESCE(SUM(CASE WHEN ac.Actor = 0 AND ac.Name = 'Accelerate' THEN ac.action_count END), 0) as Accelerate_Act_L,
        COALESCE(SUM(CASE WHEN ac.Actor = 0 AND ac.Name = 'TurnLeft' THEN ac.action_count END), 0) as TurnLeft_Act_L,
        COALESCE(SUM(CASE WHEN ac.Actor = 0 AND ac.Name = 'TurnRight' THEN ac.action_count END), 0) as TurnRight_Act_L,
        COALESCE(SUM(CASE WHEN ac.Actor = 0 AND ac.Name = 'Dash' THEN ac.action_count END), 0) as Dash_Act_L,
        COALESCE(SUM(CASE WHEN ac.Actor = 0 AND ac.Name = 'SkillBoost' THEN ac.action_count END), 0) as SkillBoost_Act_L,
        COALESCE(SUM(CASE WHEN ac.Actor = 0 AND ac.Name = 'SkillStone' THEN ac.action_count END), 0) as SkillStone_Act_L,

        COALESCE(SUM(CASE WHEN ac.Actor = 1 AND ac.Name = 'Accelerate' THEN ac.action_count END), 0) as Accelerate_Act_R,
        COALESCE(SUM(CASE WHEN ac.Actor = 1 AND ac.Name = 'TurnLeft' THEN ac.action_count END), 0) as TurnLeft_Act_R,
        COALESCE(SUM(CASE WHEN ac.Actor = 1 AND ac.Name = 'TurnRight' THEN ac.action_count END), 0) as TurnRight_Act_R,
        COALESCE(SUM(CASE WHEN ac.Actor = 1 AND ac.Name = 'Dash' THEN ac.action_count END), 0) as Dash_Act_R,
        COALESCE(SUM(CASE WHEN ac.Actor = 1 AND ac.Name = 'SkillBoost' THEN ac.action_count END), 0) as SkillBoost_Act_R,
        COALESCE(SUM(CASE WHEN ac.Actor = 1 AND ac.Name = 'SkillStone' THEN ac.action_count END), 0) as SkillStone_Act_R,

        -- Action-specific durations
        COALESCE(SUM(CASE WHEN ad.Actor = 0 AND ad.Name = 'Accelerate' THEN ad.ActualDuration END), 0) as Accelerate_Dur_L,
        COALESCE(SUM(CASE WHEN ad.Actor = 0 AND ad.Name = 'TurnLeft' THEN ad.ActualDuration END), 0) as TurnLeft_Dur_L,
        COALESCE(SUM(CASE WHEN ad.Actor = 0 AND ad.Name = 'TurnRight' THEN ad.ActualDuration END), 0) as TurnRight_Dur_L,
        COALESCE(SUM(CASE WHEN ad.Actor = 0 AND ad.Name = 'Dash' THEN ad.ActualDuration END), 0) as Dash_Dur_L,
        COALESCE(SUM(CASE WHEN ad.Actor = 0 AND ad.Name = 'SkillBoost' THEN ad.ActualDuration END), 0) as SkillBoost_Dur_L,
        COALESCE(SUM(CASE WHEN ad.Actor = 0 AND ad.Name = 'SkillStone' THEN ad.ActualDuration END), 0) as SkillStone_Dur_L,

        COALESCE(SUM(CASE WHEN ad.Actor = 1 AND ad.Name = 'Accelerate' THEN ad.ActualDuration END), 0) as Accelerate_Dur_R,
        COALESCE(SUM(CASE WHEN ad.Actor = 1 AND ad.Name = 'TurnLeft' THEN ad.ActualDuration END), 0) as TurnLeft_Dur_R,
        COALESCE(SUM(CASE WHEN ad.Actor = 1 AND ad.Name = 'TurnRight' THEN ad.ActualDuration END), 0) as TurnRight_Dur_R,
        COALESCE(SUM(CASE WHEN ad.Actor = 1 AND ad.Name = 'Dash' THEN ad.ActualDuration END), 0) as Dash_Dur_R,
        COALESCE(SUM(CASE WHEN ad.Actor = 1 AND ad.Name = 'SkillBoost' THEN ad.ActualDuration END), 0) as SkillBoost_Dur_R,
        COALESCE(SUM(CASE WHEN ad.Actor = 1 AND ad.Name = 'SkillStone' THEN ad.ActualDuration END), 0) as SkillStone_Dur_R

    FROM game_meta gm
    LEFT JOIN action_durations ad ON
        gm.GameIndex = ad.GameIndex AND
        gm.bot_a = ad.bot_a AND
        gm.bot_b = ad.bot_b AND
        gm.config_name = ad.config_name
    LEFT JOIN action_counts ac ON
        gm.GameIndex = ac.GameIndex AND
        gm.bot_a = ac.bot_a AND
        gm.bot_b = ac.bot_b AND
        gm.config_name = ac.config_name
    LEFT JOIN collision_data cd ON
        gm.GameIndex = cd.GameIndex AND
        gm.bot_a = cd.bot_a AND
        gm.bot_b = cd.bot_b AND
        gm.config_name = cd.config_name
    GROUP BY gm.GameIndex, gm.GameWinner, gm.bot_a, gm.bot_b, gm.config_name, gm.match_duration
    ORDER BY gm.bot_a, gm.bot_b, gm.config_name, gm.GameIndex
    """

    # Execute and get as Polars DataFrame
    game_metrics = con.execute(query).pl()

    con.close()

    print(f"✅ Computed metrics for {len(game_metrics):,} games")

    # Parse config names and add columns
    print("🔧 Parsing config names...")
    config_data = []
    for config_name in game_metrics['config_name'].unique():
        parsed = parse_config_name(config_name)
        config_data.append({
            'config_name': config_name,
            'Timer': parsed.get('Timer'),
            'ActInterval': parsed.get('ActInterval'),
            'Round': parsed.get('Round'),
            'SkillLeft': parsed.get('SkillLeft'),
            'SkillRight': parsed.get('SkillRight')
        })

    config_df = pl.DataFrame(config_data)
    game_metrics = game_metrics.join(config_df, on='config_name', how='left')

    # Drop config_name column (no longer needed)
    game_metrics = game_metrics.drop('config_name')

    return game_metrics


def compute_time_bins(db_path="sumobot_data.duckdb", time_bin_size=5):
    """
    Compute time-binned ACTION counts only.
    Matches original generator.py logic.
    """
    con = duckdb.connect(db_path, read_only=True)

    print("🔍 Computing time-binned action data...")

    # Get all action data with game metadata
    query = """
    SELECT
        gl.GameIndex,
        gl.bot_a,
        gl.bot_b,
        gl.config_name,
        gl.Actor,
        gl.UpdatedAt,
        gl.Name,
        gm.match_duration
    FROM game_logs gl
    JOIN (
        SELECT
            GameIndex,
            bot_a,
            bot_b,
            config_name,
            MAX(UpdatedAt) as match_duration
        FROM game_logs
        GROUP BY GameIndex, bot_a, bot_b, config_name
    ) gm ON
        gl.GameIndex = gm.GameIndex AND
        gl.bot_a = gm.bot_a AND
        gl.bot_b = gm.bot_b AND
        gl.config_name = gm.config_name
    WHERE gl.Category = 'Action'
        AND gl.State != 2
    ORDER BY gl.bot_a, gl.bot_b, gl.config_name, gl.GameIndex, gl.UpdatedAt
    """

    raw_data = con.execute(query).pl()
    con.close()

    print(f"   Retrieved {len(raw_data):,} action events")

    # Parse config names
    config_data = []
    for config_name in raw_data['config_name'].unique():
        parsed = parse_config_name(config_name)
        config_data.append({
            'config_name': config_name,
            'Timer': parsed.get('Timer'),
            'ActInterval': parsed.get('ActInterval'),
            'Round': parsed.get('Round'),
            'SkillLeft': parsed.get('SkillLeft'),
            'SkillRight': parsed.get('SkillRight')
        })

    config_df = pl.DataFrame(config_data)
    raw_data = raw_data.join(config_df, on='config_name', how='left')

    # Process time bins per game
    time_fragment_list = []

    # Group by game
    for (game_idx, bot_a, bot_b, config_name), game_df in raw_data.group_by(['GameIndex', 'bot_a', 'bot_b', 'config_name']):
        match_dur = game_df['match_duration'][0]

        # Create time bins
        bins = np.arange(0, match_dur + time_bin_size, time_bin_size)

        if len(bins) < 2:
            continue

        # Convert to pandas for pd.cut (Polars doesn't have equivalent)
        game_pd = game_df.to_pandas()

        # Process actions
        for side in [0, 1]:
            actor_data = game_pd[game_pd['Actor'] == side]

            if len(actor_data) == 0:
                continue

            # Assign time bins
            actor_data = actor_data.copy()
            actor_data['TimeBin'] = pd.cut(actor_data['UpdatedAt'], bins=bins, labels=bins[:-1], include_lowest=True)

            # Get config info
            config_info = game_df.filter(pl.col('Actor') == side).select(['Timer', 'ActInterval', 'Round', 'SkillLeft', 'SkillRight']).row(0)

            # Count per time bin and action
            grouped = actor_data.groupby(['TimeBin', 'Name']).size().reset_index(name='Count')

            for _, row in grouped.iterrows():
                time_fragment_list.append({
                    'GameIndex': game_idx,
                    'Bot': bot_a if side == 0 else bot_b,
                    'Timer': config_info[0],
                    'ActInterval': config_info[1],
                    'Round': config_info[2],
                    'SkillLeft': config_info[3],
                    'SkillRight': config_info[4],
                    'TimeBin': float(row['TimeBin']),
                    'Action': row['Name'],
                    'Count': row['Count']
                })

    time_fragment_df = pl.DataFrame(time_fragment_list)

    print(f"✅ Computed {len(time_fragment_df):,} time-binned records")

    return time_fragment_df


def compute_collision_time_bins(db_path="sumobot_data.duckdb", time_bin_size=5):
    """
    Compute time-binned COLLISION data with Actor, Target, and Tie information.
    Creates Actor_L, Target_L, Tie_L for left side and Actor_R, Target_R, Tie_R for right side.
    """
    con = duckdb.connect(db_path, read_only=True)

    print("🔍 Computing time-binned collision data...")

    # Get all collision data with game metadata
    query = """
    SELECT
        gl.GameIndex,
        gl.bot_a,
        gl.bot_b,
        gl.config_name,
        gl.Actor,
        gl.Target,
        gl.ColTieBreaker,
        gl.UpdatedAt,
        gl.ColActor,
        gm.match_duration
    FROM game_logs gl
    JOIN (
        SELECT
            GameIndex,
            bot_a,
            bot_b,
            config_name,
            MAX(UpdatedAt) as match_duration
        FROM game_logs
        GROUP BY GameIndex, bot_a, bot_b, config_name
    ) gm ON
        gl.GameIndex = gm.GameIndex AND
        gl.bot_a = gm.bot_a AND
        gl.bot_b = gm.bot_b AND
        gl.config_name = gm.config_name
    WHERE gl.Category = 'Collision'
        AND gl.State == 0
    ORDER BY gl.bot_a, gl.bot_b, gl.config_name, gl.GameIndex, gl.UpdatedAt
    """

    raw_data = con.execute(query).pl()
    con.close()

    print(f"   Retrieved {len(raw_data):,} collision events")

    # Parse config names
    config_data = []
    for config_name in raw_data['config_name'].unique():
        parsed = parse_config_name(config_name)
        config_data.append({
            'config_name': config_name,
            'Timer': parsed.get('Timer'),
            'ActInterval': parsed.get('ActInterval'),
            'Round': parsed.get('Round'),
            'SkillLeft': parsed.get('SkillLeft'),
            'SkillRight': parsed.get('SkillRight')
        })

    config_df = pl.DataFrame(config_data)
    raw_data = raw_data.join(config_df, on='config_name', how='left')

    # Process collision time bins per game
    collision_fragment_list = []

    # Group by game
    for (game_idx, bot_a, bot_b, config_name), game_df in raw_data.group_by(['GameIndex', 'bot_a', 'bot_b', 'config_name']):
        match_dur = game_df['match_duration'][0]

        # Create time bins
        bins = np.arange(0, match_dur + time_bin_size, time_bin_size)

        if len(bins) < 2:
            continue

        # Convert to pandas for pd.cut
        game_pd = game_df.to_pandas()

        # Get config info
        config_info = game_df.select(['Timer', 'ActInterval', 'Round', 'SkillLeft', 'SkillRight']).row(0)

        # For each time bin, collect collision info
        game_pd['TimeBin'] = pd.cut(game_pd['UpdatedAt'], bins=bins, labels=bins[:-1], include_lowest=True)

        # Group by time bin
        for time_bin, bin_data in game_pd.groupby('TimeBin'):
            # Separate left (Actor==0) and right (Actor==1) collisions
            actor_L_count = len(bin_data[(bin_data['Actor'] == 0) & (bin_data['ColTieBreaker'] == 0) & (bin_data["ColActor"] == 1)])
            actor_R_count = len(bin_data[(bin_data['Actor'] == 1) & (bin_data['ColTieBreaker'] == 0) & (bin_data["ColActor"] == 1)])

            tie = bin_data['ColTieBreaker'].sum() if 'ColTieBreaker' in bin_data.columns else 0

            collision_fragment_list.append({
                'GameIndex': game_idx,
                'Bot_L': bot_a,
                'Bot_R': bot_b,
                'Timer': config_info[0],
                'ActInterval': config_info[1],
                'Round': config_info[2],
                'SkillLeft': config_info[3],
                'SkillRight': config_info[4],
                'TimeBin': float(time_bin),
                'Actor_L': actor_L_count,
                'Actor_R': actor_R_count,
                'Tie': int(tie),
            })

    collision_fragment_df = pl.DataFrame(collision_fragment_list)

    print(f"✅ Computed {len(collision_fragment_df):,} collision time-binned records")

    return collision_fragment_df


def create_summary_matchup(all_games):
    """Create matchup summary using Polars"""
    group_cols = ["Bot_L", "Bot_R", "Timer", "ActInterval", "Round", "SkillLeft", "SkillRight"]

    # Find all action-specific columns
    action_cols = [col for col in all_games.columns if any(col.endswith(suffix) for suffix in ("_Act_L", "_Act_R", "_Dur_L", "_Dur_R"))]

    # Build aggregation list
    agg_list = [
        pl.col("GameIndex").n_unique().alias("Games"),
        (pl.col("Winner") == 0).sum().alias("Winner_L"),
        (pl.col("Winner") == 1).sum().alias("Winner_R"),
        pl.col("ActionCounts_L").sum(),
        pl.col("ActionCounts_R").sum(),
        pl.col("TotalActions").sum(),
        pl.col("Duration_L").sum(),
        pl.col("Duration_R").sum(),
        pl.col("Collisions_L").sum(),
        pl.col("Collisions_R").sum(),
        pl.col("Collisions_Tie").sum(),
        pl.col("MatchDur").mean(),
    ]

    # Add all action-specific columns
    for col in action_cols:
        agg_list.append(pl.col(col).sum())

    matchup_summary = all_games.group_by(group_cols).agg(agg_list)

    # Add win rates
    matchup_summary = matchup_summary.with_columns([
        (pl.col("Winner_L") / pl.col("Games")).alias("WinRate_L"),
        (pl.col("Winner_R") / pl.col("Games")).alias("WinRate_R")
    ])

    # Compute bot rankings based on overall performance
    # Aggregate left bots
    bot_summary_L = matchup_summary.group_by("Bot_L").agg([
        pl.col("Games").sum().alias("TotalGames"),
        pl.col("Winner_L").sum().alias("TotalWins"),
    ]).rename({"Bot_L": "Bot"})

    # Aggregate right bots
    bot_summary_R = matchup_summary.group_by("Bot_R").agg([
        pl.col("Games").sum().alias("TotalGames"),
        pl.col("Winner_R").sum().alias("TotalWins"),
    ]).rename({"Bot_R": "Bot"})

    # Combine and compute ranks
    bot_ranks = pl.concat([bot_summary_L, bot_summary_R])
    bot_ranks = bot_ranks.group_by("Bot").agg([
        pl.col("TotalGames").sum(),
        pl.col("TotalWins").sum(),
    ])
    bot_ranks = bot_ranks.with_columns([
        (pl.col("TotalWins") / pl.col("TotalGames")).alias("WinRate")
    ])
    bot_ranks = bot_ranks.with_columns([
        pl.col("WinRate").rank(descending=True).cast(pl.Int32).alias("Rank")
    ])
    bot_ranks = bot_ranks.select(["Bot", "Rank"])

    # Join ranks back to matchup summary
    matchup_summary = matchup_summary.join(
        bot_ranks.rename({"Bot": "Bot_L", "Rank": "Rank_L"}),
        on="Bot_L",
        how="left"
    )
    matchup_summary = matchup_summary.join(
        bot_ranks.rename({"Bot": "Bot_R", "Rank": "Rank_R"}),
        on="Bot_R",
        how="left"
    )

    # Sort
    matchup_summary = matchup_summary.sort(["Bot_L", "Bot_R", "Timer", "ActInterval"])

    # Save to CSV
    matchup_summary.write_csv("summary_matchup.csv")
    print("✅ Saved summary_matchup.csv")

    return matchup_summary


def create_summary_bot(matchup_summary):
    """Create bot summary using Polars"""

    bot_summary_L = matchup_summary.group_by("Bot_L").agg([
        pl.col("Games").sum().alias("TotalGames"),
        pl.col("Winner_L").sum().alias("TotalWins"),
        pl.col("Duration_L").sum().alias("Duration"),
        pl.col("ActionCounts_L").sum().alias("TotalActions"),
        pl.col("Collisions_L").sum().alias("Collisions"),
        pl.col("Collisions_Tie").sum().alias("CollisionsTie"),
    ]).rename({"Bot_L": "Bot"})

    bot_summary_R = matchup_summary.group_by("Bot_R").agg([
        pl.col("Games").sum().alias("TotalGames"),
        pl.col("Winner_R").sum().alias("TotalWins"),
        pl.col("Duration_R").sum().alias("Duration"),
        pl.col("ActionCounts_R").sum().alias("TotalActions"),
        pl.col("Collisions_R").sum().alias("Collisions"),
        pl.col("Collisions_Tie").sum().alias("CollisionsTie"),
    ]).rename({"Bot_R": "Bot"})

    # Combine
    bot_summary = pl.concat([bot_summary_L, bot_summary_R])

    # Aggregate
    bot_summary = bot_summary.group_by("Bot").agg([
        pl.col("TotalGames").sum(),
        pl.col("TotalWins").sum(),
        pl.col("Duration").sum(),
        pl.col("TotalActions").sum(),
        pl.col("Collisions").sum(),
        pl.col("CollisionsTie").sum(),
    ])

    # Compute WinRate
    bot_summary = bot_summary.with_columns([
        (pl.col("TotalWins") / pl.col("TotalGames")).alias("WinRate")
    ])

    # Compute Rank
    bot_summary = bot_summary.with_columns([
        pl.col("WinRate").rank(descending=True).cast(pl.Int32).alias("Rank")
    ])

    # Sort
    bot_summary = bot_summary.sort("Rank")

    # Save
    bot_summary.write_csv("summary_bot.csv")
    print("✅ Saved summary_bot.csv")

    return bot_summary


def summarize_action_timebins(time_fragment_df):
    """
    Summarize action time fragment data.
    Computes mean counts per bot/config/timebin/action.
    """
    print("📊 Summarizing action time-binned data...")

    # Group and compute mean
    summary = time_fragment_df.group_by(
        ['Bot', 'Timer', 'ActInterval', 'Round', 'TimeBin', 'Action']
    ).agg([
        pl.col('Count').mean().alias('MeanCount')
    ])

    # Sort for readability
    summary = summary.sort(['Bot', 'Timer', 'ActInterval', 'Round', 'TimeBin', 'Action'])

    # Save CSV
    summary.write_csv("summary_action_timebins.csv")
    print("✅ Saved summary_action_timebins.csv")

    return summary


def create_collision_details(collision_fragment_df):
    """
    Calculate collision time fragment data.
    Aggregates Actor, Target, Tie counts per config/timebin.
    """
    print("📊 Creating collision detail time-binned data...")

    # Group and compute mean for each collision metric
    summary = collision_fragment_df.group_by(
        ['Bot_L', 'Bot_R', 'Timer', 'ActInterval', 'Round', 'TimeBin']
    ).agg([
        pl.col('Actor_L').sum().alias('Actor_L'),
        pl.col('Actor_R').sum().alias('Actor_R'),
        pl.col('Tie').sum().alias('Tie'),
    ])

    # Sort for readability
    summary = summary.sort(['Bot_L', 'Bot_R', 'Timer', 'ActInterval', 'Round', 'TimeBin'])

    # Save CSV
    summary.write_csv("collision_details.csv")
    print("✅ Saved collision_details.csv")

    return summary


def generate(db_path="sumobot_data.duckdb", time_bin_size=5):
    """Main generation function"""

    print("=" * 60)
    print("🚀 DuckDB + Polars: Maximum Performance")
    print("=" * 60)

    # Process all games in SQL
    print("\n📊 Computing game metrics with DuckDB...")
    all_games = process_all_games_sql(db_path)

    print(f"\n✅ Processed {len(all_games):,} games")

    # Compute action time bins
    # print("\n📊 Computing action time-binned data...")
    # action_timebin_df = compute_time_bins(db_path, time_bin_size=time_bin_size)

    # Compute collision time bins
    # print("\n📊 Computing collision time-binned data...")
    # collision_timebin_df = compute_collision_time_bins(db_path, time_bin_size=time_bin_size)

    # Create summaries with Polars
    print("\n📊 Creating matchup summary with Polars...")
    matchup_summary = create_summary_matchup(all_games)

    print("\n📊 Creating bot summary with Polars...")
    bot_summary = create_summary_bot(matchup_summary)

    # print("\n📊 Creating action time-bin summary with Polars...")
    # action_timebin_summary = summarize_action_timebins(action_timebin_df)

    # print("\n📊 Creating collision time-bin summary with Polars...")
    # collision_timebin_summary = create_collision_details(collision_timebin_df)

    print("\n" + "=" * 60)
    print("🎉 Done! Created:")
    print("   - summary_matchup.csv")
    print("   - summary_bot.csv")
    print("   - summary_action_timebins.csv")
    print("   - summary_collision_timebins.csv")
    print("=" * 60)

    return matchup_summary, bot_summary, 
# action_timebin_summary, collision_timebin_summary


if __name__ == "__main__":
    import sys

    db_path = "sumobot_data.duckdb"
    time_bin_size = 3  # Default: 5 second bins

    # Optional: accept time_bin_size from command line
    if len(sys.argv) > 1:
        time_bin_size = float(sys.argv[1])
        print(f"Using time_bin_size = {time_bin_size}")

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    generate(db_path, time_bin_size=time_bin_size)
