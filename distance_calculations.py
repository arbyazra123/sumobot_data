"""
Distance calculation functions for sumobot analyzer
"""
import polars as pl
import numpy as np
from analyzer_config import arena_center


def calculate_distance_between_bots(df):
    """
    Calculate distance between Bot 1 (Actor 0) and Bot 2 (Actor 1) for each game frame

    Args:
        df: Polars DataFrame with columns including Actor, BotPosX, BotPosY, GameIndex, UpdatedAt

    Returns:
        Polars DataFrame with distance between bots for each frame
    """
    # Split data by actor - cast Actor inline for filtering
    bot1_df = df.filter(pl.col("Actor").cast(pl.Int64) == 0).select([
        "GameIndex", "UpdatedAt", "BotPosX", "BotPosY"
    ]).rename({"BotPosX": "Bot1_X", "BotPosY": "Bot1_Y"})

    bot2_df = df.filter(pl.col("Actor").cast(pl.Int64) == 1).select([
        "GameIndex", "UpdatedAt", "BotPosX", "BotPosY"
    ]).rename({"BotPosX": "Bot2_X", "BotPosY": "Bot2_Y"})

    # Merge on GameIndex and UpdatedAt to align frames
    merged = bot1_df.join(bot2_df, on=["GameIndex", "UpdatedAt"], how="inner")

    # Calculate Euclidean distance
    merged = merged.with_columns([
        (((pl.col("Bot1_X") - pl.col("Bot2_X"))**2 +
          (pl.col("Bot1_Y") - pl.col("Bot2_Y"))**2).sqrt()).alias("Distance")
    ])

    return merged


def calculate_distance_from_center(df):
    """
    Calculate distance from arena center for each bot

    Args:
        df: Polars DataFrame with columns including Actor, BotPosX, BotPosY

    Returns:
        Polars DataFrame with distance from center for each bot
    """
    # Calculate distance from center for each position
    df = df.with_columns([
        (((pl.col("BotPosX") - arena_center[0])**2 +
          (pl.col("BotPosY") - arena_center[1])**2).sqrt()).alias("DistanceFromCenter")
    ])

    return df


def split_into_phases(df, num_phases=3):
    """
    Split game data into phases based on UpdatedAt time PER GAME.
    Each game (GameIndex) is split into early/mid/late phases independently.

    Args:
        df: Polars DataFrame with game data (must have GameIndex and UpdatedAt columns)
        num_phases: Number of phases to split into (default: 3 for early/mid/late)

    Returns:
        List of Polars DataFrames, one per phase (aggregated across all games)
    """
    if df.is_empty():
        return [pl.DataFrame()] * num_phases

    # Initialize phase containers
    phases = [[] for _ in range(num_phases)]

    # Process each game independently
    for game_idx in df["GameIndex"].unique().to_list():
        game_df = df.filter(pl.col("GameIndex") == game_idx)

        if game_df.is_empty():
            continue

        # Calculate time boundaries for THIS game
        min_time = game_df["UpdatedAt"].min()
        max_time = game_df["UpdatedAt"].max()
        time_range = max_time - min_time

        # Avoid division by zero for games with no time range
        if time_range == 0:
            # Put all data in the first phase
            phases[0].append(game_df)
            continue

        phase_size = time_range / num_phases

        # Split this game into phases
        for i in range(num_phases):
            phase_start = min_time + (i * phase_size)
            phase_end = min_time + ((i + 1) * phase_size)

            if i == num_phases - 1:
                # Include the last timestamp in the final phase
                phase_df = game_df.filter(
                    (pl.col("UpdatedAt") >= phase_start) & (pl.col("UpdatedAt") <= phase_end)
                )
            else:
                phase_df = game_df.filter(
                    (pl.col("UpdatedAt") >= phase_start) & (pl.col("UpdatedAt") < phase_end)
                )

            if not phase_df.is_empty():
                phases[i].append(phase_df)

    # Concatenate all games for each phase
    result_phases = []
    for phase_data in phases:
        if phase_data:
            result_phases.append(pl.concat(phase_data))
        else:
            result_phases.append(pl.DataFrame())

    return result_phases
