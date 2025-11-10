import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import glob
from tqdm import tqdm

# =====================
# Config
# =====================
arena_center = np.array([0.24, 1.97])
arena_radius = 4.73485

# Adjustable parameters
tile_size = 0.7   # Larger = bigger heatmap tiles (lower resolution)
# arrow_size = 50   # Larger = longer arrows

def load_data_chunked(csv_path, chunksize=50000, actor_filter=None):
    """
    Load CSV data in chunks to handle large files efficiently

    Args:
        csv_path: Path to CSV file
        chunksize: Number of rows per chunk
        actor_filter: Filter for specific actor (0 for left, 1 for right, None for both)
    """
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # Filter by actor if specified
        if actor_filter is not None:
            chunk = chunk[chunk["Actor"].astype(str) == str(actor_filter)]

        # Drop invalid entries
        chunk = chunk.dropna(subset=["BotPosX", "BotPosY", "BotRot"])

        if not chunk.empty:
            chunks.append(chunk)

    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()

def split_into_phases(df, num_phases=3):
    """
    Split game data into phases based on UpdatedAt time PER GAME.
    Each game (GameIndex) is split into early/mid/late phases independently.

    Args:
        df: DataFrame with game data (must have GameIndex and UpdatedAt columns)
        num_phases: Number of phases to split into (default: 3 for early/mid/late)

    Returns:
        List of DataFrames, one per phase (aggregated across all games)
    """
    if df.empty:
        return [pd.DataFrame()] * num_phases

    # Initialize phase containers
    phases = [[] for _ in range(num_phases)]

    # Process each game independently
    for game_idx in df["GameIndex"].unique():
        game_df = df[df["GameIndex"] == game_idx].copy()

        if game_df.empty:
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
                phase_df = game_df[(game_df["UpdatedAt"] >= phase_start) & (game_df["UpdatedAt"] <= phase_end)]
            else:
                phase_df = game_df[(game_df["UpdatedAt"] >= phase_start) & (game_df["UpdatedAt"] < phase_end)]

            if not phase_df.empty:
                phases[i].append(phase_df)

    # Concatenate all games for each phase
    result_phases = []
    for phase_data in phases:
        if phase_data:
            result_phases.append(pd.concat(phase_data, ignore_index=True))
        else:
            result_phases.append(pd.DataFrame())

    return result_phases

def create_heatmap_data(x, y, tile_size):
    """Create heatmap data from position coordinates"""
    if len(x) == 0:
        return None, None, None

    xrange = np.arange(x.min(), x.max() + tile_size, tile_size)
    yrange = np.arange(y.min(), y.max() + tile_size, tile_size)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[xrange, yrange])

    return heatmap, xedges, yedges

def plot_phase_heatmap(ax, phase_df, phase_name):
    """Plot contour density heatmap for a single phase"""
    if phase_df.empty:
        ax.text(0.5, 0.5, f"No data for {phase_name}",
                ha='center', va='center', transform=ax.transAxes)
        return

    x = phase_df["BotPosX"].values
    y = phase_df["BotPosY"].values

    # Create 2D kernel density estimation for smooth contours
    if len(x) > 1:
        from scipy.stats import gaussian_kde

        # Create KDE
        try:
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)

            # Create grid for evaluation
            x_min, x_max = arena_center[0] - arena_radius - 1, arena_center[0] + arena_radius + 1
            y_min, y_max = arena_center[1] - arena_radius - 1, arena_center[1] + arena_radius + 1

            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = np.reshape(kde(positions).T, xx.shape)

            # Plot filled contours (density heatmap)
            ax.contourf(xx, yy, density, levels=15, cmap="Greens", alpha=0.8, zorder=1)

            # Optionally add contour lines for better definition
            ax.contour(xx, yy, density, levels=5, colors='darkgreen', alpha=0.3, linewidths=0.5, zorder=2)

        except Exception as e:
            # Fallback to scatter if KDE fails
            print(f"Warning: KDE failed for {phase_name}, using scatter plot. Error: {e}")
            ax.scatter(x, y, alpha=0.1, s=1, c='green', zorder=1)

    # Draw arena boundary AFTER contours so it appears on top
    circle = plt.Circle(arena_center, arena_radius,
                       fill=False, edgecolor="red",
                       linewidth=2, linestyle="--", zorder=3)
    ax.add_artist(circle)

    # Labels & Arena Bounds
    ax.set_title(f"{phase_name}\n(n={len(phase_df):,} samples)")
    ax.set_xlabel("BotPosX")
    ax.set_ylabel("BotPosY")
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(arena_center[0] - arena_radius - 1, arena_center[0] + arena_radius + 1)
    ax.set_ylim(arena_center[1] - arena_radius - 1, arena_center[1] + arena_radius + 1)

    # Add grid
    ax.grid(True, alpha=0.3, zorder=0)


def plot_position_distribution(df_combined, bot_name, actor_position="both"):
    """
    Plot X and Y position distributions in a single frame (overlaid histograms)
    Y values are shifted by -2 since the game starts at y=2

    Args:
        df_combined: Combined DataFrame with bot position data
        bot_name: Name of the bot
        actor_position: Position filter text for title

    Returns:
        matplotlib figure
    """
    if df_combined.empty:
        return None

    x = df_combined["BotPosX"].values
    y = df_combined["BotPosY"].values - 2  # Shift Y by -2 (start position is at y=2)

    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot X distribution
    ax.hist(x, bins=100, alpha=0.7, color='green', edgecolor='darkgreen',
            label=f'{bot_name} X', linewidth=0.5)

    # Plot Y distribution (overlaid, shifted)
    ax.hist(y, bins=100, alpha=0.7, color='red', edgecolor='darkred',
            label=f'{bot_name} Y (shifted from start)', linewidth=0.5)

    # Customize plot
    position_text = f" ({actor_position} side)" if actor_position != "both" else ""
    ax.set_title(f"Distribution of {bot_name} Positions (Overlayed){position_text}\n(n={len(df_combined):,} samples)",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Position (Y shifted by -2 from start)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig

def create_phased_heatmap(csv_path, output_path=None, chunksize=50000):
    """
    Create a 3-phase heatmap visualization (early, mid, late game)

    Args:
        csv_path: Path to the game log CSV file
        output_path: Path to save the output image (optional)
        chunksize: Size of chunks for reading large CSV files
    """
    print(f"Loading data from {csv_path}...")
    df = load_data_chunked(csv_path, chunksize)

    if df.empty:
        print("No valid data found in the CSV file.")
        return

    print(f"Total samples: {len(df):,}")
    print(f"Time range: {df['UpdatedAt'].min():.2f} - {df['UpdatedAt'].max():.2f}")

    # Split into phases
    print("Splitting into phases...")
    phases = split_into_phases(df, num_phases=3)
    phase_names = ["Early Game", "Mid Game", "Late Game"]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Plot each phase
    for ax, phase_df, phase_name in zip(axes, phases, phase_names):
        print(f"Plotting {phase_name}...")
        plot_phase_heatmap(ax, phase_df, phase_name)

    plt.suptitle(f"Sumobot Arena Heatmap - Phased Analysis\n{Path(csv_path).name}",
                 fontsize=16, y=0.98)
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

def load_bot_data_from_simulation(base_dir, bot_name, actor_position="left", chunksize=50000, max_configs=None):
    """
    Load all CSV data for a specific bot from the simulation directory

    Args:
        base_dir: Base simulation directory
        bot_name: Name of the bot (e.g., "Bot_BT", "Bot_NN", "Bot_Primitive")
        actor_position: "left" (Actor 0) or "right" (Actor 1) or "both"
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of config folders to process (None for all)

    Returns:
        Combined DataFrame with all bot data
    """
    all_data = []

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
        if len(parts) == 2:
            left_bot = parts[0]
            is_left_bot = (bot_name == left_bot)

            if actor_position == "left" and is_left_bot:
                actor_filter = 0
            elif actor_position == "left" and not is_left_bot:
                continue  # Skip this matchup
            elif actor_position == "right" and not is_left_bot:
                actor_filter = 1
            elif actor_position == "right" and is_left_bot:
                continue  # Skip this matchup
            elif actor_position == "both":
                actor_filter = 0 if is_left_bot else 1
            else:
                continue
        else:
            continue

        # Get all config folders
        config_folders = [f for f in os.listdir(matchup_path)
                         if os.path.isdir(os.path.join(matchup_path, f))]

        if max_configs:
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

                if not df.empty:
                    all_data.append(df)
                    total_csvs += 1

    if not all_data:
        print("No valid data found.")
        return pd.DataFrame()

    print(f"\nLoaded {total_csvs} CSV files")
    print("Combining all data...")
    df_combined = pd.concat(all_data, ignore_index=True)

    print(f"Total samples: {len(df_combined):,}")

    return df_combined


def create_phased_heatmap_for_bot(base_dir, bot_name, actor_position="left", output_path=None, chunksize=50000, max_configs=None):
    """
    Create a 3-phase heatmap for a specific bot from simulation directory

    Args:
        base_dir: Base simulation directory
        bot_name: Name of the bot (e.g., "Bot_BT", "Bot_NN")
        actor_position: "left" or "right" or "both" (which side to analyze)
        output_path: Path to save the output image
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process per matchup (None for all)
    """
    print("=" * 60)
    print(f"Creating phased heatmap for {bot_name} (position: {actor_position})")
    print("=" * 60)

    # Load all data for this bot
    df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs)

    if df_combined.empty:
        print("No data to plot.")
        return

    print(f"Time range: {df_combined['UpdatedAt'].min():.2f} - {df_combined['UpdatedAt'].max():.2f}")

    # Split into phases
    print("\nSplitting into phases...")
    phases = split_into_phases(df_combined, num_phases=3)
    phase_names = ["Early Game", "Mid Game", "Late Game"]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Plot each phase
    for ax, phase_df, phase_name in zip(axes, phases, phase_names):
        print(f"Plotting {phase_name}...")
        plot_phase_heatmap(ax, phase_df, phase_name)

    position_text = f" ({actor_position} side)" if actor_position != "both" else ""
    plt.suptitle(f"Sumobot Arena Heatmap - Phased Analysis: {bot_name}{position_text}\n({len(df_combined):,} total samples)",
                 fontsize=16, y=0.98)
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {output_path}")
    else:
        plt.show()


def create_phased_heatmap_combined(csv_paths, bot_name, output_path=None, chunksize=50000):
    """
    Create a 3-phase heatmap from multiple CSV files combined

    Args:
        csv_paths: List of paths to CSV files
        bot_name: Name of the bot for the title
        output_path: Path to save the output image
        chunksize: Size of chunks for reading large CSV files
    """
    all_data = []

    print(f"Loading {len(csv_paths)} CSV files...")
    for i, csv_path in enumerate(csv_paths):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(csv_paths)} files...")

        df = load_data_chunked(csv_path, chunksize, actor_filter=0)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("No valid data found.")
        return

    print("Combining all data...")
    df_combined = pd.concat(all_data, ignore_index=True)

    print(f"Total samples: {len(df_combined):,}")
    print(f"Time range: {df_combined['UpdatedAt'].min():.2f} - {df_combined['UpdatedAt'].max():.2f}")

    # Split into phases
    print("Splitting into phases...")
    phases = split_into_phases(df_combined, num_phases=3)
    phase_names = ["Early Game", "Mid Game", "Late Game"]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Plot each phase
    for ax, phase_df, phase_name in zip(axes, phases, phase_names):
        print(f"Plotting {phase_name}...")
        plot_phase_heatmap(ax, phase_df, phase_name)

    plt.suptitle(f"Sumobot Arena Heatmap - Phased Analysis: {bot_name}\n({len(csv_paths)} matches, {len(df_combined):,} total samples)",
                 fontsize=16, y=0.98)
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

def get_bot_heatmap_figures(base_dir, bot_name, actor_position="both", chunksize=50000, max_configs=None):
    """
    Generate matplotlib figures for bot heatmaps (for use in Streamlit/web display)

    Args:
        base_dir: Base simulation directory
        bot_name: Name of the bot (e.g., "Bot_BT", "Bot_NN")
        actor_position: "left", "right", or "both"
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process per matchup

    Returns:
        List of 3 matplotlib figures [early_fig, mid_fig, late_fig]
    """
    print(f"Loading data for {bot_name}...")

    # Load all data for this bot
    df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs)

    if df_combined.empty:
        print(f"No data found for {bot_name}")
        return [None, None, None]

    print(f"Total samples: {len(df_combined):,}")

    # Split into phases
    print("Splitting into phases...")
    phases = split_into_phases(df_combined, num_phases=3)
    phase_names = ["Early Game", "Mid Game", "Late Game"]

    # Create figures for each phase
    figures = []
    for phase_df, phase_name in zip(phases, phase_names):
        if phase_df.empty:
            figures.append(None)
            continue

        # Create single figure for this phase
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plot_phase_heatmap(ax, phase_df, phase_name)

        position_text = f" ({actor_position} side)" if actor_position != "both" else ""
        plt.suptitle(f"{bot_name}{position_text} - {phase_name}\n({len(phase_df):,} samples)",
                    fontsize=16, y=0.98)
        plt.tight_layout()

        figures.append(fig)

    return figures


def create_phased_heatmaps_all_bots(base_dir, output_dir="arena_heatmap", actor_position="both", chunksize=50000, max_configs=None, mode="all"):
    """
    Create phased heatmaps and position distribution plots for all bots in the simulation directory
    Saves individual phase images for each bot

    Args:
        base_dir: Base simulation directory
        output_dir: Output directory for heatmaps (default: "arena_heatmap")
        actor_position: "left", "right", or "both"
        chunksize: Chunk size for reading CSV files
        max_configs: Maximum number of configs to process per matchup
        mode: What to generate - "heatmap", "position", or "all" (default: "all")
    """
    # Find all unique bot names from matchup folders
    matchup_folders = [f for f in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, f)) and "_vs_" in f]

    bot_names = set()
    for matchup in matchup_folders:
        parts = matchup.split("_vs_")
        if len(parts) == 2:
            bot_names.add(parts[0])
            bot_names.add(parts[1])

    bot_names = sorted(bot_names)
    print(f"Found {len(bot_names)} unique bots: {bot_names}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each bot
    for bot_name in bot_names:
        print("\n" + "=" * 60)
        print(f"Processing {bot_name}")
        print("=" * 60)

        # Create bot-specific directory
        bot_dir = os.path.join(output_dir, bot_name)
        os.makedirs(bot_dir, exist_ok=True)

        # Load all data for this bot
        df_combined = load_bot_data_from_simulation(base_dir, bot_name, actor_position, chunksize, max_configs)

        if df_combined.empty:
            print(f"No data found for {bot_name}, skipping...")
            continue

        print(f"Time range: {df_combined['UpdatedAt'].min():.2f} - {df_combined['UpdatedAt'].max():.2f}")

        # Generate heatmaps if requested
        if mode in ["heatmap", "all"]:
            # Split into phases
            print("\nSplitting into phases...")
            phases = split_into_phases(df_combined, num_phases=3)
            phase_names = ["Early Game", "Mid Game", "Late Game"]

            # Create and save individual heatmaps for each phase
            for idx, (phase_df, phase_name) in enumerate(zip(phases, phase_names)):
                print(f"Creating {phase_name} heatmap...")

                if phase_df.empty:
                    print(f"  No data for {phase_name}, skipping...")
                    continue

                # Create single figure for this phase
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                plot_phase_heatmap(ax, phase_df, phase_name)

                position_text = f" ({actor_position} side)" if actor_position != "both" else ""
                plt.suptitle(f"{bot_name}{position_text} - {phase_name}\n({len(phase_df):,} samples)",
                            fontsize=16, y=0.98)
                plt.tight_layout()

                # Save
                output_path = os.path.join(bot_dir, f"{idx}.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"  Saved to {output_path}")
                plt.close(fig)

        # Generate position distribution if requested
        if mode in ["position", "all"]:
            # Create position distribution plot
            print(f"Creating position distribution plot...")
            fig_dist = plot_position_distribution(df_combined, bot_name, actor_position)

            if fig_dist is not None:
                dist_path = os.path.join(bot_dir, "position_distribution.png")
                fig_dist.savefig(dist_path, dpi=150, bbox_inches='tight')
                print(f"  Saved to {dist_path}")
                plt.close(fig_dist)

    print("\n" + "=" * 60)
    print(f"✅ Completed! All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    default_base_dir = "/Users/defdef/Library/Application Support/DefaultCompany/Sumobot/Simulation"
    parser = argparse.ArgumentParser(
        description="Create phased heatmap visualizations for sumobot arena data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single CSV file
  python detailed_analyzer.py single game_log.csv -o output.png

  # Single bot analysis
  python detailed_analyzer.py bot Bot_BT

  # Generate ALL visualizations for all bots (heatmaps + position plots)
  python detailed_analyzer.py all

  # Generate only heatmaps for all bots
  python detailed_analyzer.py all heatmap

  # Generate only position distribution plots for all bots
  python detailed_analyzer.py all position

  # All visualizations with custom path and limited configs
  python detailed_analyzer.py all all "/custom/path" --max-configs 10
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis mode")

    # Single file mode
    single_parser = subparsers.add_parser("single", help="Analyze a single CSV file")
    single_parser.add_argument("csv_path", help="Path to CSV file")
    single_parser.add_argument("-o", "--output", help="Output path for the image")
    single_parser.add_argument("-c", "--chunksize", type=int, default=50000,
                              help="Chunk size for reading CSV (default: 50000)")

    # Bot analysis mode
    bot_parser = subparsers.add_parser("bot", help="Analyze a specific bot from simulation directory")
    bot_parser.add_argument("bot_name", help="Bot name (e.g., Bot_BT, Bot_NN, Bot_Primitive)")
    bot_parser.add_argument("base_dir", nargs='?', default=default_base_dir,
                           help=f"Base simulation directory (default: {default_base_dir})")
    bot_parser.add_argument("-o", "--output", help="Output path for the image")
    bot_parser.add_argument("-p", "--position", choices=["left", "right", "both"], default="both",
                           help="Analyze bot when on left side, right side, or both (default: both)")
    bot_parser.add_argument("-c", "--chunksize", type=int, default=50000,
                           help="Chunk size for reading CSV files (default: 50000)")
    bot_parser.add_argument("--max-configs", type=int,
                           help="Maximum number of config folders to process per matchup (for testing)")

    # All bots mode
    all_parser = subparsers.add_parser("all", help="Generate visualizations for all bots")
    all_parser.add_argument("mode", nargs='?', default="all", choices=["heatmap", "position", "all"],
                           help="What to generate: 'heatmap' (arena heatmaps), 'position' (position distribution plots), or 'all' (both) (default: all)")
    all_parser.add_argument("base_dir", nargs='?', default=default_base_dir,
                           help=f"Base simulation directory (default: {default_base_dir})")
    all_parser.add_argument("-o", "--output", default="arena_heatmap",
                           help="Output directory for visualizations (default: arena_heatmap)")
    all_parser.add_argument("-p", "--position", choices=["left", "right", "both"], default="both",
                           help="Analyze bot when on left side, right side, or both (default: both)")
    all_parser.add_argument("-c", "--chunksize", type=int, default=50000,
                           help="Chunk size for reading CSV files (default: 50000)")
    all_parser.add_argument("--max-configs", type=int,
                           help="Maximum number of config folders to process per matchup (for testing)")

    args = parser.parse_args()

    if args.command == "single":
        create_phased_heatmap(args.csv_path, args.output, args.chunksize)

    elif args.command == "bot":
        output = args.output or f"phased_heatmap_{args.bot_name}_{args.position}.png"
        create_phased_heatmap_for_bot(
            args.base_dir,
            args.bot_name,
            args.position,
            output,
            args.chunksize,
            args.max_configs
        )

    elif args.command == "all":
        create_phased_heatmaps_all_bots(
            args.base_dir,
            args.output,
            args.position,
            args.chunksize,
            args.max_configs,
            args.mode
        )

    else:
        parser.print_help()
