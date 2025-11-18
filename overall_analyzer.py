import numpy as np
import io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import plotly.io as pio
import streamlit as st

# =====================
# Bot Marker Configuration
# =====================
# Map bot names to matplotlib marker shapes for easy visual differentiation
BOT_MARKER_MAP = {
    "Bot_BT": "o",           # Circle
    "Bot_NN": "s",           # Square
    "Bot_Primitive": "^",    # Triangle up
    "Bot_MCTS": "D",         # Diamond
    "Bot_RL": "v",           # Triangle down
    "Bot_Hybrid": "P",       # Plus (filled)
    "Bot_Random": "*",       # Star
    "Bot_Aggressive": "X",   # X
    "Bot_Defensive": "p",    # Pentagon
    "Bot_Custom": "h",       # Hexagon
}

# Default marker if bot not in map
DEFAULT_MARKER = "o"


def get_bot_marker(bot_name):
    """
    Get marker shape for a given bot name.

    Args:
        bot_name: Name of the bot

    Returns:
        Matplotlib marker string
    """
    return BOT_MARKER_MAP.get(bot_name, DEFAULT_MARKER)


def plot_with_bot_markers(ax, data, x, y, hue, hue_order=None, **kwargs):
    """
    Plot line plot with bot-specific markers.

    Args:
        ax: Matplotlib axes object
        data: pandas DataFrame with plot data
        x: Column name for x-axis
        y: Column name for y-axis
        hue: Column name for grouping (bot names or bot names with rank)
        hue_order: List specifying order of hue values (optional)
        **kwargs: Additional plot keywords (linewidth, alpha, etc.)

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_with_bot_markers(ax, data=df, x="Timer", y="WinRate",
        ...                       hue="BotWithRank", hue_order=bot_order)
    """
    # Default plot settings
    plot_kwargs = {'linewidth': 2, 'markersize': 8}
    plot_kwargs.update(kwargs)

    # Determine which bots to plot
    bots_to_plot = hue_order if hue_order else data[hue].unique()

    for bot_label in bots_to_plot:
        bot_data = data[data[hue] == bot_label]
        if bot_data.empty:
            continue

        # Extract original bot name (before " (#rank)" if present)
        bot_name = bot_label.split(" (")[0] if " (" in str(bot_label) else str(bot_label)
        marker = get_bot_marker(bot_name)

        ax.plot(bot_data[x], bot_data[y], marker=marker, label=bot_label, **plot_kwargs)


def update_bot_marker_map(new_mappings):
    """
    Update the bot marker map with new mappings.

    Args:
        new_mappings: Dictionary of {bot_name: marker_shape}

    Example:
        >>> update_bot_marker_map({"Bot_NewBot": "H"})
    """
    BOT_MARKER_MAP.update(new_mappings)


def get_bot_winrates(summary: pd.DataFrame, bot_name: str):
    """Return aggregated winrates for one bot against all others."""
    left = (
        summary[summary["Bot_L"] == bot_name]
        .groupby("Bot_R")["WinRate_L"]
        .mean()
        .reset_index()
        .rename(columns={"Bot_R": "Enemy", "WinRate_L": "WinRate"})
    )

    right = (
        summary[summary["Bot_R"] == bot_name]
        .groupby("Bot_L")["WinRate_R"]
        .mean()
        .reset_index()
        .rename(columns={"Bot_L": "Enemy", "WinRate_R": "WinRate"})
    )

    combined = pd.concat([left, right])
    final = combined.groupby("Enemy")["WinRate"].mean().reset_index()

    return final.sort_values("WinRate", ascending=False)

def build_winrate_matrix(summary: pd.DataFrame):
    """Return pivot matrix of win rates (row = bot, col = enemy)."""
    # Combine both directions
    left = summary[["Bot_L", "Bot_R", "WinRate_L","Rank_L"]].rename(
        columns={"Bot_L": "Left_Side", "Bot_R": "Right_Side", "WinRate_L": "WinRate","Rank_L":"Rank"}
    )
    right = summary[["Bot_R", "Bot_L", "WinRate_R","Rank_R"]].rename(
        columns={"Bot_R": "Left_Side", "Bot_L": "Right_Side", "WinRate_R": "WinRate","Rank_L":"Rank"}
    )

    combined = pd.concat([left, right], ignore_index=True)

    # --- Get bot rank mapping ---
    rank_map = (
        combined.groupby("Left_Side")["Rank"]
        .mean()
        .sort_values()
        .round(0)
        .astype(int)
        .to_dict()
    )

    # --- Rename bot labels with rank ---
    combined["BotWithRankLeft"] = combined["Left_Side"].map(
        lambda b: f"{b} (#{rank_map.get(b, '?')})"
    )
    combined["BotWithRankRight"] = combined["Right_Side"].map(
        lambda b: f"{b} (#{rank_map.get(b, '?')})"
    )

    # Aggregate mean winrate over all configs
    matrix_df = combined.groupby(["BotWithRankLeft", "BotWithRankRight"])["WinRate"].mean().reset_index()

    # Pivot into matrix
    pivot = matrix_df.pivot(index="BotWithRankLeft", columns="BotWithRankRight", values="WinRate")

    # Fill missing (never faced) with NaN
    return pivot

@st.cache_data
def plot_winrate_matrix(summary, width=8, height=6):
    fig = plt.figure(figsize=(width, height))
    pivot = build_winrate_matrix(summary)
    sns.heatmap(
        pivot, annot=True, cmap="Blues", center=0.5,
        fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Win Rate'}
    )
    plt.title("Bot vs Bot Win Rate Matrix")
    plt.ylabel("Bot")
    plt.xlabel("Enemy Bot")
    plt.tight_layout()
    return fig

@st.cache_data
def plot_time_related(summary, width=8, height=6):
    figs = []
    # group by ActInterval, Timer, and Bot_L to average duration per bot per timer
    grouped = (
        summary.groupby(["ActInterval", "Timer", "Bot_L","Rank_L"], as_index=False)
        .agg({"MatchDur": "mean"})
    )
    grouped["AvgDuration"] = grouped["MatchDur"]

    rank_map = (
        grouped.groupby("Bot_L")["Rank_L"]
        .mean()
        .sort_values()
        .round(0)
        .astype(int)
        .to_dict()
    )

    # --- Rename bot labels with rank ---
    grouped["BotWithRank"] = grouped["Bot_L"].map(
        lambda b: f"{b} (#{rank_map.get(b, '?')})"
    )

    for interval in grouped["ActInterval"].unique():
        fig, ax = plt.subplots(figsize=(width, height))
        subset = grouped[grouped["ActInterval"] == interval]

        for bot in subset["Bot_L"].unique():
            bot_data = subset[subset["Bot_L"] == bot]
            label = bot_data["BotWithRank"].iloc[0]
            marker = get_bot_marker(bot)
            ax.plot(bot_data["Timer"], bot_data["AvgDuration"], marker=marker, label=label, markersize=8)

        ax.set_title(f"Avg Match Duration vs Timer (ActInterval = {interval})")
        ax.set_xlabel("Timer (s)")
        ax.set_ylabel("Actual Match Duration (s)")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        ax.grid(True, linestyle="--", alpha=0.5)

        unique_timers = sorted(subset["Timer"].unique())
        ax.set_xticks(unique_timers)
        ax.set_xticklabels([f"{t}s" for t in unique_timers])

        figs.append(fig)
    return figs

@st.cache_data
def plot_action_win_related(summary, width=8, height=6):
    # Step 1: Compute average actions per game (as before)
    summary["AvgActions_L"] = summary["ActionCounts_L"] / summary["Games"]
    summary["AvgActions_R"] = summary["ActionCounts_R"] / summary["Games"]

    # Step 2: Convert each matchup into per-bot rows
    left = summary[["Bot_L", "Bot_R", "AvgActions_L", "WinRate_L"]].rename(
        columns={"Bot_L": "Bot", "Bot_R": "Enemy", "AvgActions_L": "Actions", "WinRate_L": "WinRate"}
    )
    right = summary[["Bot_R", "Bot_L", "AvgActions_R", "WinRate_R"]].rename(
        columns={"Bot_R": "Bot", "Bot_L": "Enemy", "AvgActions_R": "Actions", "WinRate_R": "WinRate"}
    )
    combined = pd.concat([left, right], ignore_index=True)

    corr = combined["Actions"].corr(combined["WinRate"])
    print(f"Correlation between Actions and Win Rate: {corr:.3f}")

    fig = plt.figure(figsize=(width,height))
    sns.regplot(data=combined, x="Actions", y="WinRate", scatter_kws={"alpha":0.6})
    plt.title("Correlation Between Actions Taken and Win Rate")
    plt.xlabel("Average Actions per Game")
    plt.ylabel("Win Rate")
    plt.grid(alpha=0.3)

    plt.text(
        0.6, 0.85,
        f"Correlation result: {corr}.\n"
        "> 0.5 → strong positive relationship (more actions → more wins)\n~0.0 → no clear relation\n< -0.5 → inverse relationship (passive bots win more)",
        transform=plt.gca().transAxes,
        fontsize=6,
        bbox=dict(facecolor='lightyellow', alpha=0.5, edgecolor='gold', boxstyle="round,pad=0.4")
    )
    
    return fig

@st.cache_data
def plot_highest_action(summary, width=8, height=6, n_action = 6):
    action_cols = [col for col in summary.columns if col.endswith("_Act_L")]

    # Get rank mapping if available
    if "Rank_L" in summary.columns:
        rank_map = summary.groupby("Bot_L")["Rank_L"].first().to_dict()
        bot_order = sorted(rank_map.keys(), key=lambda b: rank_map[b])
    else:
        rank_map = {}
        bot_order = sorted(summary["Bot_L"].unique())

    df_actions = summary.melt(
        id_vars=["Bot_L"],
        value_vars=action_cols,
        var_name="Action",
        value_name="Count"
    )
    df_actions["Action"] = df_actions["Action"].str.replace("_Act_L", "")
    df_actions = df_actions.groupby(["Bot_L", "Action"])["Count"].sum().reset_index()

    # Add rank to bot names if available
    if rank_map:
        df_actions["BotWithRank"] = df_actions["Bot_L"].map(lambda b: f"{b} (#{int(rank_map.get(b, 999))})")
        bot_order_with_rank = [f"{b} (#{int(rank_map[b])})" for b in bot_order]
        hue_col = "BotWithRank"
        hue_order = bot_order_with_rank
    else:
        hue_col = "Bot_L"
        hue_order = bot_order

    top_actions = df_actions.groupby("Bot_L").apply(lambda x: x.nlargest(n_action, "Count")).reset_index(drop=True)

    # Re-add BotWithRank for top_actions
    if rank_map:
        top_actions["BotWithRank"] = top_actions["Bot_L"].map(lambda b: f"{b} (#{int(rank_map.get(b, 999))})")

    fig = plt.figure(figsize=(width,height))
    sns.barplot(
        data=top_actions,
        x="Count",
        y="Action",
        hue=hue_col,
        hue_order=hue_order
    )
    plt.title("Top 3 Actions Taken per Bot")
    plt.xlabel("Action Count")
    plt.ylabel("Action")
    legend_title = "Bot (Rank)" if rank_map else "Bot"
    plt.legend(title=legend_title, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.tight_layout()
    return fig

@st.cache_data
def plot_win_rate_stability_over_timer(summary, width=8, height=6):
    # Melt the WinRate columns so both sides are in one column
    df_melted = summary.melt(
        id_vars=["Bot_L", "Bot_R", "Timer"],
        value_vars=["WinRate_L", "WinRate_R"],
        var_name="Side",
        value_name="WinRate"
    )

    # Extract bot name depending on side
    df_melted["Bot"] = df_melted.apply(
        lambda r: r["Bot_L"] if r["Side"] == "WinRate_L" else r["Bot_R"],
        axis=1
    )
    avg = df_melted.groupby(["Bot", "Timer"])["WinRate"].mean().reset_index()

    # Plot
    heat = avg.pivot(index="Bot", columns="Timer", values="WinRate")

    fig = plt.figure(figsize=(width, height))
    sns.heatmap(heat, annot=True, cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Win Rate Stability vs Timer (Heatmap)")
    plt.xlabel("Timer")
    plt.ylabel("Bot")
    return fig

def plot_timebins_intensity(
    df,
    group_by="Bot",
    timer=None,
    act_interval=None,
    round=None,
    mode="total",        # "total" | "per_action" | "select"
    action_name=None,    # used when mode == "select"
    width=10,
    height=6,
    summary_df=None,     # Optional: summary dataframe with Rank_L column for bot ranking
):
    """
    Plot action intensity over time (with optional timer cutoff).
    Modes:
      - "total": sum MeanCount across all actions -> one line per bot
      - "per_action": show per-action trends; creates one subplot per action
      - "select": plot a single action_name for all bots
    """

    # --- Filters ---
    if timer is not None:
        df = df[df["Timer"] == timer]
    if act_interval is not None:
        df = df[df["ActInterval"] == act_interval]
    if round is not None:
        df = df[df["Round"] == round]

    if df.empty:
        print("⚠️ No data after filtering.")
        return None

    # --- Preprocess TimeBin ---
    df = df.copy()
    df["TimeBin"] = pd.to_numeric(df["TimeBin"], errors="coerce")
    df = df.dropna(subset=["TimeBin"])
    df = df.sort_values("TimeBin")

    # --- Add rank to bot names if group_by is "Bot" ---
    rank_map = None
    if group_by == "Bot":
        # Try to get rank from summary_df first, then from df itself
        if summary_df is not None and "Rank_L" in summary_df.columns:
            rank_map = summary_df.groupby("Bot_L")["Rank_L"].first().to_dict()
        elif "Rank" in df.columns:
            rank_map = df.groupby("Bot")["Rank"].first().to_dict()
        elif "Rank_L" in df.columns:
            rank_map = df.groupby("Bot")["Rank_L"].first().to_dict()

    if rank_map:
        df["BotWithRank"] = df["Bot"].map(lambda b: f"{b} (#{int(rank_map.get(b, 999))})")
        group_by_plot = "BotWithRank"
        # Sort bots by rank
        bot_order = sorted(rank_map.keys(), key=lambda b: rank_map.get(b, 999))
        bot_order_with_rank = [f"{b} (#{int(rank_map[b])})" for b in bot_order]
    else:
        group_by_plot = group_by
        bot_order_with_rank = None

    # --- Helper: apply x-axis cutoff ---
    def apply_timer_xlim(ax):
        if timer is not None:
            ax.set_xlim(0, timer)
            ax.set_xticks(range(0, int(timer) + 1, max(1, int(timer // 10) or 1)))

    # --- Plot modes ---
    if mode == "select":
        if not action_name:
            raise ValueError("action_name must be provided when mode='select'")
        df_sel = df[df["Action"] == action_name]
        if df_sel.empty:
            print(f"⚠️ No rows for action '{action_name}' after filtering.")
            return None
        grouped = df_sel.groupby([group_by_plot, "TimeBin"], as_index=False)["MeanCount"].mean()

        fig, ax = plt.subplots(figsize=(width, height))
        plot_with_bot_markers(ax, data=grouped, x="TimeBin", y="MeanCount",
                            hue=group_by_plot, hue_order=bot_order_with_rank)
        ax.set_title(f"Mean {action_name} over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Count")
        legend_title = "Bot (Rank)" if (group_by == "Bot" and rank_map is not None) else group_by
        ax.legend(title=legend_title, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        ax.grid(True, alpha=0.3)
        apply_timer_xlim(ax)
        fig.tight_layout()
        return fig

    elif mode == "total":
        grouped = df.groupby([group_by_plot, "TimeBin"], as_index=False)["MeanCount"].mean()
        fig, ax = plt.subplots(figsize=(width, height))
        plot_with_bot_markers(ax, data=grouped, x="TimeBin", y="MeanCount",
                            hue=group_by_plot, hue_order=bot_order_with_rank)
        ax.set_title("Total action intensity over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Count (summed over actions)")
        legend_title = "Bot (Rank)" if (group_by == "Bot" and rank_map is not None) else group_by
        ax.legend(title=legend_title, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        ax.grid(True, alpha=0.3)
        apply_timer_xlim(ax)
        fig.tight_layout()
        return fig

    elif mode == "per_action":
        actions = sorted(df["Action"].unique())
        n = len(actions)
        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(width, max(height, 2.5 * nrows)),
            squeeze=False
        )
        axes = axes.flatten()

        handles, labels = None, None
        for i, action in enumerate(actions):
            ax = axes[i]
            sub = df[df["Action"] == action].groupby([group_by_plot, "TimeBin"], as_index=False)["MeanCount"].mean()
            if sub.empty:
                ax.set_visible(False)
                continue
            plot_with_bot_markers(ax, data=sub, x="TimeBin", y="MeanCount",
                                hue=group_by_plot, hue_order=bot_order_with_rank)

            # Capture legend handles from first plot
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

            ax.set_title(action)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mean Count")
            ax.grid(True, alpha=0.3)
            apply_timer_xlim(ax)

        # Hide unused axes
        for j in range(len(actions), len(axes)):
            axes[j].set_visible(False)

        # Add global legend below
        if handles and labels:
            legend_title = "Bot (Rank)" if (group_by == "Bot" and rank_map is not None) else group_by
            fig.legend(
                handles, labels, title=legend_title,
                loc="upper center", bbox_to_anchor=(0.5, -0.02),
                ncol=min(6, len(labels))
            )
        fig.suptitle("Per-action intensity over timer")
        fig.tight_layout()
        return fig

    else:
        raise ValueError("mode must be one of ['total','per_action','select']")


def plot_full_cross_heatmap_half(df, bot_name="Bot_NN", key="WinRate_L", max_labels=40, lower_triangle=True):
    cfg_cols = ["Timer", "ActInterval", "Round", "SkillLeft", "SkillRight"]
    df_bot = df[df["Bot_L"] == bot_name].copy()
    
    # Melt configurations
    melted = df_bot.melt(
        id_vars=[key],
        value_vars=cfg_cols,
        var_name="ConfigType",
        value_name="ConfigValue"
    )

    # Cartesian join (self merge)
    merged = melted.merge(melted, on=key, suffixes=("_X", "_Y"))
    merged = merged[merged["ConfigType_X"] != merged["ConfigType_Y"]]

    # Aggregate mean WinRate
    grouped = (
        merged.groupby(["ConfigType_X", "ConfigValue_X", "ConfigType_Y", "ConfigValue_Y"])[key]
        .mean()
        .reset_index()
    )

    # Label for axes
    grouped["X"] = grouped["ConfigType_X"] + "=" + grouped["ConfigValue_X"].astype(str)
    grouped["Y"] = grouped["ConfigType_Y"] + "=" + grouped["ConfigValue_Y"].astype(str)

    # Pivot into matrix
    pivot = grouped.pivot(index="Y", columns="X", values=key)

    # Drop all-NaN rows and columns

    # Clip to manageable size
    if len(pivot) > max_labels or len(pivot.columns) > max_labels:
        pivot = pivot.iloc[:max_labels, :max_labels]

    # Ensure symmetry (optional, if slightly different values occur)
    pivot = (pivot + pivot.T) / 2

    pivot = pivot.dropna(axis=0, how="all")
    pivot = pivot.dropna(axis=1, how="all")

    # Build triangular mask
    # mask = np.triu(np.ones_like(pivot, dtype=bool)) if lower_triangle else np.tril(np.ones_like(pivot, dtype=bool))

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*0.4), max(8, len(pivot)*0.3)))
    sns.heatmap(
        pivot,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        # mask=mask,            # ✅ Hide upper (or lower) triangle
        linewidths=0.5,
        cbar_kws={'label': 'Win Rate'},
        ax=ax
    )

    ax.set_title(f"Cross Configuration Win Rate (Half Matrix) for {bot_name}", fontsize=14, pad=12)
    ax.set_xlabel("Config X", fontsize=12)
    ax.set_ylabel("Config Y", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig


def plot_grouped_config_winrates(
    df: pd.DataFrame,
    bot_col: str = "Bot_L",
    metric: str = "WinRate_L",
    config_col: str = "Timer",
    width: int = 10,
    height: int = 6,
    title: str = None,
    ylabel: str = None,
):
    """
    Create a grouped bar chart showing win-rates (or other metrics) grouped by a single configuration parameter.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe (e.g., matchup_summary)
    bot_col : str
        Column name for bots (default: "Bot_L")
    metric : str
        Metric to plot (default: "WinRate_L")
    config_col : str
        Configuration column to group by (default: "Timer")
        Special case: "Skill" will use both SkillLeft and SkillRight
    width : int
        Figure width
    height : int
        Figure height
    title : str
        Plot title (optional)
    ylabel : str
        Y-axis label (optional)

    Returns
    -------
    matplotlib.figure.Figure
    """

    # Get unique bots and sort by rank
    rank_col = "Rank_L" if bot_col == "Bot_L" else "Rank_R"
    if rank_col in df.columns:
        rank_map = df.groupby(bot_col)[rank_col].first().to_dict()
        bots = sorted(df[bot_col].unique(), key=lambda b: rank_map.get(b, 9999))
    else:
        bots = sorted(df[bot_col].unique())
        rank_map = {}

    # Determine if we need to calculate per-game averages
    per_game_metrics = ["Collisions", "ActionCounts", "Duration", "MatchDur"]
    needs_per_game = any(m in metric for m in per_game_metrics)

    # Special handling for "Skill" - use both SkillLeft and SkillRight
    if config_col == "Skill":
        # Merge left and right data
        if needs_per_game:
            # Include Games column for per-game calculation
            left_data = df[[bot_col, "SkillLeft", metric, "Games"]].copy()
            left_data = left_data.rename(columns={"SkillLeft": "SkillType"})

            right_data = df[[bot_col.replace("_L", "_R"), "SkillRight", metric.replace("_L", "_R"), "Games"]].copy()
            right_data = right_data.rename(columns={
                bot_col.replace("_L", "_R"): bot_col,
                "SkillRight": "SkillType",
                metric.replace("_L", "_R"): metric
            })

            combined = pd.concat([left_data, right_data], ignore_index=True)
            # Calculate per-game average first
            combined['metric_per_game'] = combined[metric] / combined['Games']
            grouped = combined.groupby([bot_col, "SkillType"])['metric_per_game'].agg(['mean', 'std']).reset_index()
        else:
            left_data = df[[bot_col, "SkillLeft", metric]].copy()
            left_data = left_data.rename(columns={"SkillLeft": "SkillType"})

            right_data = df[[bot_col.replace("_L", "_R"), "SkillRight", metric.replace("_L", "_R")]].copy()
            right_data = right_data.rename(columns={
                bot_col.replace("_L", "_R"): bot_col,
                "SkillRight": "SkillType",
                metric.replace("_L", "_R"): metric
            })

            combined = pd.concat([left_data, right_data], ignore_index=True)
            grouped = combined.groupby([bot_col, "SkillType"])[metric].agg(['mean', 'std']).reset_index()

        config_values = sorted(grouped["SkillType"].unique())
        config_col_display = "SkillType"
    else:
        # Normal handling for other config columns
        config_values = sorted(df[config_col].unique())

        if needs_per_game:
            # Calculate per-game average first
            df_copy = df.copy()
            df_copy['metric_per_game'] = df_copy[metric] / df_copy['Games']
            grouped = df_copy.groupby([bot_col, config_col])['metric_per_game'].agg(['mean', 'std']).reset_index()
        else:
            grouped = df.groupby([bot_col, config_col])[metric].agg(['mean', 'std']).reset_index()

        config_col_display = config_col

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(width, height))

    # Define colors for each config value
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#9467bd', '#8c564b']
    config_colors = {val: colors[i % len(colors)] for i, val in enumerate(config_values)}

    # Set up bar positions
    n_bots = len(bots)
    n_configs = len(config_values)
    bar_width = 0.8 / n_configs
    x_positions = np.arange(n_bots)

    # Plot bars for each config value
    for i, config_val in enumerate(config_values):
        if config_col == "Skill":
            config_data = grouped[grouped["SkillType"] == config_val]
        else:
            config_data = grouped[grouped[config_col] == config_val]

        means = []
        # stds = []

        for bot in bots:
            bot_data = config_data[config_data[bot_col] == bot]
            if not bot_data.empty:
                means.append(bot_data['mean'].values[0])
                # std_val = bot_data['std'].values[0]
                # stds.append(std_val if not pd.isna(std_val) else 0)
            else:
                means.append(0)
                # stds.append(0)

        offset = (i - n_configs/2 + 0.5) * bar_width
        bars = ax.bar(x_positions + offset, means, bar_width,
               label=str(config_val), color=config_colors[config_val],)
            #    yerr=stds, capsize=3, error_kw={'linewidth': 1.5, 'elinewidth': 1})

        # Add value labels inside bars
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            if height > 0:  # Only add label if bar has height
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{mean_val:.1f}',
                        ha='center', va='center', fontsize=7, fontweight='bold', color='white')

    # Customize plot
    ax.set_xlabel('Bots', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel if ylabel else metric.replace('_', ' '), fontsize=12, fontweight='bold')

    if title:
        plot_title = title
    else:
        display_name = "Skill" if config_col == "Skill" else config_col
        plot_title = f'{metric.replace("_", " ")} grouped by {display_name}'

    ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_positions)
    # Create bot labels with rank
    if rank_map:
        bot_labels = [f"{bot} (#{int(rank_map[bot])})" for bot in bots]
    else:
        bot_labels = bots
    ax.set_xticklabels(bot_labels, rotation=30, ha='right')
    ax.legend(title=config_col_display, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(6, n_configs),
              fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    return fig


def plot_overall_bot_metrics(
    df: pd.DataFrame,
    bot_col: str = "Bot_L",
    metric: str = "Collisions_L",
    width: int = 10,
    height: int = 6,
    title: str = None,
    ylabel: str = None,
):
    """
    Create a simple bar chart showing mean metric values per bot across all configurations.

    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe (e.g., matchup_summary)
    bot_col : str
        Column name for bots (default: "Bot_L")
    metric : str
        Metric to plot. Options:
        - "Collisions_L" or "Collisions_R": Total collisions
        - "ActionCounts_L" or "ActionCounts_R": Total action counts
        - "Duration_L" or "Duration_R": Action duration
        - "MatchDur": Match duration
        - "Games": Total games
    width : int
        Figure width
    height : int
        Figure height
    title : str
        Plot title (optional)
    ylabel : str
        Y-axis label (optional)

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_overall_bot_metrics(df, metric="Collisions_L", title="Mean Collisions per Bot")
    >>> plot_overall_bot_metrics(df, metric="ActionCounts_L", title="Mean Actions per Bot")
    >>> plot_overall_bot_metrics(df, metric="MatchDur", title="Mean Match Duration per Bot")
    """

    # Get unique bots and sort by rank
    rank_col = "Rank_L" if bot_col == "Bot_L" else "Rank_R"
    if rank_col in df.columns:
        rank_map = df.groupby(bot_col)[rank_col].first().to_dict()
        bots = sorted(df[bot_col].unique(), key=lambda b: rank_map.get(b, 9999))
    else:
        bots = sorted(df[bot_col].unique())
        rank_map = {}

    # Determine if we need to calculate per-game averages
    per_game_metrics = ["Collisions", "ActionCounts", "Duration", "MatchDur"]
    needs_per_game = any(m in metric for m in per_game_metrics)

    if needs_per_game:
        # Calculate per-game average
        df_copy = df.copy()
        df_copy['metric_per_game'] = df_copy[metric] / df_copy['Games']
        grouped = df_copy.groupby(bot_col)['metric_per_game'].mean().reset_index()
        grouped.columns = [bot_col, 'mean_value']
    else:
        # Direct mean for metrics like winrate
        grouped = df.groupby(bot_col)[metric].mean().reset_index()
        grouped.columns = [bot_col, 'mean_value']

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(width, height))

    # Prepare data for plotting
    means = []
    for bot in bots:
        bot_data = grouped[grouped[bot_col] == bot]
        if not bot_data.empty:
            means.append(bot_data['mean_value'].values[0])
        else:
            means.append(0)

    # Plot bars
    x_positions = np.arange(len(bots))
    bars = ax.bar(x_positions, means, width=0.6, color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels inside bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{mean_val:.1f}',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Customize plot
    ax.set_xlabel('Bots', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel if ylabel else metric.replace('_', ' '), fontsize=12, fontweight='bold')

    if title:
        plot_title = title
    else:
        plot_title = f'Mean {metric.replace("_", " ")} per Bot (across all configurations)'

    ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_positions)

    # Create bot labels with rank
    if rank_map:
        bot_labels = [f"{bot} (#{int(rank_map[bot])})" for bot in bots]
    else:
        bot_labels = bots
    ax.set_xticklabels(bot_labels, rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    return fig


def plot_action_radar(df, bot_col="Bot_L", width=14, height=12, scale=None, radial_limit="auto"):
    """
    Create a radar chart showing mean action counts per bot.

    Parameters:
        scale (str): Scale for radial axis (None = linear, "sqrt", "log")
        radial_limit (str or float):
                    - "auto": Set max based on 95th percentile (recommended for better spacing)
                    - "max": Use the absolute max value
                    - float: Manually set the max radial value
    """
    # Get all action columns
    action_cols = [col for col in df.columns if col.endswith("_Act_L")]

    # Get unique bots and sort by rank
    rank_col = "Rank_L" if bot_col == "Bot_L" else "Rank_R"
    if rank_col in df.columns:
        rank_map = df.groupby(bot_col)[rank_col].first().to_dict()
        bots = sorted(df[bot_col].unique(), key=lambda b: rank_map.get(b, 9999))
    else:
        bots = sorted(df[bot_col].unique())
        rank_map = {}

    # Calculate mean action counts per bot (raw values)
    # Merge SkillBoost and SkillStone into single "Skill"
    bot_data_raw = {}
    action_names = []

    for bot in bots:
        bot_df = df[df[bot_col] == bot]
        means = []

        # Build action list on first iteration
        if not action_names:
            for col in action_cols:
                name = col.replace("_Act_L", "")

                # Skip SkillStone (will be merged with SkillBoost)
                if name == "SkillStone":
                    continue

                # Rename SkillBoost to Skill
                if name == "SkillBoost":
                    action_names.append("Skill")
                else:
                    action_names.append(name)

        # Calculate means with merged skills
        for col in action_cols:
            name = col.replace("_Act_L", "")

            if name == "SkillStone":
                continue  # Skip, already merged

            if name == "SkillBoost":
                # Merge SkillBoost + SkillStone
                skill_boost = bot_df[col].mean()
                skill_stone = bot_df.get("SkillStone_Act_L", bot_df[col] * 0).mean()  # Handle if column doesn't exist
                means.append(skill_boost + skill_stone)
            else:
                means.append(bot_df[col].mean())

        bot_data_raw[bot] = means

    # Transform values based on scale
    bot_data = {}
    for bot, values in bot_data_raw.items():
        if scale == "sqrt":
            bot_data[bot] = [np.sqrt(v) for v in values]
        elif scale == "log":
            bot_data[bot] = [np.log10(v + 1) for v in values]  # +1 to handle zeros
        else:  # linear / None
            bot_data[bot] = values

    # Set up radar chart
    num_vars = len(action_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(width, height), subplot_kw=dict(projection='polar'))

    # Plot each bot
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#9467bd', '#8c564b']
    for i, (bot, values) in enumerate(bot_data.items()):
        values += values[:1]  # Complete the circle
        # Format label with rank if available
        if rank_map and bot in rank_map:
            label = f"{bot} (#{int(rank_map[bot])})"
        else:
            label = bot
        # Use bot-specific marker
        marker = get_bot_marker(bot)
        ax.plot(angles, values, f'{marker}-', linewidth=2.5, markersize=8,
                label=label, color=colors[i % len(colors)])

    # Set labels with better styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(action_names, fontsize=11)

    # Set radial limit for better spacing
    all_values = [v for bot_vals in bot_data.values() for v in bot_vals[:-1]]
    if radial_limit == "auto":
        max_val = np.percentile(all_values, 95)  # Use 95th percentile
        ax.set_ylim(0, max_val * 1.15)  # Add 15% padding
    elif radial_limit == "max":
        max_val = max(all_values)
        ax.set_ylim(0, max_val * 1.1)
    elif isinstance(radial_limit, (int, float)):
        ax.set_ylim(0, radial_limit)

    # Add more radial grid lines for better readability
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    ax.tick_params(axis='y', labelsize=9)

    # Set y-axis label based on scale
    if scale == "sqrt":
        ax.set_ylabel('√(Mean Action Count)', labelpad=35, fontsize=11)
    elif scale == "log":
        ax.set_ylabel('log₁₀(Mean Action Count + 1)', labelpad=35, fontsize=11)
    else:
        ax.set_ylabel('Mean Action Count', labelpad=35, fontsize=11)

    ax.set_title('Actions Behaviour', size=16, pad=20, fontweight='bold')
    legend_title = "Bot (Rank)" if rank_map else "Bot"
    ax.legend(title=legend_title, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=10, framealpha=0.9, ncol=3)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    return fig


def plot_collision_radar(df, bot_col="Bot_L", width=14, height=12, scale=None):
    """
    Create a triangular radar chart showing collision outcomes per bot.
    Three axes: hit (wins), tie (draws), being_hit (losses)

    Parameters:
        scale (str): Scale for radial axis. Options:
                    - "linear": Raw values
                    - "sqrt": Square root scale (recommended - shows all values clearly)
                    - "log": Logarithmic scale (more aggressive compression)
    """
    # Get unique bots and sort by rank
    rank_col = "Rank_L" if bot_col == "Bot_L" else "Rank_R"
    if rank_col in df.columns:
        rank_map = df.groupby(bot_col)[rank_col].first().to_dict()
        bots = sorted(df[bot_col].unique(), key=lambda b: rank_map.get(b, 9999))
    else:
        bots = sorted(df[bot_col].unique())
        rank_map = {}

    # Calculate collision statistics per bot (raw values)
    bot_data_raw = {}
    for bot in bots:
        # Get data for this bot on left side
        left_df = df[df[bot_col] == bot]

        # Calculate totals (raw counts)
        hit = left_df["Collisions_L"].sum()
        being_hit = left_df["Collisions_R"].sum()
        ties = left_df["Collisions_Tie"].sum()

        # Store as a list: [hit, tie, being_hit]
        bot_data_raw[bot] = [hit, ties, being_hit]

    # Transform values based on scale
    bot_data = {}
    for bot, values in bot_data_raw.items():
        if scale == "sqrt":
            bot_data[bot] = [np.sqrt(v) for v in values]
        elif scale == "log":
            bot_data[bot] = [np.log10(v + 1) for v in values]  # +1 to handle zeros
        else:  # linear
            bot_data[bot] = values

    # Set up triangular radar chart (3 vertices)
    collision_types = ['hit', 'tie', 'being hit']
    num_vars = len(collision_types)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(width, height), subplot_kw=dict(projection='polar'))

    # Plot each bot
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#9467bd', '#8c564b']
    for i, (bot, values) in enumerate(bot_data.items()):
        values += values[:1]  # Complete the circle
        # Format label with rank if available
        if rank_map and bot in rank_map:
            label = f"{bot} (#{int(rank_map[bot])})"
        else:
            label = bot
        # Use bot-specific marker
        marker = get_bot_marker(bot)
        ax.plot(angles, values, f'{marker}-', linewidth=2.5, markersize=8,
                label=label, color=colors[i % len(colors)])

    # Set labels with better styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(collision_types, fontsize=11)

    # Add more radial grid lines for better readability
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    ax.tick_params(axis='y', labelsize=9)

    # Set y-axis label based on scale
    if scale == "sqrt":
        ax.set_ylabel('√(Collision Count)', labelpad=35, fontsize=11)
    elif scale == "log":
        ax.set_ylabel('log₁₀(Collision Count + 1)', labelpad=35, fontsize=11)
    else:
        ax.set_ylabel('Collision Count', labelpad=35, fontsize=11)

    ax.set_title('Collision Behaviour', size=16, pad=20, fontweight='bold')
    legend_title = "Bot (Rank)" if rank_map else "Bot"
    ax.legend(title=legend_title, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=10, framealpha=0.9, ncol=3)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    return fig


def plot_action_distribution_stacked(df, bot_col="Bot_L", width=10, height=6, normalize=False):
    """
    Create a stacked bar chart showing action type distribution per bot.

    Parameters:
        df: Summary dataframe with action counts
        bot_col: Column name for bots (default: "Bot_L")
        width: Figure width
        height: Figure height
        normalize: If True, normalize bars to 100% (show proportions)
                   If False, show absolute counts

    Returns:
        matplotlib.figure.Figure
    """
    # Define action columns (merge SkillBoost and SkillStone into "Skill")
    action_mapping = {
        'Accelerate': 'Accelerate_Act_L',
        'TurnLeft': 'TurnLeft_Act_L',
        'TurnRight': 'TurnRight_Act_L',
        'Dash': 'Dash_Act_L',
        'Skill': ['SkillBoost_Act_L', 'SkillStone_Act_L']
    }

    # Get unique bots and sort by rank
    rank_col = "Rank_L" if bot_col == "Bot_L" else "Rank_R"
    if rank_col in df.columns:
        rank_map = df.groupby(bot_col)[rank_col].first().to_dict()
        bots = sorted(df[bot_col].unique(), key=lambda b: rank_map.get(b, 9999))
    else:
        bots = sorted(df[bot_col].unique())
        rank_map = {}

    # Prepare data for stacking
    action_data = {action: [] for action in action_mapping.keys()}

    for bot in bots:
        bot_df = df[df[bot_col] == bot]

        for action_name, col_names in action_mapping.items():
            if isinstance(col_names, list):
                # Merge multiple columns (for Skill)
                total = sum(bot_df[col].sum() for col in col_names if col in bot_df.columns)
            else:
                # Single column
                total = bot_df[col_names].sum() if col_names in bot_df.columns else 0

            action_data[action_name].append(total)

    # Convert to DataFrame for easier plotting
    data_df = pd.DataFrame(action_data, index=bots)

    # Normalize if requested
    if normalize:
        data_df = data_df.div(data_df.sum(axis=1), axis=0) * 100

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(width, height))

    # Define colors for each action type
    colors = {
        'Accelerate': '#d62728',    # Red
        'TurnLeft': '#ff7f0e',      # Orange
        'TurnRight': '#2ca02c',     # Green
        'Dash': '#17becf',          # Cyan
        'Skill': '#1f77b4'          # Blue
    }

    # Create bot labels with rank
    if rank_map:
        bot_labels = [f"{bot} (#{int(rank_map[bot])})" for bot in bots]
    else:
        bot_labels = bots

    # Plot stacked bars
    bottom = np.zeros(len(bots))
    x_pos = np.arange(len(bots))
    for action in action_mapping.keys():
        ax.bar(x_pos, data_df[action], bottom=bottom,
               label=action, color=colors[action], width=0.6)
        bottom += data_df[action]

    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bot_labels)
    ax.set_xlabel('Bots', fontsize=12, fontweight='bold')

    if normalize:
        ax.set_ylabel('Action Distribution (%)', fontsize=12, fontweight='bold')
        ax.set_title('Action Type Distribution per Bot (Normalized)',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 100)
    else:
        ax.set_ylabel('Total Action Count', fontsize=12, fontweight='bold')
        ax.set_title('Action Type Distribution per Bot',
                     fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5,
              fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Rotate x-axis labels if many bots
    if len(bots) > 5:
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    fig.tight_layout()
    return fig


def plot_collision_distribution_stacked(df, bot_col="Bot_L", width=10, height=6, normalize=False):
    """
    Create a stacked bar chart showing collision type distribution per bot.

    Parameters:
        df: Summary dataframe with collision counts
        bot_col: Column name for bots (default: "Bot_L")
        width: Figure width
        height: Figure height
        normalize: If True, normalize bars to 100% (show proportions)
                   If False, show absolute counts

    Returns:
        matplotlib.figure.Figure
    """
    # Get unique bots and sort by rank
    rank_col = "Rank_L" if bot_col == "Bot_L" else "Rank_R"
    if rank_col in df.columns:
        rank_map = df.groupby(bot_col)[rank_col].first().to_dict()
        bots = sorted(df[bot_col].unique(), key=lambda b: rank_map.get(b, 9999))
    else:
        bots = sorted(df[bot_col].unique())
        rank_map = {}

    # Prepare data for stacking
    collision_data = {
        'Hit': [],           # Collisions won (Collisions_L)
        'Being Hit': [],     # Collisions lost (Collisions_R)
        'Tie': []            # Tie collisions
    }

    for bot in bots:
        bot_df = df[df[bot_col] == bot]

        # Calculate totals
        hit = bot_df['Collisions_L'].sum() if 'Collisions_L' in bot_df.columns else 0
        being_hit = bot_df['Collisions_R'].sum() if 'Collisions_R' in bot_df.columns else 0
        tie = bot_df['Collisions_Tie'].sum() if 'Collisions_Tie' in bot_df.columns else 0

        collision_data['Hit'].append(hit)
        collision_data['Being Hit'].append(being_hit)
        collision_data['Tie'].append(tie)

    # Convert to DataFrame for easier plotting
    data_df = pd.DataFrame(collision_data, index=bots)

    # Normalize if requested
    if normalize:
        data_df = data_df.div(data_df.sum(axis=1), axis=0) * 100

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(width, height))

    # Define colors for each collision type
    colors = {
        'Hit': '#2ca02c',         # Green (wins)
        'Being Hit': '#d62728',   # Red (losses)
        'Tie': '#ff7f0e'          # Orange (ties)
    }

    # Create bot labels with rank
    if rank_map:
        bot_labels = [f"{bot} (#{int(rank_map[bot])})" for bot in bots]
    else:
        bot_labels = bots

    # Plot stacked bars
    bottom = np.zeros(len(bots))
    x_pos = np.arange(len(bots))
    for collision_type in collision_data.keys():
        ax.bar(x_pos, data_df[collision_type], bottom=bottom,
               label=collision_type, color=colors[collision_type], width=0.6)
        bottom += data_df[collision_type]

    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bot_labels)
    ax.set_xlabel('Bots', fontsize=12, fontweight='bold')

    if normalize:
        ax.set_ylabel('Collision Distribution (%)', fontsize=12, fontweight='bold')
        ax.set_title('Collision Type Distribution per Bot (Normalized)',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 100)
    else:
        ax.set_ylabel('Total Collision Count', fontsize=12, fontweight='bold')
        ax.set_title('Collision Type Distribution per Bot',
                     fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
              fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Rotate x-axis labels if many bots
    if len(bots) > 5:
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    fig.tight_layout()
    return fig


def plot_collision_timebins_intensity(
    df,
    group_by="Bot_L",  # "Bot_L" or "Bot_R"
    timer=None,
    act_interval=None,
    round=None,
    mode="total",        # "total" | "per_type" | "select"
    collision_type=None,  # "Actor_L" | "Actor_R" | "Tie" (used when mode == "select")
    width=10,
    height=6,
    summary_df=None,     # Optional: summary dataframe with Rank_L column for bot ranking
):
    """
    Plot collision intensity over time (with optional timer cutoff).

    Modes:
      - "total": sum all collision types -> one line per bot pairing
      - "per_type": show per-collision-type trends; creates one subplot per type (Actor_L, Actor_R, Tie)
      - "select": plot a single collision_type for all bot pairings

    Args:
        df: DataFrame from summary_collision_timebins.csv
        group_by: "Bot_L" or "Bot_R" to group by bot
        timer: Filter by specific timer value
        act_interval: Filter by specific action interval
        round: Filter by specific round
        mode: Visualization mode
        collision_type: Which collision type to show (for mode="select")
        width, height: Figure dimensions
    """

    # --- Filters ---
    if timer is not None:
        df = df[df["Timer"] == timer]
    if act_interval is not None:
        df = df[df["ActInterval"] == act_interval]
    if round is not None:
        df = df[df["Round"] == round]

    if df.empty:
        print("⚠️ No data after filtering.")
        return None

    # --- Preprocess TimeBin ---
    df = df.copy()
    df["TimeBin"] = pd.to_numeric(df["TimeBin"], errors="coerce")
    df = df.dropna(subset=["TimeBin"])
    df = df.sort_values("TimeBin")

    # --- Add rank to bot names if group_by is a bot column ---
    rank_map = None
    if group_by in ["Bot_L", "Bot_R"]:
        # Try to get rank from summary_df first, then from df itself
        if summary_df is not None and "Rank_L" in summary_df.columns:
            rank_map = summary_df.groupby("Bot_L")["Rank_L"].first().to_dict()
        elif "Rank_L" in df.columns:
            rank_map = df.groupby(group_by)["Rank_L"].first().to_dict()

    if rank_map:
        df["BotWithRank"] = df[group_by].map(lambda b: f"{b} (#{int(rank_map.get(b, 999))})")
        group_by_plot = "BotWithRank"
        # Sort bots by rank
        bot_order = sorted(rank_map.keys(), key=lambda b: rank_map.get(b, 999))
        bot_order_with_rank = [f"{b} (#{int(rank_map[b])})" for b in bot_order]
    else:
        group_by_plot = group_by
        bot_order_with_rank = None

    # --- Helper: apply x-axis cutoff ---
    def apply_timer_xlim(ax):
        if timer is not None:
            ax.set_xlim(0, timer)
            ax.set_xticks(range(0, int(timer) + 1, max(1, int(timer // 10) or 1)))

    # --- Plot modes ---
    if mode == "select":
        if not collision_type:
            raise ValueError("collision_type must be provided when mode='select'")
        if collision_type not in ["Actor_L", "Actor_R", "Tie"]:
            raise ValueError("collision_type must be one of ['Actor_L', 'Actor_R', 'Tie']")

        grouped = df.groupby([group_by_plot, "TimeBin"], as_index=False)[collision_type].mean()

        fig, ax = plt.subplots(figsize=(width, height))
        plot_with_bot_markers(ax, data=grouped, x="TimeBin", y=collision_type,
                            hue=group_by_plot, hue_order=bot_order_with_rank)
        ax.set_title(f"Mean {collision_type} collisions over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Count")
        legend_title = "Bot (Rank)" if (group_by in ["Bot_L", "Bot_R"] and rank_map is not None) else group_by
        ax.legend(title=legend_title, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        ax.grid(True, alpha=0.3)
        apply_timer_xlim(ax)
        fig.tight_layout()
        return fig

    elif mode == "total":
        # Sum all collision types
        df["TotalCollisions"] = df["Actor_L"] + df["Actor_R"] + df["Tie"]
        grouped = df.groupby([group_by_plot, "TimeBin"], as_index=False)["TotalCollisions"].mean()

        fig, ax = plt.subplots(figsize=(width, height))
        plot_with_bot_markers(ax, data=grouped, x="TimeBin", y="TotalCollisions",
                            hue=group_by_plot, hue_order=bot_order_with_rank)
        ax.set_title("Total collision intensity over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Count (summed over collision types)")
        legend_title = "Bot (Rank)" if (group_by in ["Bot_L", "Bot_R"] and rank_map is not None) else group_by
        ax.legend(title=legend_title, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        ax.grid(True, alpha=0.3)
        apply_timer_xlim(ax)
        fig.tight_layout()
        return fig

    elif mode == "per_type":
        collision_types = ["Actor_L", "Actor_R", "Tie"]
        n = len(collision_types)
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n,
            figsize=(width, height),
            squeeze=False
        )
        axes = axes.flatten()

        handles, labels = None, None
        for i, ctype in enumerate(collision_types):
            ax = axes[i]
            sub = df.groupby([group_by_plot, "TimeBin"], as_index=False)[ctype].mean()
            if sub.empty:
                ax.set_visible(False)
                continue

            # Plot with bot markers
            plot_with_bot_markers(ax, data=sub, x="TimeBin", y=ctype,
                                hue=group_by_plot, hue_order=bot_order_with_rank)

            # Capture legend handles from first plot, then remove it
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

            ax.set_title(ctype)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mean Count")
            ax.grid(True, alpha=0.3)
            apply_timer_xlim(ax)

        # Add global legend below
        if handles and labels:
            legend_title = "Bot (Rank)" if (group_by in ["Bot_L", "Bot_R"] and rank_map is not None) else group_by
            fig.legend(
                handles, labels, title=legend_title,
                loc="upper center", bbox_to_anchor=(0.5, -0.02),
                ncol=min(6, len(labels))
            )
        fig.suptitle("Per-collision-type intensity over timer")
        fig.tight_layout()
        return fig

    else:
        raise ValueError("mode must be one of ['total','per_type','select']")


def show_overall_analysis(df,filters,df_timebins, df_collision_timebins,toc,width,height):
    toc.h2("Overall Reports")
    st.markdown("Analyze bot agent facing other agent with similar configurations")

    # Action and Collision Behaviour Charts
    toc.h3("Bot Behaviour Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Actions Behaviour**")
        st.markdown("Mean action counts per bot across all configurations")
        st.pyplot(plot_action_radar(df))

    with col2:
        st.markdown("**Collision Behaviour**")
        st.markdown("Hit/Being hit/Tie distribution per bot")
        st.pyplot(plot_collision_radar(df))

    # Win Rate Matrix
    toc.h3("Win Rate Matrix")
    st.markdown("Shows how often each bot wins against others across different matchups.")
    st.markdown("This is calculated with taking mean of each configuration (10-games iteration matchup) resulting 240 games in total")
    st.pyplot(plot_winrate_matrix(df,width, height))

    toc.h3("Action Taken all configuration")
    st.pyplot(plot_overall_bot_metrics(df, metric="ActionCounts_L", title="Mean Action per Bot"))

    toc.h3("Action Duration all configuration")
    st.pyplot(plot_overall_bot_metrics(df, metric="Duration_L", title="Mean Action Duration per Bot"))

    toc.h3("Collision over all configuration")
    st.pyplot(plot_overall_bot_metrics(df, metric="Collisions_L", title="Mean Collisions per Bot"))

    toc.h3("Match Duration all configuration")
    st.pyplot(plot_overall_bot_metrics(df, metric="MatchDur", title="Mean Match Duration per Bot"))

    toc.h3("Win Rate grouped by Timer")
    st.pyplot(plot_grouped_config_winrates(df, config_col="Timer"))

    toc.h3("Win Rate grouped by ActInterval")
    st.pyplot(plot_grouped_config_winrates(df, config_col="ActInterval"))

    toc.h3("Win Rate grouped by Round")
    st.pyplot(plot_grouped_config_winrates(df, config_col="Round"))

    toc.h3("Win Rate grouped by Skill")
    st.pyplot(plot_grouped_config_winrates(df, config_col="Skill"))

    toc.h3("Collision grouped by Timer")
    st.pyplot(plot_grouped_config_winrates(df, metric="Collisions_L", config_col="Timer"))

    toc.h3("Collision grouped by ActInterval")  
    st.pyplot(plot_grouped_config_winrates(df, metric="Collisions_L", config_col="ActInterval"))

    toc.h3("Collision grouped by Round")
    st.pyplot(plot_grouped_config_winrates(df, metric="Collisions_L", config_col="Round"))

    toc.h3("Collision grouped by Skill")
    st.pyplot(plot_grouped_config_winrates(df, metric="Collisions_L", config_col="Skill"))

    toc.h3("Action Taken grouped by Timer")
    st.pyplot(plot_grouped_config_winrates(df, metric="ActionCounts_L", config_col="Timer"))

    toc.h3("Action Taken grouped by ActInterval")
    st.pyplot(plot_grouped_config_winrates(df, metric="ActionCounts_L", config_col="ActInterval"))

    toc.h3("Action Taken grouped by Round")
    st.pyplot(plot_grouped_config_winrates(df, metric="ActionCounts_L", config_col="Round"))

    toc.h3("Action Taken grouped by Skill")
    st.pyplot(plot_grouped_config_winrates(df, metric="ActionCounts_L", config_col="Skill"))

    toc.h3("Action Duration grouped by Timer")
    st.pyplot(plot_grouped_config_winrates(df, metric="Duration_L", config_col="Timer"))

    toc.h3("Action Duration grouped by ActInterval")
    st.pyplot(plot_grouped_config_winrates(df, metric="Duration_L", config_col="ActInterval"))

    toc.h3("Action Duration grouped by Round")
    st.pyplot(plot_grouped_config_winrates(df, metric="Duration_L", config_col="Round"))

    toc.h3("Action Duration grouped by Skill")
    st.pyplot(plot_grouped_config_winrates(df, metric="Duration_L", config_col="Skill"))

    toc.h3("Match Duration grouped by Timer")
    st.pyplot(plot_grouped_config_winrates(df, metric="MatchDur", config_col="Timer"))

    toc.h3("Match Duration grouped by ActInterval")
    st.pyplot(plot_grouped_config_winrates(df, metric="MatchDur", config_col="ActInterval"))

    toc.h3("Match Duration grouped by Round")
    st.pyplot(plot_grouped_config_winrates(df, metric="MatchDur", config_col="Round"))

    toc.h3("Match Duration grouped by Skill")
    st.pyplot(plot_grouped_config_winrates(df, metric="MatchDur", config_col="Skill"))

    # toc.h3("Win Rate over Timer Configuration")
    # st.pyplot(plot_bot_winrate_by_config(df,config_col="Timer"))

    # toc.h3("Win Rate over Round Configuration")
    # st.pyplot(plot_bot_winrate_by_config(df,config_col="Round"))

    # toc.h3("Win Rate over Action Interval Configuration")
    # st.pyplot(plot_bot_winrate_by_config(df,config_col="ActInterval"))

    # toc.h3("Full Configuration Cross Analysis")
    # st.pyplot(plot_full_cross_heatmap_half(df, bot_name="Bot_NN", lower_triangle=True))

    # toc.h3("Win Rate over SkillLeft Configuration")
    # st.pyplot(plot_bot_winrate_by_config(df,config_col="Timer"))

    # toc.h3("Win Rate over SkillRight Configuration")
    # st.pyplot(plot_bot_winrate_by_config(df,config_col="Timer"))

    # Time-Related Trends
    toc.h3("Time-Related Trends")
    st.markdown("Analyzes Bots aggressivenes over game duration with determining how much action taken duration related to the overall game duration (Time Setting)")
    st.markdown("Higher timers don't always lead to longer matches.\n"\
        "Some matchups finish fights early regardless of time limit.")
    figs = plot_time_related(df,width, height)
    for fig in figs:
        st.pyplot(fig)

    toc.h3(f"Action distribution per bots")
    fig = plot_action_distribution_stacked(df, normalize=True)
    st.pyplot(fig)

    for timI in filters["Timer"]:
        for actI in filters["ActInterval"]:

            toc.h3(f"Action intensity over Timer={timI}, ActionInterval={actI}")
            fig = plot_timebins_intensity(df_timebins, timer=timI, act_interval=actI, mode="total", summary_df=df)
            st.pyplot(fig)

            fig = plot_timebins_intensity(df_timebins, timer=timI, act_interval=actI, mode="per_action", summary_df=df)
            st.pyplot(fig)

    
    toc.h3(f"Action intensity over All Configuration")
    fig = plot_timebins_intensity(df_timebins, mode="total", summary_df=df)
    st.pyplot(fig)

    fig = plot_timebins_intensity(df_timebins, mode="per_action", summary_df=df)
    st.pyplot(fig)

    for timI in filters["Timer"]:
        for actI in filters["ActInterval"]:

            toc.h3(f"Collision intensity over Timer={timI}, ActionInterval={actI}")
            fig = plot_collision_timebins_intensity(df_collision_timebins, timer=timI, act_interval=actI, mode="total", summary_df=df)
            st.pyplot(fig)

            fig = plot_collision_timebins_intensity(df_collision_timebins, timer=timI, act_interval=actI, mode="per_type", summary_df=df)
            st.pyplot(fig)

    toc.h3(f"Collision detail distribution per bots")
    fig = plot_collision_distribution_stacked(df, normalize=True)
    st.pyplot(fig)

    toc.h3(f"Collision intensity over All Configuration")
    fig = plot_collision_timebins_intensity(df_collision_timebins, mode="total", summary_df=df)
    st.pyplot(fig)

    fig = plot_collision_timebins_intensity(df_collision_timebins, mode="per_type", summary_df=df)
    st.pyplot(fig)

    # Action vs. Win Relation
    toc.h3("Action taken vs. Win Relation")
    st.markdown("Does spending most action (aggresive) leads to a win?")
    st.markdown("This taking mean of action-taken per games versus win-rate")
    st.pyplot(plot_action_win_related(df,width, height))

    # Top Actions per Bot
    toc.h2("Top Actions per Bot")
    st.markdown("Shows the top 3 most frequent actions taken by each bot.")
    st.pyplot(plot_highest_action(df,width, height))

    # # Win Rate Stability vs. Timer
    # toc.h2("Win Rate Stability vs. Timer")
    # st.markdown("Examines if a bot's win rate fluctuates across different match durations.")
    # st.pyplot(plot_win_rate_stability_over_timer(df,width, height))
    
    