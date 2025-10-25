import numpy as np
import io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import plotly.io as pio
import streamlit as st


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
            ax.plot(bot_data["Timer"], bot_data["AvgDuration"], marker="o", label=label)

        ax.set_title(f"Avg Match Duration vs Timer (ActInterval = {interval})")
        ax.set_xlabel("Timer (s)")
        ax.set_ylabel("Actual Match Duration (s)")
        ax.legend()
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

    df_actions = summary.melt(
        id_vars=["Bot_L"], 
        value_vars=action_cols,
        var_name="Action", 
        value_name="Count"
    )
    df_actions["Action"] = df_actions["Action"].str.replace("_Act_L", "")
    df_actions = df_actions.groupby(["Bot_L", "Action"])["Count"].sum().reset_index()
    top_actions = df_actions.groupby("Bot_L").apply(lambda x: x.nlargest(n_action, "Count")).reset_index(drop=True)
    fig = plt.figure(figsize=(width,height))
    sns.barplot(
        data=top_actions, 
        x="Count", 
        y="Action", 
        hue="Bot_L"
    )
    plt.title("Top 3 Actions Taken per Bot")
    plt.xlabel("Action Count")
    plt.ylabel("Action")
    plt.legend(title="Bot")
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

@st.cache_data
def plot_win_rate_with_actinterval(summary, width, height):
    # Optional: make ActInterval numeric
    summary["ActInterval"] = pd.to_numeric(summary["ActInterval"], errors='coerce')

    # Plot: Win Rate vs ActInterval
    fig, ax = plt.subplots(figsize=(width,height))
    sns.lineplot(
        data=summary,
        x="ActInterval",
        y="WinRate_L",
        hue="Bot_L",
        marker="o",
        ax=ax
    )

    # Titles and labels
    ax.set_title("Win Rate vs Action Interval (ActInterval) per Bot", fontsize=16)
    ax.set_xlabel("Action Interval (ms)", fontsize=14)
    ax.set_ylabel("Win Rate (Left Bot)", fontsize=14)
    ax.set_ylim(0, 1)

    unique_timers = sorted(summary["ActInterval"].unique())
    plt.xticks(unique_timers, [f"{t}" for t in unique_timers])

    # Optional: annotate each point with win rate
    for line in ax.lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            ax.text(x, y + 0.02, f"{y:.2f}", ha='center', fontsize=9)

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
        grouped = df_sel.groupby([group_by, "TimeBin"], as_index=False)["MeanCount"].mean()

        fig, ax = plt.subplots(figsize=(width, height))
        sns.lineplot(data=grouped, x="TimeBin", y="MeanCount", hue=group_by, marker="o", ax=ax)
        ax.set_title(f"Mean {action_name} over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Count")
        ax.grid(True, alpha=0.3)
        apply_timer_xlim(ax)
        fig.tight_layout()
        return fig

    elif mode == "total":
        grouped = df.groupby([group_by, "TimeBin"], as_index=False)["MeanCount"].mean()
        fig, ax = plt.subplots(figsize=(width, height))
        sns.lineplot(data=grouped, x="TimeBin", y="MeanCount", hue=group_by, marker="o", ax=ax)
        ax.set_title("Total action intensity over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean Count (summed over actions)")
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

        for i, action in enumerate(actions):
            ax = axes[i]
            sub = df[df["Action"] == action].groupby([group_by, "TimeBin"], as_index=False)["MeanCount"].mean()
            if sub.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(data=sub, x="TimeBin", y="MeanCount", hue=group_by, marker="o", ax=ax, legend=(i==0))
            ax.set_title(action)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mean Count")
            ax.grid(True, alpha=0.3)
            apply_timer_xlim(ax)

        # Hide unused axes
        for j in range(len(actions), len(axes)):
            axes[j].set_visible(False)

        # Add global legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, title=group_by,
            loc="upper center", bbox_to_anchor=(0.5, 0.98),
            ncol=min(6, len(labels))
        )
        fig.suptitle("Per-action intensity over timer")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    else:
        raise ValueError("mode must be one of ['total','per_action','select']")


def ablation_summary(df, metric="WinRate", ignore_params=None, group_by=None):
    """
    Perform an 'ablation-style' aggregation by ignoring one or more configuration parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The summary dataframe (e.g. matchup_summary or bot_summary).
    metric : str
        The target metric to average, e.g. "WinRate_L", "WinRate_R", "TotalActions", etc.
    ignore_params : list[str]
        List of column names to *ignore* (aggregate over).
        Example: ["Timer"] means you don't care about timer differences.
    group_by : list[str] or None
        Columns to group by (other than ignored ones). 
        If None, automatically uses all config columns minus ignored ones.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame averaged over the ignored parameters.
    """
    if ignore_params is None:
        ignore_params = []

    # Detect configuration columns
    config_cols = ["Bot_L", "Bot_R", "Timer", "ActInterval", "Round", "SkillLeft", "SkillRight"]

    # Build group-by columns: remove ignored ones
    if group_by is None:
        group_by = [c for c in config_cols if c not in ignore_params]

    # Perform aggregation
    agg_df = (
        df.groupby(group_by, dropna=False)[metric]
        .mean()
        .reset_index()
        .sort_values(metric, ascending=False)
    )

    agg_df["IgnoredParams"] = ", ".join(ignore_params) if ignore_params else "(none)"
    return agg_df

def plot_ablation_compare(df, metric="WinRate_L", ignore_param="Timer", x="SkillLeft", hue="SkillRight"):
    """
    Plot comparison of WinRate with and without a specified parameter (e.g. Timer).

    Parameters
    ----------
    df : pd.DataFrame
        Your summary DataFrame (e.g. matchup_summary or bot_summary)
    metric : str
        Metric to compare, default "WinRate_L"
    ignore_param : str
        Parameter to ablate (ignore)
    x : str
        Column for x-axis (e.g. "SkillLeft")
    hue : str
        Column for hue grouping (e.g. "SkillRight")

    Returns
    -------
    matplotlib.figure.Figure
    """
    # --- Normal grouping (with Timer) ---
    df_with = df.groupby([x, hue, ignore_param])[metric].mean().reset_index()
    df_with["Type"] = f"With {ignore_param}"

    # --- Ablated grouping (ignore Timer) ---
    df_without = df.groupby([x, hue])[metric].mean().reset_index()
    df_without["Type"] = f"Without {ignore_param}"
    df_without[ignore_param] = "ALL"  # placeholder for visualization

    # --- Combine ---
    df_compare = pd.concat([df_with, df_without], ignore_index=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_compare,
        x=x,
        y=metric,
        hue="Type",
        errorbar=None,
        ax=ax
    )

    ax.set_title(f"{metric} Comparison: With vs Without {ignore_param}")
    ax.set_xlabel(x)
    ax.set_ylabel(f"Mean {metric}")
    ax.legend(title="Condition")
    plt.tight_layout()
    return fig

def plot_ablation_winrate(df, bot_col="Bot_L", ignore_param="Timer", winrate_col="WinRate_L"):
    """
    Creates a bar chart comparing win rates with vs ignoring a parameter.
    Returns a matplotlib figure for use in Streamlit (st.pyplot).

    Parameters:
        df (pd.DataFrame): The summary_matchup dataframe
        bot_col (str): Column name for the bot
        opponent_col (str): Column name for opponent
        ignore_param (str): Parameter to ignore for ablation
        winrate_col (str): Column for winrate ('WinRate_L' or 'WinRate_R')

    Returns:
        matplotlib.figure.Figure
    """

    # --- With the parameter included ---
    df_with = df.copy()
    df_with["Label"] = "With " + ignore_param

    # --- Ignoring the parameter (ablation) ---
    group_cols = [c for c in df.columns if c not in [ignore_param, "Winner_L", "Winner_R", "Games", "WinRate_L", "WinRate_R", "Rank_L", "Rank_R"]]
    df_ignore = df.groupby(group_cols, as_index=False).agg({
        "Games": "sum",
        "Winner_L": "sum",
        "Winner_R": "sum"
    })

    df_ignore["WinRate_L"] = df_ignore["Winner_L"] / df_ignore["Games"]
    df_ignore["WinRate_R"] = df_ignore["Winner_R"] / df_ignore["Games"]
    df_ignore["Label"] = "Ignore " + ignore_param

    # --- Combine for plotting ---
    df_plot = pd.concat([df_with[group_cols + [winrate_col, "Label"]],
                         df_ignore[group_cols + [winrate_col, "Label"]]],
                        ignore_index=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_plot, x=bot_col, y=winrate_col, hue="Label", ax=ax)
    ax.set_title(f"Ablation: WinRate with vs ignoring {ignore_param}")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.legend(title="")
    plt.tight_layout()

    return fig


def plot_surface_interaction(
    summary: pd.DataFrame,
    key: str = "WinRate_L",
    bot_col: str = "Bot_L",
    x_col: str = "Timer",
    y_col: str = "ActInterval",
    bot_name: str | None = None,
    kind: str = "heatmap",  # or "surface"
    width: int = 8,
    height: int = 6,
    ax=None,
):
    """
    Cross analysis plot: visualize relationship between (x_col × y_col) vs a metric (key).
    Compatible with Streamlit (returns Figure).
    """

    df = summary.copy()

    # --- Filter bot if specified ---
    if bot_name is not None:
        df = df[df[bot_col] == bot_name]
        title_bot = bot_name
    else:
        title_bot = "All Bots (avg)"
        
    
    # --- Create pivot table ---
    pivot = df.pivot_table(index=y_col, columns=x_col, values=key, aggfunc="mean")

    if pivot.empty:
        print("⚠️ No data available for this combination.")
        return None

    # --- Create figure object ---
    fig = plt.figure(figsize=(width, height))

    
    if ax is None:
        ax = fig.add_subplot(111)
    sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    label = key.replace("_L","")
    ax.set_title(f"{label} Heatmap ({title_bot})")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    fig.tight_layout()
    return fig

def plot_bot_winrate_by_config(
    summary: pd.DataFrame,
    key: str = "WinRate_L",
    bot_col: str = "Bot_L",
    config_col: str = "Timer",
    width: int = 8,
    height: int = 6,
    kind: str = "line",  # "bar" or "line"
):
    """
    Plot average WinRate (or other metric) per Bot across a given configuration column.
    Adds rank info (from rank_col) to legend labels.
    """

    # --- Compute average per bot and config ---
    grouped = summary.groupby([bot_col, config_col])[key].mean().reset_index()

    # --- Get bot rank mapping ---
    rank_map = (
        summary.groupby(bot_col)["Rank_L"]
        .mean()
        .sort_values()
        .round(0)
        .astype(int)
        .to_dict()
    )

    # --- Rename bot labels with rank ---
    grouped["BotWithRank"] = grouped[bot_col].map(
        lambda b: f"{b} (#{rank_map.get(b, '?')})"
    )

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(width, height))

    # --- Plot ---
    if kind == "bar":
        sns.barplot(
            data=grouped,
            x=config_col,
            y=key,
            hue="BotWithRank",
            ax=ax,
            errorbar=None,
        )
    else:
        sns.lineplot(
            data=grouped,
            x=config_col,
            y=key,
            hue="BotWithRank",
            marker="o",
            linewidth=2,
            ax=ax,
        )

    # --- Fix x-axis for numeric config_col ---
    unique_vals = sorted(grouped[config_col].unique())
    if pd.api.types.is_numeric_dtype(grouped[config_col]):
        ax.set_xticks(unique_vals)
        ax.set_xticklabels([str(v) for v in unique_vals])

    # --- Style ---
    label = key.replace("_L", "")
    ax.set_title(f"{label} vs {config_col} (Avg per Bot)")
    ax.set_xlabel(config_col)
    ax.set_ylabel(label)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Bot (Rank)")

    fig.tight_layout()
    return fig

def plot_multi_surface_interactions(df, bot_name, key="WinRate_L"):
    pairs = [
        ("Timer", "ActInterval"),
        ("Timer", "Round"),
        ("Timer", "SkillLeft"),
        ("Timer", "SkillRight"),
        ("ActInterval", "Round"),
        ("ActInterval", "SkillLeft"),
        ("ActInterval", "SkillRight"),
        ("Round", "SkillLeft"),
        ("Round", "SkillRight"),
        ("SkillLeft", "SkillRight"),
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, (x_col, y_col) in enumerate(pairs):
        ax = axes[i]
        plot_surface_interaction(df, bot_name=bot_name, key=key, x_col=x_col, y_col=y_col, ax=ax)
        ax.set_title(f"{x_col} x {y_col}", fontsize=10)
    
    # Hide any extra subplots if pairs < grid size
    for j in range(len(pairs), len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle(f"Cross Analysis Win Rate for {bot_name}", fontsize=14)
    fig.tight_layout()
    return fig

def plot_cross_matrix(df, bot_name="Bot_NN", key="WinRate_L"):
    cfg_cols = ["ActInterval", "Round", "SkillLeft", "SkillRight"]
    
    # Filter for this bot
    df_bot = df[df["Bot_L"] == bot_name].copy()

    # Melt configuration columns into a single axis
    melted = df_bot.melt(
        id_vars=["Timer", key],
        value_vars=cfg_cols,
        var_name="ConfigType",
        value_name="ConfigValue"
    )

    # Compute mean win rate by Timer × ConfigType × ConfigValue
    grouped = (
        melted.groupby(["Timer", "ConfigType", "ConfigValue"])[key]
        .mean()
        .reset_index()
    )

    # Combine ConfigType + Value into a readable label
    grouped["Config"] = grouped["ConfigType"] + "=" + grouped["ConfigValue"].astype(str)

    # Pivot for heatmap
    pivot = grouped.pivot(index="Config", columns="Timer", values=key)

    # Plot
    fig, ax = plt.subplots(figsize=(8, len(pivot) * 0.4 + 2))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f", ax=ax)
    
    ax.set_title(f"Cross Analysis of Win Rate vs Timer for {bot_name}")
    ax.set_xlabel("Timer")
    ax.set_ylabel("Configuration (Type=Value)")

    fig.tight_layout()
    return fig

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


def show_overall_analysis(df,filters,df_timebins,toc,width,height):
    toc.h2("Overall Reports")
    st.markdown("Analyze bot agent facing other agent with similar configurations")

    # Win Rate Matrix
    toc.h3("Win Rate Matrix")
    st.markdown("Shows how often each bot wins against others across different matchups.")
    st.markdown("This is calculated with taking mean of each configuration (10-games iteration matchup) resulting 240 games in total")
    st.pyplot(plot_winrate_matrix(df,width, height))

    toc.h3("Win Rate over Timer Configuration")
    st.pyplot(plot_bot_winrate_by_config(df,config_col="Timer"))

    toc.h3("Win Rate over Round Configuration")
    st.pyplot(plot_bot_winrate_by_config(df,config_col="Round"))

    toc.h3("Win Rate over Action Interval Configuration")
    st.pyplot(plot_bot_winrate_by_config(df,config_col="ActInterval"))

    toc.h3("Full Configuration Cross Analysis")
    st.pyplot(plot_full_cross_heatmap_half(df, bot_name="Bot_NN", lower_triangle=True))


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

    for timI in filters["Timer"]:
        for actI in filters["ActInterval"]:

            toc.h3(f"Action intensity over Timer={timI}, ActionInterval={actI}")
            fig = plot_timebins_intensity(df_timebins, timer=timI, act_interval=actI, mode="total")
            st.pyplot(fig)

            fig = plot_timebins_intensity(df_timebins, timer=timI, act_interval=actI, mode="per_action")
            st.pyplot(fig)

    
    toc.h3(f"Action intensity over All Configuration")
    fig = plot_timebins_intensity(df_timebins, mode="total")
    st.pyplot(fig)

    fig = plot_timebins_intensity(df_timebins, mode="per_action")
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
    
    