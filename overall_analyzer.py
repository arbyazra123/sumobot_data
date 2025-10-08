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
    left = summary[["Bot_L", "Bot_R", "WinRate_L"]].rename(
        columns={"Bot_L": "Bot", "Bot_R": "Enemy", "WinRate_L": "WinRate"}
    )
    right = summary[["Bot_R", "Bot_L", "WinRate_R"]].rename(
        columns={"Bot_R": "Bot", "Bot_L": "Enemy", "WinRate_R": "WinRate"}
    )

    combined = pd.concat([left, right], ignore_index=True)

    # Aggregate mean winrate over all configs
    matrix_df = combined.groupby(["Bot", "Enemy"])["WinRate"].mean().reset_index()

    # Pivot into matrix
    pivot = matrix_df.pivot(index="Bot", columns="Enemy", values="WinRate")

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
        summary.groupby(["ActInterval", "Timer", "Bot_L"], as_index=False)
        .agg({"Duration_L": "mean", "Games": "mean"})
    )
    grouped["AvgDuration"] = grouped["Duration_L"] / grouped["Games"]

    for interval in grouped["ActInterval"].unique():
        fig, ax = plt.subplots(figsize=(width, height))
        subset = grouped[grouped["ActInterval"] == interval]

        for bot in subset["Bot_L"].unique():
            bot_data = subset[subset["Bot_L"] == bot]
            ax.plot(bot_data["Timer"], bot_data["AvgDuration"], marker="o", label=bot)

        ax.set_title(f"Avg Match Duration vs Timer (ActInterval = {interval})")
        ax.set_xlabel("Timer (s)")
        ax.set_ylabel("Estimated Match Duration (s)")
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
def plot_highest_action(summary, width=8, height=6, n_action = 3):
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