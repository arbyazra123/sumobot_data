import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped(summary, key="WinRate", group_by="ActInterval", width=8, height=6):
    """
    Plot average win rate per bot, grouped by a specific configuration variable.
    group_by: one of ["ActInterval", "Timer", "Round", "Skill"]
    """
    # Merge both sides for fairness
    left = summary[["Bot_L", f"{key}_L", group_by]].rename(
        columns={"Bot_L": "Bot", f"{key}_L": key}
    )
    right = summary[["Bot_R", f"{key}_R", group_by]].rename(
        columns={"Bot_R": "Bot", f"{key}_R": key}
    )
    combined = pd.concat([left, right], ignore_index=True)

    # Aggregate average win rate
    grouped = (
        combined.groupby(["Bot", group_by])[key]
        .mean()
        .reset_index()
    )

    # Setup plotting
    bots = grouped["Bot"].unique()
    groups = sorted(grouped[group_by].unique())
    x = np.arange(len(bots))
    bar_width = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=(width, height))

    for i, g in enumerate(groups):
        subset = grouped[grouped[group_by] == g]
        avg_by_bot = subset.set_index("Bot").reindex(bots)[f"{key}"].fillna(0)
        ax.bar(x + i * bar_width, avg_by_bot, width=bar_width, label=str(g))

    # Labels & styling
    ax.set_xticks(x + bar_width * (len(groups) - 1) / 2)
    ax.set_xticklabels(bots, rotation=30, ha="right")
    ax.set_ylabel(key)
    ax.set_xlabel("Bots")
    # ax.set_ylim(0, 1)
    ax.legend(title=group_by)
    ax.set_title(f"{key} per Bot grouped by {group_by}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    return fig