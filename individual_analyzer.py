import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_grouped(summary, key="WinRate", group_by="ActInterval", width=10, height=7, chart_type="line", error_type="std"):
    """
    Plot average win rate per bot, grouped by a specific configuration variable.

    Parameters:
        group_by: one of ["ActInterval", "Timer", "Round", "SkillType"]
        chart_type: "line" for line chart with error bands, "bar" for bar chart
        error_type: "se" for standard error (recommended), "std" for standard deviation,
                    "ci" for 95% confidence interval
    """

    # --- Handle SkillType special case ---
    # SkillType combines SkillLeft and SkillRight into a unified grouping
    if group_by == "SkillType":
        left_group_col = "SkillLeft"
        right_group_col = "SkillRight"
    else:
        left_group_col = group_by
        right_group_col = group_by

    # --- Merge both sides ---
    left_cols = ["Bot_L", f"{key}_L", left_group_col]
    right_cols = ["Bot_R", f"{key}_R", right_group_col]

    if "Rank_L" in summary.columns:
        left_cols.append("Rank_L")
    if "Rank_R" in summary.columns:
        right_cols.append("Rank_R")

    left = summary[left_cols].rename(
        columns={"Bot_L": "Bot", f"{key}_L": key, "Rank_L": "Rank", left_group_col: group_by}
    )
    right = summary[right_cols].rename(
        columns={"Bot_R": "Bot", f"{key}_R": key, "Rank_R": "Rank", right_group_col: group_by}
    )

    combined = pd.concat([left, right], ignore_index=True)

    # Fill missing Rank with large number so unranked bots go last
    if "Rank" not in combined.columns:
        combined["Rank"] = np.nan
    combined["Rank"] = combined["Rank"].fillna(9999)

    # --- Aggregate (with std and count) ---
    grouped = (
        combined.groupby(["Bot", group_by], dropna=False)
        .agg({key: ["mean", "std", "count"], "Rank": "first"})
        .reset_index()
    )

    # Flatten column names
    grouped.columns = ["Bot", group_by, f"{key}_mean", f"{key}_std", f"{key}_count", "Rank"]
    grouped[f"{key}_std"] = grouped[f"{key}_std"].fillna(0)  # Handle cases with no std
    grouped[f"{key}_count"] = grouped[f"{key}_count"].fillna(1)  # Avoid division by zero

    # --- Sort bots by Rank ---
    bot_order = grouped.groupby("Bot")["Rank"].first().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(width, height))

    if chart_type == "line":
        # --- Line chart with error bands ---
        x_values = sorted(grouped[group_by].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(bot_order)))

        for i, bot in enumerate(bot_order):
            bot_data = grouped[grouped["Bot"] == bot].sort_values(group_by)
            rank = int(bot_data["Rank"].iloc[0])

            means = []
            errors = []
            for x_val in x_values:
                row = bot_data[bot_data[group_by] == x_val]
                if not row.empty:
                    mean_val = row[f"{key}_mean"].values[0]
                    std_val = row[f"{key}_std"].values[0]
                    count_val = row[f"{key}_count"].values[0]

                    means.append(mean_val)

                    # Calculate error based on error_type
                    if error_type == "se":
                        # Standard Error
                        error = std_val / np.sqrt(count_val) if count_val > 0 else 0
                    elif error_type == "ci":
                        # 95% Confidence Interval (approximation using 1.96 * SE)
                        error = 1.96 * (std_val / np.sqrt(count_val)) if count_val > 0 else 0
                    else:  # "std"
                        error = std_val

                    errors.append(error)
                else:
                    means.append(np.nan)
                    errors.append(0)

            means = np.array(means)
            errors = np.array(errors)

            # Plot line with thicker style
            ax.plot(x_values, means, 'o-', linewidth=2.5, markersize=7,
                   label=f"{bot} (#{rank})", color=colors[i])

            # Plot error band with lighter transparency
            ax.fill_between(x_values, means - errors, means + errors,
                          alpha=0.15, color=colors[i])

        ax.set_xlabel(group_by, fontsize=12, fontweight='bold')
        ax.set_ylabel(key, fontsize=12, fontweight='bold')
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values])

        # Set Y-axis limits for WinRate
        if key == "WinRate":
            ax.set_ylim(-0.05, 1.05)

    else:
        # --- Bar chart (original) ---
        grouped_bar = grouped.rename(columns={f"{key}_mean": key})
        grouped_bar["Bot"] = pd.Categorical(grouped_bar["Bot"], categories=bot_order, ordered=True)
        grouped_bar = grouped_bar.sort_values(["Bot", group_by])

        labels = [
            f"{b} (#{int(grouped_bar[grouped_bar['Bot'] == b]['Rank'].iloc[0])})"
            for b in bot_order
        ]

        groups = sorted(grouped_bar[group_by].unique())
        x = np.arange(len(bot_order))
        bar_width = 0.8 / len(groups)

        for i, g in enumerate(groups):
            subset = grouped_bar[grouped_bar[group_by] == g]
            avg_by_bot = subset.set_index("Bot").reindex(bot_order)[key].fillna(0)
            ax.bar(x + i * bar_width, avg_by_bot, width=bar_width, label=str(g))

        ax.set_xticks(x + bar_width * (len(groups) - 1) / 2)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Bots")

    # --- Common styling ---
    ax.set_title(f"{key} grouped by {group_by}", fontsize=14, fontweight='bold', pad=15)
    ax.legend(title="Bot (Rank)" if chart_type == "line" else group_by,
             loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.8)
    fig.tight_layout()

    return fig

def show_individual_report(df, toc, width, height):
    toc.h2("Individual Reports")
    st.markdown("Analyze bot agent against its different configurations")
    st.markdown("Each of report: Win Rate; Collision; Action-Taken; Duration; is calculated with averaging data from matchup (left and right position)")
    
    # Individual Win Rates
    toc.h3("Win Rates")
    st.markdown("Reports of win rates each of bot")
    
    toc.h4("Win Rate by Timer")
    st.pyplot(plot_grouped(df,group_by="Timer",width=width, height=height))

    toc.h4("Win Rate by ActInterval")
    st.pyplot(plot_grouped(df,group_by="ActInterval",width=width, height=height))

    toc.h4("Win Rate by Round")
    st.pyplot(plot_grouped(df,group_by="Round",width=width, height=height))

    toc.h4("Win Rate by SkillType")
    st.pyplot(plot_grouped(df,group_by="SkillType",width=width, height=height))


    # Individual Action Taken
    toc.h3("Action Taken")
    st.markdown("Reports of action taken from each of bot")

    toc.h4("Action Counts by Timer")
    st.pyplot(plot_grouped(df,key="ActionCounts", group_by="Timer",width=width, height=height))

    toc.h4("Action Counts by ActInterval")
    st.pyplot(plot_grouped(df,key="ActionCounts", group_by="ActInterval",width=width, height=height))

    toc.h4("Action Counts by Round")
    st.pyplot(plot_grouped(df,key="ActionCounts", group_by="Round",width=width, height=height))

    toc.h4("Action Counts by SkillType")
    st.pyplot(plot_grouped(df,key="ActionCounts", group_by="SkillType",width=width, height=height))
    
    # Individual Collision
    toc.h3("Collisions")
    st.markdown("Reports of collision made from each of bot")

    toc.h4("Collisions by Timer")
    st.pyplot(plot_grouped(df,key="Collisions", group_by="Timer",width=width, height=height))

    toc.h4("Collisions by ActInterval")
    st.pyplot(plot_grouped(df,key="Collisions", group_by="ActInterval",width=width, height=height))

    toc.h4("Collisions by Round")
    st.pyplot(plot_grouped(df,key="Collisions", group_by="Round",width=width, height=height))

    toc.h4("Collisions by SkillType")
    st.pyplot(plot_grouped(df,key="Collisions", group_by="SkillType",width=width, height=height))

    # Individual Duration
    toc.h3("Duration")
    st.markdown("Reports of action-taken duration produced from each of bot")

    toc.h4("Duration by Timer")
    st.pyplot(plot_grouped(df,key="Duration", group_by="Timer",width=width, height=height))

    toc.h4("Duration by ActInterval")
    st.pyplot(plot_grouped(df,key="Duration", group_by="ActInterval",width=width, height=height))

    toc.h4("Duration by Round")
    st.pyplot(plot_grouped(df,key="Duration", group_by="Round",width=width, height=height))

    toc.h4("Duration by SkillType")
    st.pyplot(plot_grouped(df,key="Duration", group_by="SkillType",width=width, height=height))
