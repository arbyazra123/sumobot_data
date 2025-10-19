import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_grouped(summary, key="WinRate", group_by="ActInterval", width=8, height=6):
    """
    Plot average win rate per bot, grouped by a specific configuration variable.
    group_by: one of ["ActInterval", "Timer", "Round", "Skill"]
    """

    # --- Merge both sides ---
    left_cols = ["Bot_L", f"{key}_L", group_by]
    right_cols = ["Bot_R", f"{key}_R", group_by]

    if "Rank_L" in summary.columns:
        left_cols.append("Rank_L")
    if "Rank_R" in summary.columns:
        right_cols.append("Rank_R")

    left = summary[left_cols].rename(
        columns={"Bot_L": "Bot", f"{key}_L": key, "Rank_L": "Rank"}
    )
    right = summary[right_cols].rename(
        columns={"Bot_R": "Bot", f"{key}_R": key, "Rank_R": "Rank"}
    )

    combined = pd.concat([left, right], ignore_index=True)

    # Fill missing Rank with large number so unranked bots go last
    if "Rank" not in combined.columns:
        combined["Rank"] = np.nan
    combined["Rank"] = combined["Rank"].fillna(9999)

    # --- Aggregate ---
    grouped = (
        combined.groupby(["Bot", group_by], dropna=False)
        .agg({key: "mean", "Rank": "first"})
        .reset_index()
    )

    # --- Sort bots by Rank ---
    bot_order = grouped.groupby("Bot")["Rank"].first().sort_values().index.tolist()
    grouped["Bot"] = pd.Categorical(grouped["Bot"], categories=bot_order, ordered=True)
    grouped = grouped.sort_values(["Bot", group_by])

    # --- Create X labels with rank ---
    labels = [
        f"{b} (#{int(grouped[grouped['Bot'] == b]['Rank'].iloc[0])})"
        for b in bot_order
    ]

    # --- Plot ---
    groups = sorted(grouped[group_by].unique())
    x = np.arange(len(bot_order))
    bar_width = 0.8 / len(groups)

    fig, ax = plt.subplots(figsize=(width, height))
    for i, g in enumerate(groups):
        subset = grouped[grouped[group_by] == g]
        avg_by_bot = subset.set_index("Bot").reindex(bot_order)[key].fillna(0)
        ax.bar(x + i * bar_width, avg_by_bot, width=bar_width, label=str(g))

    # --- Style ---
    ax.set_xticks(x + bar_width * (len(groups) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(key)
    ax.set_xlabel("Bots")
    ax.legend(title=group_by)
    ax.set_title(f"{key} per Bot grouped by {group_by}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

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

    toc.h4("Win Rate by SkillLeft")
    st.pyplot(plot_grouped(df,group_by="SkillLeft",width=width, height=height))

    toc.h4("Win Rate by SkillRight")
    st.pyplot(plot_grouped(df,group_by="SkillRight",width=width, height=height))


    # Individual Action Taken
    toc.h3("Action Taken")
    st.markdown("Reports of action taken from each of bot")

    toc.h4("Action Counts by Timer")
    st.pyplot(plot_grouped(df,key="ActionCounts", group_by="Timer",width=width, height=height))

    toc.h4("Action Counts by ActInterval")
    st.pyplot(plot_grouped(df,key="ActionCounts", group_by="ActInterval",width=width, height=height))

    toc.h4("Action Counts by Round")
    st.pyplot(plot_grouped(df,key="ActionCounts", group_by="Round",width=width, height=height))

    toc.h4("Action Counts by SkillLeft")
    st.pyplot(plot_grouped(df,key="ActionCounts", group_by="SkillLeft",width=width, height=height))

    toc.h4("Action Counts by SkillRight")
    st.pyplot(plot_grouped(df,key="ActionCounts", group_by="SkillRight",width=width, height=height))
    
    # Individual Collision
    toc.h3("Collisions")
    st.markdown("Reports of collision made from each of bot")

    toc.h4("Collisions by Timer")
    st.pyplot(plot_grouped(df,key="Collisions", group_by="Timer",width=width, height=height))

    toc.h4("Collisions by ActInterval")
    st.pyplot(plot_grouped(df,key="Collisions", group_by="ActInterval",width=width, height=height))

    toc.h4("Collisions by Round")
    st.pyplot(plot_grouped(df,key="Collisions", group_by="Round",width=width, height=height))

    toc.h4("Collisions by SkillLeft")
    st.pyplot(plot_grouped(df,key="Collisions", group_by="SkillLeft",width=width, height=height))

    toc.h4("Collisions by SkillRight")
    st.pyplot(plot_grouped(df,key="Collisions", group_by="SkillRight",width=width, height=height))

    # Individual Duration
    toc.h3("Duration")
    st.markdown("Reports of action-taken duration produced from each of bot")

    toc.h4("Duration by Timer")
    st.pyplot(plot_grouped(df,key="Duration", group_by="Timer",width=width, height=height))

    toc.h4("Duration by ActInterval")
    st.pyplot(plot_grouped(df,key="Duration", group_by="ActInterval",width=width, height=height))

    toc.h4("Duration by Round")
    st.pyplot(plot_grouped(df,key="Duration", group_by="Round",width=width, height=height))

    toc.h4("Duration by SkillLeft")
    st.pyplot(plot_grouped(df,key="Duration", group_by="SkillLeft",width=width, height=height))

    toc.h4("Duration by SkillRight")
    st.pyplot(plot_grouped(df,key="Duration", group_by="SkillRight",width=width, height=height))
