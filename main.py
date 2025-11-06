from streamlit_modal import Modal
import pandas as pd
import streamlit as st
from stoc import stoc

from overall_analyzer import (
    show_overall_analysis,
)

from individual_analyzer import (
    show_individual_report
)

@st.cache_data
def load_summary_data():
    df_sum = pd.read_csv("summary_bot.csv").rename(columns={"Duration": "Duration (ms)"})
    df = pd.read_csv("summary_matchup.csv")
    df_timebins = pd.read_csv("summary_action_timebins.csv")
    df_sum = df_sum[df_sum["Bot"] != "Bot_FSM"]
    df = df[df["Bot_L"] != "Bot_FSM"]
    df = df[df["Bot_R"] != "Bot_FSM"]
    return df_sum, df, df_timebins

if __name__ == "__main__":
    toc = stoc()

    df_sum, df, df_timebins = load_summary_data()
    df_sum = df_sum.rename(columns={"Duration":"Duration (ms)"})

    st.set_page_config(page_title="Sumobot Performance Dashboard")
    width = st.sidebar.slider("plot width", 1, 25, 8)
    height = st.sidebar.slider("plot height", 1, 25, 6)

    cfg = {
        "Timer": sorted(df["Timer"].unique().tolist()),
        "ActInterval": sorted(df["ActInterval"].unique().tolist()),
        "Round": sorted(df["Round"].unique().tolist()),
        "SkillLeft": sorted(df["SkillLeft"].unique().tolist()),
        "SkillRight": sorted(df["SkillRight"].unique().tolist()),
        "Bots": sorted(df["Bot_L"].unique().tolist()),
    }
    bots = str.join(", ", cfg["Bots"])

    toc.h1("Sumobot Performance Dashboard")
    st.markdown("A quick visual overview of bot performance metrics across matchups, timers, and actions")
    st.markdown(f"This experiment conducted with bots: {bots}")
    st.markdown("Configuration :" )
    st.write(cfg)
    st.markdown("Source code: https://github.com/arbyazra123/sumobot_data")

    # Summary
    toc.h2("Summary Matchup")
    st.dataframe(df_sum)

    modal = Modal("Complete Matchup", key="matchup")
    if st.button("View complete matchup"):
        modal.open()

    if modal.is_open():
        with modal.container():
            st.dataframe(df, use_container_width=True, hide_index=True)

    show_individual_report(df,toc,width,height)

    show_overall_analysis(df,cfg,df_timebins,toc,width,height)

    toc.toc()

