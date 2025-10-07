import pandas as pd
import streamlit as st

from performance_chart import (
    plot_winrate_matrix,
    plot_time_related,
    plot_action_win_related,
    plot_highest_action,
    plot_win_rate_stability_over_timer,
    plot_win_rate_with_actinterval
)


if __name__ == "__main__":
    df_sum = pd.read_csv("summary_bot.csv")
    df_sum = df_sum.rename(columns={"Duration":"Duration (ms)"})
    df = pd.read_csv("summary_matchup.csv")

    st.set_page_config(page_title="Sumobot Performance Dashboard")
    width = st.sidebar.slider("plot width", 1, 25, 8)
    height = st.sidebar.slider("plot height", 1, 25, 6)

    st.markdown("""
<style>
/* Each .page class starts a new printed page */
.page {
    page-break-after: always;
    margin-bottom: 40px;
}

/* Optional: make titles more consistent across pages */
h1, h2, h3 {
    font-family: 'Helvetica', sans-serif;
}
</style>
""", unsafe_allow_html=True)

    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("# Sumobot Performance Dashboard")
    st.markdown("### A quick visual overview of bot performance metrics across matchups, timers, and actions.")
    st.markdown("This experiment conducted with bots: BT_Bot, Bot_MCTS, Bot_NN, and Bot_Primitive.")
    st.markdown("The configurations are:" \
    "\n- Timer\t= 45, 60" \
    "\n- Action Interval\t= 0.1, 0.2" \
    "\n- Round\t= Best of 1" \
    "\n- Skill\t= Boost")

    # Summary
    st.markdown("## Summary")
    st.dataframe(df_sum)
    st.markdown('</div>', unsafe_allow_html=True)

    # Win Rate Matrix
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## Win Rate Matrix")
    st.markdown("Shows how often each bot wins against others across different matchups.")
    st.markdown("This is calculated with taking mean of each configuration (10-games iteration matchup) resulting 240 games in total")
    st.pyplot(plot_winrate_matrix(df,width, height))
    st.markdown('</div class="page">', unsafe_allow_html=True)

    # Time-Related Trends
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## Time-Related Trends")
    st.markdown("Analyzes Bots aggressivenes over game duration with determining how much action taken duration related to the overall game duration (Time Setting)")
    st.markdown("Higher timers don't always lead to longer matches.\n"\
        "Some matchups finish fights early regardless of time limit.")
    figs = plot_time_related(df,width, height)
    for fig in figs:
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # Action vs. Win Relation
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## Action taken vs. Win Relation")
    st.markdown("Does spending most action (aggresive) leads to a win?")
    st.markdown("This taking mean of action-taken per games versus win-rate")
    st.pyplot(plot_action_win_related(df,width, height))
    st.markdown('</div>', unsafe_allow_html=True)

    # Top Actions per Bot
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## Top Actions per Bot")
    st.markdown("Shows the top 3 most frequent actions taken by each bot.")
    st.pyplot(plot_highest_action(df,width, height))
    st.markdown('</div>', unsafe_allow_html=True)

    # Win Rate Stability vs. Timer
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## Win Rate Stability vs. Timer")
    st.markdown("Examines if a bot's win rate fluctuates across different match durations.")
    st.pyplot(plot_win_rate_stability_over_timer(df,width, height))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Win Rate vs. Action Interval
    st.markdown('<div class="page">', unsafe_allow_html=True)
    st.markdown("## How Action Interval affects bot performance")
    st.markdown("Smaller intervals (more frequent decisions) might benefit reactive AIs")
    st.pyplot(plot_win_rate_with_actinterval(df,width, height))
    st.markdown('</div>', unsafe_allow_html=True)
