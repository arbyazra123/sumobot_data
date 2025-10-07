import streamlit as st
import pandas as pd

# Load or receive the same df
df = pd.read_csv("summary_matchup.csv")  # or share from session_state

st.markdown("## Complete Matchup Table")
st.dataframe(df, use_container_width=True, hide_index=True)
