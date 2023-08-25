import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client

def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)

supabase = init_connection()

def run_query():
    return supabase.table("Leaderboard").select("*").execute()

categories = ["Leadership and Management","Sales and Marketing","Finance and Operations","Technology and Innovation","Operation and Supply Chain","Human Resource"]

rows = run_query()

st.set_page_config(page_title="Hire AI",page_icon=" :briefcase: ",layout="wide")

st.sidebar.title("Hire AI Leaderboard")
st.title("Category wise leaderboard")
data = pd.DataFrame.from_dict(rows.data).drop(columns=["id"])

for category in categories:
    st.subheader(category)
    df = data[data["Category"]==category].sort_values(by=["Final Score"], ascending=False)
    df.index = np.arange(1, len(df) + 1)
    df.index.name = "Rank"
    st.table(df[["Name","Role","Final Score"]])


