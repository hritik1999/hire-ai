import streamlit as st
import pandas as pd
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

st.sidebar.title("Hire AI Leaderboard")
st.title("Leaderboard")
st.write("Category wise leaderboard")
data = pd.DataFrame.from_dict(rows.data)

for category in categories:
    st.subheader(category)
    st.table(data[data["Category"]==category].sort_values(by=["Final Score"], ascending=False))


