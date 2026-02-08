"""Antigravity Node — Thought Trace Viewer (Streamlit UI)."""

import os

import pandas as pd
import pymysql
import streamlit as st

st.set_page_config(
    page_title="Antigravity Thought Trace",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Antigravity Node — Thought Trace")
st.caption("Real-time view of agent episodic memory (StarRocks)")

STARROCKS_HOST = os.environ.get("STARROCKS_HOST", "starrocks")
STARROCKS_PORT = int(os.environ.get("STARROCKS_PORT", "9030"))
STARROCKS_USER = os.environ.get("STARROCKS_USER", "root")


def get_connection():
    """Get StarRocks connection with reconnect support."""
    try:
        conn = pymysql.connect(
            host=STARROCKS_HOST,
            port=STARROCKS_PORT,
            user=STARROCKS_USER,
            database="antigravity",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
        )
        return conn
    except pymysql.Error as e:
        st.error(f"Cannot connect to StarRocks: {e}")
        return None


try:
    conn = get_connection()

    if conn is None:
        st.info("Make sure StarRocks is running and the memory schema has been initialized.")
        st.stop()

    # Ensure connection is alive before using it
    try:
        conn.ping(reconnect=True)
    except pymysql.Error as e:
        st.error(f"Connection lost: {e}")
        conn = get_connection()
        if conn is None:
            st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")
    actor_filter = st.sidebar.selectbox("Actor", ["All", "Goose", "User"])
    action_filter = st.sidebar.selectbox(
        "Action Type", ["All", "THOUGHT", "TOOL_USE", "RESPONSE", "TASK_REQUEST", "FILE_UPLOAD"]
    )
    limit = st.sidebar.slider("Max Records", 10, 500, 100)

    # Build query
    where_clauses = []
    params = []
    if actor_filter != "All":
        where_clauses.append("actor = %s")
        params.append(actor_filter)
    if action_filter != "All":
        where_clauses.append("action_type = %s")
        params.append(action_filter)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        SELECT timestamp, actor, action_type, content, session_id, tenant_id
        FROM memory_episodic
        {where_sql}
        ORDER BY timestamp DESC
        LIMIT %s
    """
    params.append(limit)

    # Ensure connection is alive before executing query
    conn.ping(reconnect=True)
    df = pd.read_sql(query, conn, params=params)

    if df.empty:
        st.info("No thought trace records found. The agent hasn't started processing yet.")
    else:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", len(df))
        col2.metric("Unique Sessions", df["session_id"].nunique())
        col3.metric("Actors", df["actor"].nunique())

        # Data table
        st.subheader("Event Log")
        st.dataframe(df, use_container_width=True, height=400)

        # Timeline (if plotly available)
        try:
            import plotly.express as px

            fig = px.scatter(
                df,
                x="timestamp",
                y="actor",
                color="action_type",
                hover_data=["content"],
                title="Thought Timeline",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("Install plotly for timeline visualization.")

except Exception as e:
    st.error(f"Cannot connect to StarRocks: {e}")
    st.info("Make sure StarRocks is running and the memory schema has been initialized.")
