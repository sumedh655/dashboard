

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------- Load Data ----------------
st.set_page_config(layout="wide", page_title="Indian Mutual Funds Dashboard", page_icon="📊")

@st.cache_data
def load_data():
    """Loads the master summary CSV and performs initial data cleaning."""
    file_path = "master_summary.csv"
    try:
        df = pd.read_csv(file_path)
        # Fill NaN values with 0 for numerical columns to prevent errors
        numerical_cols = ['Initial_NAV', 'Final_NAV', 'Period_Years', 'ROI_abs_pct',
                          'CAGR_since_pct', 'CAGR_1Y_pct', 'CAGR_3Y_pct', 'CAGR_5Y_pct',
                          'CAGR_10Y_pct', 'Ann_Volatility', 'Ann_Return_pct', 'Sharpe',
                          'Max_Drawdown_pct']
        df[numerical_cols] = df[numerical_cols].fillna(0)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.stop()

summary_df = load_data()

# ---------------- Dashboard Title and Filters ----------------
st.title("Indian Mutual Funds Dashboard")
st.markdown("A comprehensive analysis based on consolidated data of over 8,000 mutual funds. 📈")

st.sidebar.header("Filter Options")

# Category filter
selected_category = st.sidebar.multiselect(
    "Select Fund Category", summary_df["Category"].unique(), default=summary_df["Category"].unique()
)

# AMC filter
selected_amc = st.sidebar.multiselect(
    "Select AMC", summary_df["AMC"].unique(), default=[]
)

# Plan filter
selected_plan = st.sidebar.multiselect(
    "Select Plan Type", summary_df["Plan"].unique(), default=[]
)

# Volatility slider to filter by risk level
max_vol = summary_df["Ann_Volatility"].max()
volatility_range = st.sidebar.slider(
    "Filter by Annualized Volatility (Risk)", 0.0, max_vol, (0.0, max_vol)
)

# Apply all filters
filtered_df = summary_df[
    (summary_df["Category"].isin(selected_category)) &
    (summary_df["AMC"].isin(selected_amc) if selected_amc else True) &
    (summary_df["Plan"].isin(selected_plan) if selected_plan else True) &
    (summary_df["Ann_Volatility"] >= volatility_range[0]) &
    (summary_df["Ann_Volatility"] <= volatility_range[1])
]

# ---------------- Main Dashboard Layout ----------------
st.header("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Funds Found", len(filtered_df))
col2.metric("Average 5-Year CAGR", f"{filtered_df['CAGR_5Y_pct'].mean():.2f}%")
col3.metric("Average Sharpe Ratio", f"{filtered_df['Sharpe'].mean():.2f}")
col4.metric("Average Max Drawdown", f"{filtered_df['Max_Drawdown_pct'].mean():.2f}%")

st.markdown("---")

# ---------------- Visualizations Section ----------------
st.header("Visual Analysis")

# Use an expander for charts to keep the dashboard clean
with st.expander("Explore Charts", expanded=True):
    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        # Scatter plot: Risk vs Return
        st.subheader("Risk vs Return (Ann. Return vs Volatility)")
        fig_scatter = px.scatter(
            filtered_df, x="Ann_Volatility", y="Ann_Return_pct",
            size="CAGR_5Y_pct", color="Category", hover_name="Fund_Name",
            title="Annualized Return vs Volatility by Category",
            labels={'Ann_Volatility': 'Annualized Volatility', 'Ann_Return_pct': 'Annualized Return (%)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_vis2:
        # Pie chart: Fund distribution by Category
        st.subheader("Fund Distribution by Category")
        category_counts = filtered_df["Category"].value_counts().reset_index()
        category_counts.columns = ['Category', 'count']
        fig_pie = px.pie(
            category_counts, values='count', names='Category',
            title="Fund Count by Category",
            hover_data=['count'], labels={'count':'Number of Funds'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    col_vis3, col_vis4 = st.columns(2)

    with col_vis3:
        # Bar chart: Average Sharpe Ratio by Category
        st.subheader("Average Sharpe Ratio by Category")
        avg_sharpe = filtered_df.groupby("Category")["Sharpe"].mean().sort_values(ascending=False).reset_index()
        fig_bar_sharpe = px.bar(
            avg_sharpe, x="Category", y="Sharpe", color="Category",
            title="Average Sharpe Ratio by Fund Category",
            labels={'Sharpe': 'Average Sharpe Ratio'}
        )
        st.plotly_chart(fig_bar_sharpe, use_container_width=True)

    with col_vis4:
        # Bar chart: Average CAGR by AMC
        st.subheader("Average 5-Year CAGR by AMC")
        avg_cagr_amc = filtered_df.groupby("AMC")["CAGR_5Y_pct"].mean().sort_values(ascending=False).head(15).reset_index()
        fig_bar_amc = px.bar(
            avg_cagr_amc, x="AMC", y="CAGR_5Y_pct", color="AMC",
            title="Top 15 AMCs by Average 5-Year CAGR",
            labels={'CAGR_5Y_pct': 'Average 5-Year CAGR (%)'}
        )
        st.plotly_chart(fig_bar_amc, use_container_width=True)

# ---------------- Top Performers Table ----------------
st.header("Top Performers Table")
top_n = st.slider("Show Top N Funds", 5, 50, 10, key="top_n_slider")
sort_by = st.selectbox("Sort by", ["CAGR_5Y_pct", "Sharpe", "Ann_Return_pct", "Max_Drawdown_pct"])
top_funds = filtered_df.sort_values(sort_by, ascending=(sort_by != "Max_Drawdown_pct")).head(top_n)
st.dataframe(top_funds[['Fund_Name', 'AMC', 'Category', sort_by, 'Max_Drawdown_pct', 'Ann_Volatility']], use_container_width=True)