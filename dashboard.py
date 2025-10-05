import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="MF Quality Lab", layout="wide")

# Updated line with the correct relative path
DATA_PATH_DEFAULT = "master_summary.csv"

@st.cache_data(show_spinner=False)
def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in ["Start_Date", "End_Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "Period_Years" in df.columns:
        df["Fund_Age_Yrs"] = df["Period_Years"]
    else:
        if "Start_Date" in df.columns and "End_Date" in df.columns:
            age_days = (df["End_Date"] - df["Start_Date"]).dt.days
            df["Fund_Age_Yrs"] = age_days / 365.25
    if "Max_Drawdown_pct" in df.columns:
        s = df["Max_Drawdown_pct"]
        if s.dropna().mean() < 0:
            df["MaxDD_mag_pct"] = s.abs()
        else:
            df["MaxDD_mag_pct"] = s
    if "Category" in df.columns:
        df["Category"] = df["Category"].astype(str).str.strip()
    if "AMC" in df.columns:
        df["AMC"] = df["AMC"].astype(str).str.strip()
    if "Fund_Name" in df.columns:
        df["Fund_Name"] = df["Fund_Name"].astype(str).str.strip()
    horizons = [c for c in ["CAGR_1Y_pct","CAGR_3Y_pct","CAGR_5Y_pct","CAGR_10Y_pct"] if c in df.columns]
    return df, horizons

def pct_rank(s: pd.Series):
    return s.rank(pct=True, method="average")

def build_quality_scores(dfi: pd.DataFrame, horizons, w):
    sc = pd.DataFrame(index=dfi.index)
    if horizons:
        R = pd.DataFrame({h: pct_rank(dfi[h]) for h in horizons})
        sc["return_strength"] = R.mean(axis=1)
    elif "Ann_Return_pct" in dfi.columns:
        sc["return_strength"] = pct_rank(dfi["Ann_Return_pct"])
    else:
        sc["return_strength"] = 0.0
    if "Ann_Volatility" in dfi.columns:
        sc["low_vol"] = 1 - pct_rank(dfi["Ann_Volatility"])
    else:
        sc["low_vol"] = 0.0
    dd_col = "MaxDD_mag_pct" if "MaxDD_mag_pct" in dfi.columns else ("Max_Drawdown_pct" if "Max_Drawdown_pct" in dfi.columns else None)
    if dd_col is not None:
        sc["low_dd"] = 1 - pct_rank(dfi[dd_col])
    else:
        sc["low_dd"] = 0.0
    if "Sharpe" in dfi.columns:
        sc["sharpe"] = pct_rank(dfi["Sharpe"])
    else:
        sc["sharpe"] = 0.0
    if "Fund_Age_Yrs" in dfi.columns:
        sc["age_bonus"] = pct_rank(dfi["Fund_Age_Yrs"])
    else:
        sc["age_bonus"] = 0.0
    if "CAGR_since_pct" in dfi.columns:
        sc["since_incep"] = pct_rank(dfi["CAGR_since_pct"])
    else:
        sc["since_incep"] = 0.0
    denom = (w["w_return"] + w["w_vol"] + w["w_dd"] + w["w_sharpe"] + w["w_age"] + w["w_since"])
    if denom == 0:
        denom = 1e-9
    sc["quality_score"] = (
        w["w_return"]     * sc["return_strength"] +
        w["w_vol"]        * sc["low_vol"] +
        w["w_dd"]         * sc["low_dd"] +
        w["w_sharpe"]     * sc["sharpe"] +
        w["w_age"]        * sc["age_bonus"] +
        w["w_since"]      * sc["since_incep"]
    ) / denom
    return sc

def category_medians(df, cols, by="Category"):
    med = df.groupby(by)[cols].median(numeric_only=True).reset_index()
    return med

st.sidebar.title("⚙️ Controls")

data_path = st.sidebar.text_input("CSV path", DATA_PATH_DEFAULT)
df, horizons = load_data(data_path)

amcs = sorted(df["AMC"].dropna().unique().tolist()) if "AMC" in df.columns else []
cats = sorted(df["Category"].dropna().unique().tolist()) if "Category" in df.columns else []
types = sorted(df["Type"].dropna().unique().tolist()) if "Type" in df.columns else []

sel_amc = st.sidebar.multiselect("Filter AMC", amcs, default=[])
sel_cat = st.sidebar.multiselect("Filter Category", cats, default=[])
sel_type = st.sidebar.multiselect("Filter Type", types, default=[])

max_age = float(np.nanmax(df.get("Fund_Age_Yrs", pd.Series([0]))))
min_age = st.sidebar.slider("Min Fund Age (yrs)", 0.0, max(0.5, max_age), 0.0, 0.5)

if "Sharpe" in df.columns:
    sh_min = float(df["Sharpe"].min())
    sh_max = float(df["Sharpe"].max())
else:
    sh_min, sh_max = 0.0, 3.0
min_sharpe = st.sidebar.slider("Min Sharpe", sh_min, max(sh_min+0.1, sh_max), sh_min, 0.1)

st.sidebar.markdown("### 🧮 Quality Score Weights")
w = {
    "w_return": st.sidebar.slider("Return strength", 0.0, 5.0, 3.0, 0.1),
    "w_vol":    st.sidebar.slider("Low volatility", 0.0, 5.0, 1.0, 0.1),
    "w_dd":     st.sidebar.slider("Low drawdown", 0.0, 5.0, 2.0, 0.1),
    "w_sharpe": st.sidebar.slider("Sharpe", 0.0, 5.0, 3.0, 0.1),
    "w_age":    st.sidebar.slider("Age bonus", 0.0, 5.0, 0.5, 0.1),
    "w_since":  st.sidebar.slider("Since inception CAGR", 0.0, 5.0, 0.5, 0.1),
}

fil = pd.Series(True, index=df.index)
if sel_amc:
    fil &= df["AMC"].isin(sel_amc)
if sel_cat:
    fil &= df["Category"].isin(sel_cat)
if sel_type:
    fil &= df["Type"].isin(sel_type)
if "Fund_Age_Yrs" in df.columns:
    fil &= (df["Fund_Age_Yrs"].fillna(0) >= min_age)
if "Sharpe" in df.columns:
    fil &= (df["Sharpe"].fillna(-999) >= min_sharpe)

dff = df[fil].copy()

scores = build_quality_scores(dff, horizons, w)
dff = pd.concat([dff.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

st.title("📊 Mutual Fund Quality Lab (Streamlit)")
st.caption("Goes beyond common aggregators with risk-adjusted, consistency, and clustering views.")

tab_overview, tab_quadrant, tab_consistency, tab_drawdown, tab_clusters, tab_compare, tab_table = st.tabs([
    "Overview", "Risk–Return Quadrant", "Consistency", "Drawdown Lens", "Archetype Clusters", "Compare Funds", "Data Table"
])

with tab_overview:
    left, right = st.columns([1,1], gap="large")
    with left:
        st.subheader("Universe Summary")
        n_funds = len(dff)
        st.metric("Funds (filtered)", n_funds)
        if "Category" in dff.columns:
            cat_counts = dff["Category"].value_counts().reset_index(names=["Category","Count"])
            fig = px.bar(cat_counts, x="Category", y="Count", title="Funds per Category")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Category Medians (filtered)")
        cols_med = [c for c in ["CAGR_1Y_pct","CAGR_3Y_pct","CAGR_5Y_pct","CAGR_10Y_pct","Ann_Volatility","Sharpe","MaxDD_mag_pct","Ann_Return_pct"] if c in dff.columns]
        if cols_med and "Category" in dff.columns:
            med = category_medians(dff, cols_med, by="Category")
            st.dataframe(med, use_container_width=True, hide_index=True)
        st.subheader("Top Quality")
        show_cols = [c for c in ["Sharpe","Ann_Volatility","MaxDD_mag_pct","CAGR_5Y_pct","CAGR_3Y_pct","CAGR_10Y_pct"] if c in dff.columns]
        topq = dff.sort_values("quality_score", ascending=False).head(15)[["Fund_Name","AMC","Category","quality_score"] + show_cols]
        st.dataframe(topq.reset_index(drop=True), use_container_width=True, hide_index=True)

with tab_quadrant:
    st.subheader("Risk–Return Quadrant (category-relative median lines)")
    x_opts = [c for c in ["Ann_Volatility","MaxDD_mag_pct"] if c in dff.columns]
    if not x_opts:
        st.info("Need Ann_Volatility or MaxDD_mag_pct column.")
    else:
        x = st.selectbox("X-axis", x_opts, index=0)
        y_candidates = [c for c in ["Ann_Return_pct","CAGR_5Y_pct","CAGR_3Y_pct","CAGR_10Y_pct","CAGR_since_pct"] if c in dff.columns]
        y = st.selectbox("Y-axis", y_candidates, index=0)
        color = "Category" if "Category" in dff.columns else None
        x_med = dff[x].median() if x in dff.columns else None
        y_med = dff[y].median() if y in dff.columns else None
        size = st.selectbox("Bubble size", [c for c in ["MaxDD_mag_pct","Ann_Volatility"] if c in dff.columns] + [None], index=0)
        fig = px.scatter(dff, x=x, y=y, color=color, hover_data=["Fund_Name","AMC","Category"] if {"Fund_Name","AMC","Category"}.issubset(dff.columns) else None,
                         size=size, title=f"{y} vs {x}")
        if x_med is not None and y_med is not None:
            fig.add_hline(y=y_med, line_dash="dash", line_width=1)
            fig.add_vline(x=x_med, line_dash="dash", line_width=1)
        st.plotly_chart(fig, use_container_width=True)
        if x_med is not None and y_med is not None:
            q = pd.DataFrame({"x": dff[x], "y": dff[y]})
            q["quad"] = np.select(
                [
                    (q["x"] <= x_med) & (q["y"] >= y_med),
                    (q["x"] >  x_med) & (q["y"] >= y_med),
                    (q["x"] <= x_med) & (q["y"] <  y_med),
                    (q["x"] >  x_med) & (q["y"] <  y_med),
                ],
                ["Low Risk, High Return", "High Risk, High Return", "Low Risk, Low Return", "High Risk, Low Return"],
                default="Unk"
            )
            st.write(q["quad"].value_counts())

with tab_consistency:
    st.subheader("Multi-Horizon Consistency")
    if len(horizons) >= 2:
        if "Category" in dff.columns:
            out = []
            for cat, g in dff.groupby("Category"):
                ranks = g[horizons].rank(pct=True)
                ranks["Category"] = cat
                ranks["Fund_Name"] = g["Fund_Name"].values
                out.append(ranks)
            ranks = pd.concat(out, axis=0)
            ranks["consistency_score"] = ranks[horizons].mean(axis=1)
            view = ranks[["Fund_Name","Category","consistency_score"] + horizons].sort_values("consistency_score", ascending=False)
            st.dataframe(view.reset_index(drop=True), use_container_width=True, hide_index=True)
            corr = dff[horizons].rank(pct=True).corr(method="spearman")
            figc = px.imshow(corr, text_auto=True, title="Rank Agreement Across Horizons (Spearman)")
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.info("Category column not found. Showing global percentile ranks.")
            ranks = dff[horizons].rank(pct=True)
            st.dataframe(ranks, use_container_width=True, hide_index=True)
    else:
        st.info("Need at least two horizon columns among: CAGR_1Y_pct, CAGR_3Y_pct, CAGR_5Y_pct, CAGR_10Y_pct.")

with tab_drawdown:
    st.subheader("Drawdown vs Compounding")
    if "MaxDD_mag_pct" in dff.columns:
        yopt = [c for c in ["CAGR_5Y_pct","CAGR_10Y_pct","Ann_Return_pct","CAGR_since_pct"] if c in dff.columns]
        if yopt:
            yy = st.selectbox("Compounding metric", yopt, index=0, key="dd_y")
            fig = px.scatter(dff, x="MaxDD_mag_pct", y=yy, color="Category" if "Category" in dff.columns else None,
                             hover_data=["Fund_Name","AMC","Category"] if {"Fund_Name","AMC","Category"}.issubset(dff.columns) else None,
                             title=f"{yy} vs Max Drawdown (magnitude %)")
            st.plotly_chart(fig, use_container_width=True)
            tmp = dff.dropna(subset=["MaxDD_mag_pct", yy]).copy()
            tmp.sort_values(["MaxDD_mag_pct", yy], ascending=[True, False], inplace=True)
            frontier = []
            best = -np.inf
            for _, row in tmp.iterrows():
                if row[yy] > best:
                    frontier.append(row)
                    best = row[yy]
            if frontier:
                f = pd.DataFrame(frontier)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=tmp["MaxDD_mag_pct"], y=tmp[yy], mode="markers", name="Funds", opacity=0.5))
                fig2.add_trace(go.Scatter(x=f["MaxDD_mag_pct"], y=f[yy], mode="lines+markers", name="Frontier"))
                fig2.update_layout(title="Efficiency Frontier: Min Drawdown for Max Compounding")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No compounding metric found to compare with MaxDD.")
    else:
        st.info("MaxDD_mag_pct column not found.")

with tab_clusters:
    st.subheader("Archetype Clusters (k-means)")
    feat_candidates = [c for c in ["CAGR_1Y_pct","CAGR_3Y_pct","CAGR_5Y_pct","CAGR_10Y_pct","Ann_Volatility","MaxDD_mag_pct","Sharpe","Ann_Return_pct","CAGR_since_pct"] if c in dff.columns]
    feats = st.multiselect("Features for clustering", feat_candidates, default=[c for c in ["CAGR_3Y_pct","CAGR_5Y_pct","Ann_Volatility","MaxDD_mag_pct","Sharpe"] if c in feat_candidates])
    k = st.slider("Clusters (k)", 2, 8, 4, 1)
    if feats:
        work = dff.dropna(subset=feats).copy()
        if len(work) >= k:
            X = work[feats].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            work["cluster"] = km.fit_predict(Xs)
            p = PCA(n_components=2, random_state=42)
            coords = p.fit_transform(Xs)
            work["pc1"] = coords[:,0]
            work["pc2"] = coords[:,1]
            fig = px.scatter(work, x="pc1", y="pc2", color="cluster", hover_data=["Fund_Name","AMC","Category"] if {"Fund_Name","AMC","Category"}.issubset(work.columns) else None,
                             title="Clusters (PCA projection)")
            st.plotly_chart(fig, use_container_width=True)
            prof = work.groupby("cluster")[feats].median().reset_index()
            st.write("Cluster median profiles")
            st.dataframe(prof, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough rows after dropping NaNs for the chosen features.")
    else:
        st.info("Pick at least one feature.")

with tab_compare:
    st.subheader("Compare Selected Funds (Radar)")
    fund_options = dff["Fund_Name"].unique().tolist() if "Fund_Name" in dff.columns else []
    default_pick = fund_options[:3] if fund_options else []
    chosen = st.multiselect("Select up to 5 funds", fund_options, default=default_pick)
    radar_feats = [c for c in ["CAGR_3Y_pct","CAGR_5Y_pct","Sharpe","Ann_Volatility","MaxDD_mag_pct","CAGR_10Y_pct"] if c in dff.columns]
    if chosen and radar_feats:
        sub = dff[dff["Fund_Name"].isin(chosen)].dropna(subset=radar_feats).copy()
        if len(sub) >= 1:
            norm = (sub[radar_feats] - dff[radar_feats].min()) / (dff[radar_feats].max() - dff[radar_feats].min() + 1e-9)
            fig = go.Figure()
            for _, row in norm.iterrows():
                fig.add_trace(go.Scatterpolar(r=row[radar_feats].values, theta=radar_feats, fill="toself", name=sub.loc[row.name, "Fund_Name"]))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, title="Radar (normalized to universe)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selected funds have missing values on chosen metrics.")

with tab_table:
    st.subheader("Filtered Data + Scores")
    show_cols = [c for c in ["Scheme_Code","Fund_Name","AMC","Category","Type","Fund_Age_Yrs","Ann_Volatility","Ann_Return_pct","Sharpe","Max_Drawdown_pct","MaxDD_mag_pct","ROI_abs_pct","CAGR_since_pct","CAGR_1Y_pct","CAGR_3Y_pct","CAGR_5Y_pct","CAGR_10Y_pct","quality_score"] if c in dff.columns]
    st.dataframe(dff[show_cols].sort_values("quality_score", ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data + scores (CSV)", data=dff[show_cols].to_csv(index=False).encode("utf-8"), file_name="filtered_with_scores.csv", mime="text/csv")

st.markdown("---")
st.markdown("**Tip:** Tune the weights to reflect your preference for stability (low vol & low drawdown) vs aggressive compounding (returns & Sharpe).")
