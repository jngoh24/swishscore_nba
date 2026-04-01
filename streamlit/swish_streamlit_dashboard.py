import os
import gdown
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SwishScore · NBA xP Model",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global dark theme CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* Root palette */
:root {
    --bg:       #0e0f13;
    --bg2:      #161820;
    --bg3:      #1e2028;
    --card:     #1a1c24;
    --border:   rgba(255,255,255,0.07);
    --accent:   #e87c2a;
    --accent2:  #3b82f6;
    --text1:    #f0f0f0;
    --text2:    #9a9aaa;
    --text3:    #5a5a6a;
    --green:    #22c55e;
    --red:      #ef4444;
}

/* Override Streamlit chrome */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text1) !important;
}
[data-testid="stHeader"]          { background: var(--bg2) !important; border-bottom: 1px solid var(--border); }
[data-testid="stSidebar"]         { background: var(--bg2) !important; }
section[data-testid="stSidebar"]  { display: none; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    gap: 4px;
    padding: 0 8px;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text2) !important;
    padding: 12px 16px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
[data-testid="stTabContent"] { background: var(--bg) !important; padding-top: 24px !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"]  { color: var(--text2) !important; font-size: 11px !important; letter-spacing: 0.3px !important; }
[data-testid="stMetricValue"]  { color: var(--text1) !important; font-size: 22px !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"]  { font-size: 11px !important; }

/* Plotly charts background */
.js-plotly-plot .plotly { background: transparent !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: var(--text2) !important; font-size: 12px !important; }

/* Multiselect */
[data-testid="stMultiSelect"] > div { background: var(--bg3) !important; border-color: var(--border) !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* Info / warning boxes */
[data-testid="stInfo"]    { background: rgba(59,130,246,0.08) !important; border-color: rgba(59,130,246,0.2) !important; color: var(--text2) !important; font-size: 11px !important; }

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* General text */
p, li, span, label { color: var(--text2) !important; font-size: 13px !important; }
h1 { color: var(--text1) !important; font-size: 22px !important; font-weight: 600 !important; letter-spacing: -0.5px !important; }
h2 { color: var(--text1) !important; font-size: 17px !important; font-weight: 500 !important; }
h3 { color: var(--text2) !important; font-size: 13px !important; font-weight: 500 !important; letter-spacing: 0.5px !important; text-transform: uppercase !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading with gdown ─────────────────────────────────────────────────────
DATA_DIR = "streamlit/data"
os.makedirs(DATA_DIR, exist_ok=True)

DRIVE_FILES = {
    "shots.csv": "1TjmaUe1fTPNNtB1STGMq9af0JntkvCM0",
    "df.csv":    "11_iTfCLqvOjnR_OmDQnGZMu1-_0WnhQV",
}

@st.cache_data(show_spinner=False)
def load_drive_csv(file_id: str, filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        gdown.download(id=file_id, output=path, quiet=False, fuzzy=True)
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# Load all data with a single spinner block
with st.spinner("Loading SwishScore data..."):
    shots            = load_drive_csv(DRIVE_FILES["shots.csv"], "shots.csv")
    df               = load_drive_csv(DRIVE_FILES["df.csv"],    "df.csv")
    master_xp        = load_local_csv(f"{DATA_DIR}/master_xp.csv")
    teams            = load_local_csv(f"{DATA_DIR}/team_data.csv")
    players_original = load_local_csv(f"{DATA_DIR}/player_data.csv")

# Player preprocessing
players_original["Team"] = players_original["TEAM"].str.upper()
players          = players_original.dropna()
players_filtered = players[players["GP"] >= 50]

# ── Plotly theme helper ─────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9a9aaa", size=11),
    margin=dict(l=12, r=12, t=40, b=80),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.07)",
        tickfont=dict(size=10),
        tickangle=-35,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.07)",
        tickfont=dict(size=10),
    ),
    coloraxis_showscale=False,
    title_font=dict(size=13, color="#f0f0f0"),
    title_x=0,
)

BLUE_SCALE  = ["#1e3a5f", "#1d4ed8", "#3b82f6", "#60a5fa", "#93c5fd"]
GREEN_SCALE = ["#14532d", "#15803d", "#22c55e", "#4ade80", "#86efac"]
RED_SCALE   = ["#7f1d1d", "#b91c1c", "#ef4444", "#f87171", "#fca5a5"]
ORANGE_SEQ  = ["#7c2d12", "#c2410c", "#e87c2a", "#fb923c", "#fdba74"]

# ── Utility chart functions ─────────────────────────────────────────────────────
def dark_bar(df_in, x_col, y_col, title, ascending=False,
             color_scale=None, pct_suffix=False, height=380):
    color_scale = color_scale or BLUE_SCALE
    df_sorted   = df_in.sort_values(y_col, ascending=ascending)
    text_vals   = (df_sorted[y_col].astype(str) + "%") if pct_suffix else df_sorted[y_col]
    fig = px.bar(
        df_sorted, x=x_col, y=y_col,
        text=text_vals,
        color=y_col,
        color_continuous_scale=color_scale,
        height=height,
    )
    fig.update_traces(
        textposition="inside",
        textfont=dict(size=10, color="rgba(255,255,255,0.8)"),
        marker_line_width=0,
    )
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    return fig


def dark_pie(df_in, names_col, values_col, title, height=340):
    fig = px.pie(
        df_in, names=names_col, values=values_col,
        hole=0.45, height=height,
        color_discrete_sequence=["#3b82f6", "#e87c2a", "#22c55e", "#a855f7", "#ef4444"],
    )
    fig.update_traces(
        textinfo="percent+label",
        pull=[0.04] * len(df_in),
        marker=dict(line=dict(color="#0e0f13", width=2)),
    )
    fig.update_layout(**{**PLOTLY_LAYOUT, "margin": dict(l=12, r=12, t=40, b=20)}, title=title)
    return fig


def agg_top(df_in, col, top_n=10):
    counts = df_in[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    counts["pct"]  = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts.head(top_n)


def section(label: str):
    st.markdown(f"<h3>{label}</h3>", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding:4px 0 20px;">
  <div style="width:36px;height:36px;border-radius:50%;background:#e87c2a;
              display:flex;align-items:center;justify-content:center;
              font-weight:700;font-size:16px;color:#fff;">S</div>
  <div>
    <div style="font-size:18px;font-weight:600;color:#f0f0f0;letter-spacing:-0.4px;">SwishScore</div>
    <div style="font-size:11px;color:#5a5a6a;">NBA Shot Outcome Prediction · xP Model &nbsp;·&nbsp;
      <a href="https://github.com/jngoh24/swishscore_nba"
         style="color:#e87c2a;text-decoration:none;">github.com/jngoh24/swishscore_nba</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3 = st.tabs([
    "📈  xP Performance",
    "📊  Shooting Stats",
    "🏀  Team Stats",
    "👤  Player Stats",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 0 · xP PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab0:

    # ── Filters ────────────────────────────────────────────────────────────────
    with st.expander("🔍  Filter", expanded=False):
        unique_teams = sorted(master_xp["TEAM_ABBRV"].unique())
        unique_confs = sorted(master_xp["CONF"].unique())
        unique_divs  = sorted(master_xp["DIVISION"].unique())

        if "xp_reset" not in st.session_state:
            st.session_state.xp_reset = False
        if st.session_state.xp_reset:
            for k, v in [("xp_teams", unique_teams), ("xp_confs", unique_confs), ("xp_divs", unique_divs)]:
                st.session_state[k] = v
            st.session_state.xp_reset = False
            st.rerun()

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sel_teams = st.multiselect("Team",       unique_teams, default=st.session_state.get("xp_teams", unique_teams), key="xp_teams")
        with col_f2:
            sel_confs = st.multiselect("Conference", unique_confs, default=st.session_state.get("xp_confs", unique_confs), key="xp_confs")
        with col_f3:
            sel_divs  = st.multiselect("Division",   unique_divs,  default=st.session_state.get("xp_divs",  unique_divs),  key="xp_divs")

        if st.button("↺  Reset filters"):
            st.session_state.xp_reset = True
            st.rerun()

    # ── Filtered data ───────────────────────────────────────────────────────────
    fxp = master_xp[
        master_xp["TEAM_ABBRV"].isin(sel_teams) &
        master_xp["CONF"].isin(sel_confs) &
        master_xp["DIVISION"].isin(sel_divs)
    ]

    # Team-level xP summary
    tgs = fxp.groupby(["GAME_ID", "TEAM_ABBRV"]).agg(
        total_xP=("xP", "sum"), total_pts=("pts", "sum")
    ).reset_index()
    tgs["over"] = (tgs["total_pts"] > tgs["total_xP"]).map({True: "yes", False: "no"})
    tp = tgs.groupby("TEAM_ABBRV")["over"].value_counts().unstack(fill_value=0)
    tp = tp.rename(columns={"yes": "outperform", "no": "underperform"}).reset_index()
    for c in ["outperform", "underperform"]:
        if c not in tp.columns:
            tp[c] = 0
    tp["total"]          = tp["outperform"] + tp["underperform"]
    tp["outperform_pct"] = (tp["outperform"] / tp["total"] * 100).round()
    tp["underperform_pct"] = (tp["underperform"] / tp["total"] * 100).round()

    # Player-level xP summary
    pgs = fxp.groupby(["GAME_ID", "FULL NAME"]).agg(
        total_xP=("xP", "sum"), total_pts=("pts", "sum")
    ).reset_index()
    pgs["over"] = (pgs["total_pts"] > pgs["total_xP"]).map({True: "yes", False: "no"})
    pp = pgs.groupby("FULL NAME")["over"].value_counts().unstack(fill_value=0)
    pp = pp.rename(columns={"yes": "outperform", "no": "underperform"}).reset_index()
    for c in ["outperform", "underperform"]:
        if c not in pp.columns:
            pp[c] = 0
    pp["total"]          = pp["outperform"] + pp["underperform"]
    pp = pp[pp["total"] >= 10]
    pp["outperform_pct"]   = (pp["outperform"]   / pp["total"] * 100).round()
    pp["underperform_pct"] = (pp["underperform"] / pp["total"] * 100).round()

    top_out_t  = tp.sort_values("outperform_pct",   ascending=False).head(5)
    top_out_p  = pp.sort_values("outperform_pct",   ascending=False).head(5)
    top_und_t  = tp.sort_values("underperform_pct", ascending=False).head(5)
    top_und_p  = pp.sort_values("underperform_pct", ascending=False).head(5)

    # ── KPI row ─────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Teams in view",         len(tp))
    k2.metric("Players (min 10 GP)",   len(pp))
    k3.metric("Avg outperform rate",   f"{tp['outperform_pct'].mean():.0f}%")
    k4.metric("Avg underperform rate", f"{tp['underperform_pct'].mean():.0f}%")

    st.divider()

    # ── Team charts ─────────────────────────────────────────────────────────────
    section("Team xP Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            dark_bar(top_out_t, "TEAM_ABBRV", "outperform_pct",
                     "Top 5 teams · outperformance %", pct_suffix=True,
                     color_scale=GREEN_SCALE),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            dark_bar(top_und_t, "TEAM_ABBRV", "underperform_pct",
                     "Top 5 teams · underperformance %", pct_suffix=True,
                     color_scale=RED_SCALE, ascending=False),
            use_container_width=True,
        )

    # ── Player charts ───────────────────────────────────────────────────────────
    section("Player xP Performance")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            dark_bar(top_out_p, "FULL NAME", "outperform_pct",
                     "Top 5 players · outperformance %", pct_suffix=True,
                     color_scale=GREEN_SCALE),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            dark_bar(top_und_p, "FULL NAME", "underperform_pct",
                     "Top 5 players · underperformance %", pct_suffix=True,
                     color_scale=RED_SCALE, ascending=False),
            use_container_width=True,
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 · SHOOTING STATS
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI row ─────────────────────────────────────────────────────────────────
    total_shots = len(df)
    made        = (df["EVENT_TYPE"].str.lower().str.contains("made")).sum() if "EVENT_TYPE" in df.columns else 0
    missed      = total_shots - made
    fg_pct      = round(made / total_shots * 100, 1) if total_shots else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total shots",    f"{total_shots:,}")
    k2.metric("Made",           f"{made:,}")
    k3.metric("Missed",         f"{missed:,}")
    k4.metric("FG%",            f"{fg_pct}%")

    st.divider()

    # ── Shot outcomes donut + Shots per quarter ─────────────────────────────────
    section("Shot Outcomes & Game Flow")
    c1, c2 = st.columns(2)
    with c1:
        outcome_counts = df["EVENT_TYPE"].value_counts().reset_index()
        outcome_counts.columns = ["EVENT_TYPE", "count"]
        st.plotly_chart(
            dark_pie(outcome_counts, "EVENT_TYPE", "count", "Shot outcomes"),
            use_container_width=True,
        )
    with c2:
        if "QUARTER" in shots.columns:
            q_counts = agg_top(shots, "QUARTER", top_n=10)
            st.plotly_chart(
                dark_bar(q_counts, "QUARTER", "count", "Shots per quarter",
                         color_scale=BLUE_SCALE),
                use_container_width=True,
            )

    st.divider()

    # ── Action types ────────────────────────────────────────────────────────────
    section("Shot Action Types")
    if "ACTION_TYPE" in shots.columns:
        act = agg_top(shots, "ACTION_TYPE", top_n=10)
        st.plotly_chart(
            dark_bar(act, "ACTION_TYPE", "count", "Top 10 shot action types",
                     color_scale=ORANGE_SEQ, height=400),
            use_container_width=True,
        )

    st.divider()

    # ── Zone breakdown ──────────────────────────────────────────────────────────
    section("Zone Breakdown")
    z1, z2, z3 = st.columns(3)
    zone_charts = [
        ("ZONE_NAME",  "Shot distribution by zone name"),
        ("BASIC_ZONE", "Shot distribution by basic zone"),
        ("ZONE_RANGE", "Shot range distribution"),
    ]
    for col_widget, (zone_col, title) in zip([z1, z2, z3], zone_charts):
        with col_widget:
            if zone_col in shots.columns:
                zdata = agg_top(shots, zone_col, top_n=8)
                st.plotly_chart(
                    dark_bar(zdata, zone_col, "count", title,
                             color_scale=BLUE_SCALE, height=360),
                    use_container_width=True,
                )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 · TEAM STATS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:

    # Shot volume
    top10_shots  = shots["TEAM_NAME"].value_counts().head(10).reset_index()
    top10_shots.columns  = ["TEAM_NAME", "count"]
    bot10_shots  = shots["TEAM_NAME"].value_counts().tail(10).reset_index()
    bot10_shots.columns  = ["TEAM_NAME", "count"]

    # Defensive stats
    top10_oppg   = teams[["TEAM", "oPPG"]].drop_duplicates().sort_values("oPPG").head(10)
    top10_deff   = teams[["TEAM", "dEFF"]].drop_duplicates().sort_values("dEFF").head(10)
    bot10_oppg   = teams[["TEAM", "oPPG"]].drop_duplicates().sort_values("oPPG", ascending=False).head(10)
    bot10_deff   = teams[["TEAM", "dEFF"]].drop_duplicates().sort_values("dEFF", ascending=False).head(10)

    # ── KPI row ─────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Most shot attempts",  f"{top10_shots['count'].iloc[0]:,}",  top10_shots["TEAM_NAME"].iloc[0])
    k2.metric("Fewest shot attempts", f"{bot10_shots['count'].iloc[0]:,}", bot10_shots["TEAM_NAME"].iloc[0])
    k3.metric("Best defensive oPPG",  f"{top10_oppg['oPPG'].iloc[0]:.1f}", top10_oppg["TEAM"].iloc[0])
    k4.metric("Best dEFF",            f"{top10_deff['dEFF'].iloc[0]:.1f}", top10_deff["TEAM"].iloc[0])

    st.divider()

    # ── Shot volume ─────────────────────────────────────────────────────────────
    section("Shot Volume by Team")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(dark_bar(top10_shots, "TEAM_NAME", "count",
                                 "Top 10 shot volume teams", color_scale=BLUE_SCALE),
                        use_container_width=True)
    with c2:
        st.plotly_chart(dark_bar(bot10_shots, "TEAM_NAME", "count",
                                 "Bottom 10 shot volume teams",
                                 ascending=True, color_scale=RED_SCALE),
                        use_container_width=True)

    st.divider()

    # ── Defensive stats ─────────────────────────────────────────────────────────
    section("Defensive Efficiency")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(dark_bar(top10_oppg, "TEAM", "oPPG",
                                 "Best defensive oPPG (lower = better)",
                                 ascending=True, color_scale=GREEN_SCALE),
                        use_container_width=True)
        st.plotly_chart(dark_bar(bot10_oppg, "TEAM", "oPPG",
                                 "Worst defensive oPPG",
                                 ascending=False, color_scale=RED_SCALE),
                        use_container_width=True)
    with c4:
        st.plotly_chart(dark_bar(top10_deff, "TEAM", "dEFF",
                                 "Best defensive efficiency (lower = better)",
                                 ascending=True, color_scale=GREEN_SCALE),
                        use_container_width=True)
        st.plotly_chart(dark_bar(bot10_deff, "TEAM", "dEFF",
                                 "Worst defensive efficiency",
                                 ascending=False, color_scale=RED_SCALE),
                        use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 · PLAYER STATS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:

    st.info("Showing players with 50+ games played this season")

    # Prep dataframes
    top10_shot_takers = df["PLAYER_NAME"].value_counts().head(10).reset_index()
    top10_shot_takers.columns = ["PLAYER_NAME", "Shot Attempts"]

    stat_charts = [
        ("eFG%", "Top 10 effective FG% (eFG%)"),
        ("TS%",  "Top 10 true shooting % (TS%)"),
        ("2P%",  "Top 10 two-point FG%"),
        ("3P%",  "Top 10 three-point FG%"),
        ("ORTG", "Top 10 offensive rating (ORTG)"),
    ]

    # ── KPI row ─────────────────────────────────────────────────────────────────
    best = {col: players_filtered.loc[players_filtered[col].idxmax()] for col, _ in stat_charts if col in players_filtered.columns}

    cols = st.columns(len(best))
    for (col, label), widget_col in zip([(c, l) for c, l in stat_charts if c in best], cols):
        row = best[col]
        widget_col.metric(label.replace("Top 10 ", "Best "), f"{row[col]:.1f}%"
                          if "%" in col else f"{row[col]:.1f}", row["FULL NAME"])

    st.divider()

    # ── Shot takers ─────────────────────────────────────────────────────────────
    section("Shot Volume")
    st.plotly_chart(
        dark_bar(top10_shot_takers, "PLAYER_NAME", "Shot Attempts",
                 "Top 10 shot takers", color_scale=ORANGE_SEQ, height=380),
        use_container_width=True,
    )

    st.divider()

    # ── Efficiency stats grid ───────────────────────────────────────────────────
    section("Shooting Efficiency")
    c1, c2 = st.columns(2)
    pairs = list(zip([c1, c2, c1, c2, c1], stat_charts))
    for widget_col, (stat_col, title) in pairs:
        if stat_col not in players_filtered.columns:
            continue
        top_df = players_filtered[["FULL NAME", stat_col]].sort_values(stat_col, ascending=False).head(10)
        with widget_col:
            st.plotly_chart(
                dark_bar(top_df, "FULL NAME", stat_col, title,
                         color_scale=GREEN_SCALE if stat_col != "ORTG" else BLUE_SCALE,
                         pct_suffix=("%" in stat_col), height=380),
                use_container_width=True,
            )

# ── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding:16px 0;border-top:1px solid rgba(255,255,255,0.06);
            text-align:center;font-size:11px;color:#5a5a6a;">
  SwishScore · NBA xP Model · Built by
  <a href="https://github.com/jngoh24/swishscore_nba" style="color:#e87c2a;text-decoration:none;">
    jngoh24
  </a>
</div>
""", unsafe_allow_html=True)
