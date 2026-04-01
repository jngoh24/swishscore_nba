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

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background-color: #0e0f13 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #f0f0f0 !important;
}
[data-testid="stHeader"] { background: #161820 !important; border-bottom: 1px solid rgba(255,255,255,0.07); }
section[data-testid="stSidebar"] { display: none; }

[data-testid="stTabs"] [role="tablist"] {
    background: #161820;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    gap: 4px; padding: 0 8px;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important; font-weight: 500 !important;
    color: #9a9aaa !important; padding: 12px 18px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important; border-radius: 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #e87c2a !important; border-bottom-color: #e87c2a !important;
}
[data-testid="stTabContent"] { background: #0e0f13 !important; padding-top: 24px !important; }

[data-testid="stMetric"] {
    background: #1a1c24 !important; border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important; padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: #9a9aaa !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #f0f0f0 !important; font-size: 22px !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"]  { font-size: 11px !important; }

[data-testid="stExpander"] {
    background: #1a1c24 !important; border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #9a9aaa !important; font-size: 12px !important; }
[data-testid="stInfo"] {
    background: rgba(59,130,246,0.08) !important;
    border-color: rgba(59,130,246,0.2) !important;
    color: #9a9aaa !important; font-size: 11px !important;
}
div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
p, li, span, label { color: #9a9aaa !important; font-size: 13px !important; }
h1 { color: #f0f0f0 !important; font-size: 22px !important; font-weight: 600 !important; letter-spacing: -0.5px !important; }
h2 { color: #f0f0f0 !important; font-size: 17px !important; font-weight: 500 !important; }
h3 { color: #5a5a6a !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
DATA_DIR = "streamlit/data"
os.makedirs(DATA_DIR, exist_ok=True)

DRIVE_FILES = {
    "shots.csv": "10pqJEeSlc-s3bZEmXahrhspZ3-cOSxKh",
    "df.csv":    "18O8FUhCdcG8Jbr2oyhQGxSigzk9R6Dx6",
}

@st.cache_data(show_spinner=False)
def load_drive_csv(file_id: str, filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
        gdown.download(url=url, output=path, quiet=False, fuzzy=True, use_cookies=False)
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

with st.spinner("Loading SwishScore data..."):
    shots            = load_drive_csv(DRIVE_FILES["shots.csv"], "shots.csv")
    df               = load_drive_csv(DRIVE_FILES["df.csv"],    "df.csv")
    master_xp        = load_local_csv(f"{DATA_DIR}/master_xp.csv")
    teams            = load_local_csv(f"{DATA_DIR}/team_data.csv")
    players_original = load_local_csv(f"{DATA_DIR}/player_data.csv")

players_original["Team"] = players_original["TEAM"].str.upper()
players          = players_original.dropna()
players_filtered = players[players["GP"] >= 50]

# ── Chart helpers ──────────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9a9aaa", size=11),
    margin=dict(l=12, r=12, t=36, b=90),
    coloraxis_showscale=False,
    title_font=dict(size=13, color="#f0f0f0", family="DM Sans"),
    title_x=0,
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)",
               tickfont=dict(size=10), tickangle=-35),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)",
               tickfont=dict(size=10)),
)

SCALES = {
    "blue":   ["#1e3a5f","#1d4ed8","#3b82f6","#60a5fa","#93c5fd"],
    "green":  ["#14532d","#15803d","#22c55e","#4ade80","#86efac"],
    "red":    ["#7f1d1d","#b91c1c","#ef4444","#f87171","#fca5a5"],
    "orange": ["#7c2d12","#c2410c","#e87c2a","#fb923c","#fdba74"],
}

def bar(df_in, x, y, title, ascending=False, scale="blue", pct=False, height=370):
    d = df_in.sort_values(y, ascending=ascending)
    txt = (d[y].astype(str) + "%") if pct else d[y]
    fig = px.bar(d, x=x, y=y, text=txt, color=y,
                 color_continuous_scale=SCALES[scale], height=height)
    fig.update_traces(textposition="inside",
                      textfont=dict(size=10, color="rgba(255,255,255,0.85)"),
                      marker_line_width=0)
    fig.update_layout(**BASE_LAYOUT, title=title)
    return fig

def pie(df_in, names, values, title, height=320):
    fig = px.pie(df_in, names=names, values=values, hole=0.42, height=height,
                 color_discrete_sequence=["#3b82f6","#e87c2a","#22c55e","#a855f7","#ef4444"])
    fig.update_traces(textinfo="percent+label", pull=[0.04]*len(df_in),
                      marker=dict(line=dict(color="#0e0f13", width=2)))
    fig.update_layout(**{**BASE_LAYOUT, "margin": dict(l=12,r=12,t=36,b=12)}, title=title)
    return fig

def progress_bars(data, name_col, val_col, color):
    """Render HTML progress-bar rows like the mock."""
    max_val = data[val_col].max() or 1
    rows = ""
    for _, row in data.iterrows():
        pct_bar = row[val_col] / max_val * 100
        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:11px;">
          <span style="font-size:12px;color:#f0f0f0;width:130px;flex-shrink:0;
                       font-family:'DM Sans',sans-serif;white-space:nowrap;
                       overflow:hidden;text-overflow:ellipsis;">{row[name_col]}</span>
          <div style="flex:1;height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;">
            <div style="width:{pct_bar}%;height:100%;background:{color};border-radius:3px;"></div>
          </div>
          <span style="font-size:11px;color:#9a9aaa;font-family:'DM Mono',monospace;
                       width:36px;text-align:right;">{int(row[val_col])}%</span>
        </div>"""
    return rows

def xp_card(title, tag_text, tag_color, data, name_col, val_col, bar_color):
    inner = progress_bars(data, name_col, val_col, bar_color)
    return f"""
    <div style="background:#1a1c24;border:1px solid rgba(255,255,255,0.07);
                border-radius:10px;padding:20px;height:100%;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
        <span style="font-size:13px;font-weight:500;color:#f0f0f0;">{title}</span>
        <span style="font-size:10px;padding:2px 8px;border-radius:4px;font-weight:500;
                     background:{tag_color[0]};color:{tag_color[1]};">{tag_text}</span>
      </div>
      {inner}
    </div>"""

def section(label):
    st.markdown(f"<h3>{label}</h3>", unsafe_allow_html=True)

def agg_top(df_in, col, top_n=10):
    c = df_in[col].value_counts().reset_index()
    c.columns = [col, "count"]
    c["pct"] = (c["count"] / c["count"].sum() * 100).round(1)
    return c.head(top_n)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding:4px 0 22px;">
  <div style="width:36px;height:36px;border-radius:50%;background:#e87c2a;
              display:flex;align-items:center;justify-content:center;
              font-weight:700;font-size:16px;color:#fff;flex-shrink:0;">S</div>
  <div>
    <div style="font-size:18px;font-weight:600;color:#f0f0f0;letter-spacing:-0.4px;
                font-family:'DM Sans',sans-serif;">SwishScore</div>
    <div style="font-size:11px;color:#5a5a6a;font-family:'DM Sans',sans-serif;">
      NBA Shot Outcome Prediction &nbsp;·&nbsp; xP Model &nbsp;·&nbsp;
      <a href="https://github.com/jngoh24/swishscore_nba"
         style="color:#e87c2a;text-decoration:none;">github.com/jngoh24/swishscore_nba</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3 = st.tabs([
    "📈  xP Performance",
    "📊  Shooting Stats",
    "🏀  Team Stats",
    "👤  Player Stats",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 · xP PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab0:

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

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_teams = st.multiselect("Team",       unique_teams, default=st.session_state.get("xp_teams", unique_teams), key="xp_teams")
        with fc2:
            sel_confs = st.multiselect("Conference", unique_confs, default=st.session_state.get("xp_confs", unique_confs), key="xp_confs")
        with fc3:
            sel_divs  = st.multiselect("Division",   unique_divs,  default=st.session_state.get("xp_divs",  unique_divs),  key="xp_divs")
        if st.button("↺  Reset filters"):
            st.session_state.xp_reset = True
            st.rerun()

    fxp = master_xp[
        master_xp["TEAM_ABBRV"].isin(sel_teams) &
        master_xp["CONF"].isin(sel_confs) &
        master_xp["DIVISION"].isin(sel_divs)
    ]

    # Team summary
    tgs = fxp.groupby(["GAME_ID","TEAM_ABBRV"]).agg(
        total_xP=("xP","sum"), total_pts=("pts","sum")).reset_index()
    tgs["over"] = (tgs["total_pts"] > tgs["total_xP"]).map({True:"yes", False:"no"})
    tp = tgs.groupby("TEAM_ABBRV")["over"].value_counts().unstack(fill_value=0)
    tp = tp.rename(columns={"yes":"outperform","no":"underperform"}).reset_index()
    for c in ["outperform","underperform"]:
        if c not in tp.columns: tp[c] = 0
    tp["total"] = tp["outperform"] + tp["underperform"]
    tp["outperform_pct"]   = (tp["outperform"]   / tp["total"] * 100).round()
    tp["underperform_pct"] = (tp["underperform"] / tp["total"] * 100).round()

    # Player summary
    pgs = fxp.groupby(["GAME_ID","FULL NAME"]).agg(
        total_xP=("xP","sum"), total_pts=("pts","sum")).reset_index()
    pgs["over"] = (pgs["total_pts"] > pgs["total_xP"]).map({True:"yes", False:"no"})
    pp = pgs.groupby("FULL NAME")["over"].value_counts().unstack(fill_value=0)
    pp = pp.rename(columns={"yes":"outperform","no":"underperform"}).reset_index()
    for c in ["outperform","underperform"]:
        if c not in pp.columns: pp[c] = 0
    pp["total"] = pp["outperform"] + pp["underperform"]
    pp = pp[pp["total"] >= 10]
    pp["outperform_pct"]   = (pp["outperform"]   / pp["total"] * 100).round()
    pp["underperform_pct"] = (pp["underperform"] / pp["total"] * 100).round()

    top_out_t  = tp.sort_values("outperform_pct",   ascending=False).head(5)
    top_out_p  = pp.sort_values("outperform_pct",   ascending=False).head(5)
    top_und_t  = tp.sort_values("underperform_pct", ascending=False).head(5)
    top_und_p  = pp.sort_values("underperform_pct", ascending=False).head(5)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Teams in view",         len(tp))
    k2.metric("Players (min 10 GP)",   len(pp))
    k3.metric("Avg outperform rate",   f"{tp['outperform_pct'].mean():.0f}%")
    k4.metric("Avg underperform rate", f"{tp['underperform_pct'].mean():.0f}%")

    st.divider()
    section("Team xP Performance")

    # Progress-bar cards like the mock
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(xp_card(
            "Teams · xP Outperformance %",
            "Top 5", ("rgba(34,197,94,0.12)", "#22c55e"),
            top_out_t, "TEAM_ABBRV", "outperform_pct", "#22c55e"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(xp_card(
            "Teams · xP Underperformance %",
            "Bottom 5", ("rgba(239,68,68,0.12)", "#ef4444"),
            top_und_t, "TEAM_ABBRV", "underperform_pct", "#ef4444"
        ), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    section("Player xP Performance")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(xp_card(
            "Players · xP Outperformance %",
            "Min 10 games", ("rgba(34,197,94,0.12)", "#22c55e"),
            top_out_p, "FULL NAME", "outperform_pct", "#22c55e"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(xp_card(
            "Players · xP Underperformance %",
            "Min 10 games", ("rgba(239,68,68,0.12)", "#ef4444"),
            top_und_p, "FULL NAME", "underperform_pct", "#ef4444"
        ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · SHOOTING STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    total_shots = len(df)
    made   = (df["EVENT_TYPE"].str.lower().str.contains("made")).sum() if "EVENT_TYPE" in df.columns else 0
    missed = total_shots - made
    fg_pct = round(made / total_shots * 100, 1) if total_shots else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total shots",  f"{total_shots:,}")
    k2.metric("Made",         f"{made:,}")
    k3.metric("Missed",       f"{missed:,}")
    k4.metric("FG%",          f"{fg_pct}%")

    st.divider()
    section("Shot Outcomes & Game Flow")

    c1, c2 = st.columns(2)
    with c1:
        oc = df["EVENT_TYPE"].value_counts().reset_index()
        oc.columns = ["EVENT_TYPE","count"]
        st.plotly_chart(pie(oc, "EVENT_TYPE", "count", "Shot outcomes"), use_container_width=True)
    with c2:
        if "QUARTER" in shots.columns:
            st.plotly_chart(bar(agg_top(shots,"QUARTER",10), "QUARTER","count",
                               "Shots per quarter", scale="blue"), use_container_width=True)

    st.divider()
    section("Shot Action Types")
    if "ACTION_TYPE" in shots.columns:
        st.plotly_chart(bar(agg_top(shots,"ACTION_TYPE",10), "ACTION_TYPE","count",
                           "Top 10 shot action types", scale="orange", height=400),
                       use_container_width=True)

    st.divider()
    section("Zone Breakdown")
    z1, z2, z3 = st.columns(3)
    for col_w, (zone_col, title) in zip(
        [z1, z2, z3],
        [("ZONE_NAME","By zone name"),("BASIC_ZONE","By basic zone"),("ZONE_RANGE","By range")]
    ):
        if zone_col in shots.columns:
            with col_w:
                st.plotly_chart(bar(agg_top(shots,zone_col,8), zone_col,"count",
                                   title, scale="blue", height=360),
                               use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · TEAM STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    top10_shots  = shots["TEAM_NAME"].value_counts().head(10).reset_index()
    top10_shots.columns = ["TEAM_NAME","count"]
    bot10_shots  = shots["TEAM_NAME"].value_counts().tail(10).reset_index()
    bot10_shots.columns = ["TEAM_NAME","count"]
    top10_oppg   = teams[["TEAM","oPPG"]].drop_duplicates().sort_values("oPPG").head(10)
    top10_deff   = teams[["TEAM","dEFF"]].drop_duplicates().sort_values("dEFF").head(10)
    bot10_oppg   = teams[["TEAM","oPPG"]].drop_duplicates().sort_values("oPPG",ascending=False).head(10)
    bot10_deff   = teams[["TEAM","dEFF"]].drop_duplicates().sort_values("dEFF",ascending=False).head(10)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Most attempts",   f"{top10_shots['count'].iloc[0]:,}",  top10_shots['TEAM_NAME'].iloc[0])
    k2.metric("Fewest attempts", f"{bot10_shots['count'].iloc[0]:,}",  bot10_shots['TEAM_NAME'].iloc[0])
    k3.metric("Best def. oPPG",  f"{top10_oppg['oPPG'].iloc[0]:.1f}", top10_oppg['TEAM'].iloc[0])
    k4.metric("Best dEFF",       f"{top10_deff['dEFF'].iloc[0]:.1f}", top10_deff['TEAM'].iloc[0])

    st.divider()
    section("Shot Volume by Team")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar(top10_shots,"TEAM_NAME","count","Top 10 shot volume teams",
                           scale="blue"), use_container_width=True)
    with c2:
        st.plotly_chart(bar(bot10_shots,"TEAM_NAME","count","Bottom 10 shot volume teams",
                           ascending=True, scale="red"), use_container_width=True)

    st.divider()
    section("Defensive Efficiency")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(bar(top10_oppg,"TEAM","oPPG","Best defensive oPPG (lower = better)",
                           ascending=True, scale="green"), use_container_width=True)
        st.plotly_chart(bar(bot10_oppg,"TEAM","oPPG","Worst defensive oPPG",
                           ascending=False, scale="red"), use_container_width=True)
    with c4:
        st.plotly_chart(bar(top10_deff,"TEAM","dEFF","Best defensive efficiency (lower = better)",
                           ascending=True, scale="green"), use_container_width=True)
        st.plotly_chart(bar(bot10_deff,"TEAM","dEFF","Worst defensive efficiency",
                           ascending=False, scale="red"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · PLAYER STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.info("Showing players with 50+ games played this season")

    top10_shot_takers = df["PLAYER_NAME"].value_counts().head(10).reset_index()
    top10_shot_takers.columns = ["PLAYER_NAME","Shot Attempts"]

    stat_cols = [
        ("eFG%", "Top 10 effective FG% (eFG%)",         "green"),
        ("TS%",  "Top 10 true shooting % (TS%)",         "green"),
        ("2P%",  "Top 10 two-point FG% (2P%)",           "blue"),
        ("3P%",  "Top 10 three-point FG% (3P%)",         "orange"),
        ("ORTG", "Top 10 offensive rating (ORTG)",        "blue"),
    ]

    # KPI row — best player per stat
    kpi_cols = st.columns(len(stat_cols))
    for (col, label, _), kc in zip(stat_cols, kpi_cols):
        if col in players_filtered.columns:
            best_row = players_filtered.loc[players_filtered[col].idxmax()]
            suffix = "%" if "%" in col else ""
            kc.metric(col, f"{best_row[col]:.1f}{suffix}", best_row["FULL NAME"])

    st.divider()
    section("Shot Volume")
    st.plotly_chart(bar(top10_shot_takers,"PLAYER_NAME","Shot Attempts",
                       "Top 10 shot takers", scale="orange", height=380),
                   use_container_width=True)

    st.divider()
    section("Shooting Efficiency")
    c1, c2 = st.columns(2)
    sides = [c1, c2, c1, c2, c1]
    for (col, title, scale), side in zip(stat_cols, sides):
        if col not in players_filtered.columns:
            continue
        top_df = players_filtered[["FULL NAME", col]].sort_values(col, ascending=False).head(10)
        with side:
            st.plotly_chart(bar(top_df,"FULL NAME",col,title,
                               scale=scale, pct=("%" in col), height=380),
                           use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding:16px 0;border-top:1px solid rgba(255,255,255,0.06);
            text-align:center;font-size:11px;color:#5a5a6a;font-family:'DM Sans',sans-serif;">
  SwishScore &nbsp;·&nbsp; NBA xP Model &nbsp;·&nbsp;
  <a href="https://github.com/jngoh24/swishscore_nba"
     style="color:#e87c2a;text-decoration:none;">jngoh24</a>
</div>
""", unsafe_allow_html=True)
