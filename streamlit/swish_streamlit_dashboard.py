import os
import gdown
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(
    page_title="SwishScore · NBA xP Model",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Full dark theme matching the HTML mock ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container {
    background-color: #0e0f13 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #f0f0f0 !important;
}
[data-testid="stHeader"]              { background: #161820 !important; border-bottom: 1px solid rgba(255,255,255,0.07) !important; }
[data-testid="stToolbar"]             { background: #161820 !important; }
section[data-testid="stSidebar"]      { display: none !important; }
.block-container                      { padding: 0 2rem 2rem !important; max-width: 100% !important; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: #161820 !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    padding: 0 8px !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #9a9aaa !important;
    padding: 14px 18px 12px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    letter-spacing: 0.2px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #e87c2a !important;
    border-bottom: 2px solid #e87c2a !important;
}
[data-testid="stTabContent"] {
    background: #0e0f13 !important;
    padding-top: 28px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #1a1c24 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
}
[data-testid="stMetricLabel"] p {
    color: #9a9aaa !important;
    font-size: 11px !important;
    letter-spacing: 0.3px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: #f0f0f0 !important;
    font-size: 24px !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #1a1c24 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    margin-bottom: 20px !important;
}
[data-testid="stExpander"] summary {
    color: #9a9aaa !important;
    font-size: 12px !important;
}

/* Info box */
[data-testid="stInfo"] {
    background: rgba(59,130,246,0.08) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
}
[data-testid="stInfo"] p { color: #9a9aaa !important; font-size: 11px !important; }

/* Multiselect */
[data-testid="stMultiSelect"] > div > div {
    background: #1e2028 !important;
    border-color: rgba(255,255,255,0.1) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 20px 0 !important; }

/* Text overrides */
p, span, label, li { color: #9a9aaa !important; font-family: 'DM Sans', sans-serif !important; }
h1 { color: #f0f0f0 !important; font-size: 20px !important; font-weight: 600 !important; }
h2 { color: #f0f0f0 !important; font-size: 16px !important; font-weight: 500 !important; }
h3 { color: #5a5a6a !important; font-size: 10px !important; font-weight: 600 !important;
     letter-spacing: 1.2px !important; text-transform: uppercase !important; margin: 18px 0 12px !important; }

/* iframe cards — remove default border/shadow */
iframe { border: none !important; }
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

# ── Chart / HTML helpers ───────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9a9aaa", size=11),
    margin=dict(l=12, r=12, t=40, b=100),
    coloraxis_showscale=False,
    title_font=dict(size=13, color="#f0f0f0", family="DM Sans"),
    title_x=0,
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color="#9a9aaa"),
        tickangle=-38,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=10, color="#9a9aaa"),
    ),
)

SCALES = {
    "blue":   ["#1e3a5f","#1d4ed8","#3b82f6","#60a5fa","#93c5fd"],
    "green":  ["#14532d","#15803d","#22c55e","#4ade80","#86efac"],
    "red":    ["#7f1d1d","#b91c1c","#ef4444","#f87171","#fca5a5"],
    "orange": ["#7c2d12","#c2410c","#e87c2a","#fb923c","#fdba74"],
    "purple": ["#3b0764","#6d28d9","#a855f7","#c084fc","#e9d5ff"],
}

def dark_bar(df_in, x, y, title, ascending=False, scale="blue", pct=False, height=370):
    d   = df_in.sort_values(y, ascending=ascending)
    txt = (d[y].round(1).astype(str) + "%") if pct else d[y]
    fig = px.bar(d, x=x, y=y, text=txt, color=y,
                 color_continuous_scale=SCALES[scale], height=height)
    fig.update_traces(
        textposition="inside",
        textfont=dict(size=10, color="rgba(255,255,255,0.85)"),
        marker_line_width=0,
    )
    fig.update_layout(**PLOTLY_BASE, title=title)
    return fig

def dark_pie(df_in, names, values, title, height=320):
    fig = px.pie(
        df_in, names=names, values=values, hole=0.42, height=height,
        color_discrete_sequence=["#3b82f6","#ef4444","#22c55e","#a855f7","#e87c2a"],
    )
    fig.update_traces(
        textinfo="percent+label",
        pull=[0.04]*len(df_in),
        marker=dict(line=dict(color="#0e0f13", width=2)),
        textfont=dict(size=11, color="#f0f0f0"),
    )
    layout = {**PLOTLY_BASE, "margin": dict(l=12, r=12, t=40, b=12)}
    fig.update_layout(**layout, title=title)
    return fig

def agg_top(df_in, col, top_n=10):
    c = df_in[col].value_counts().reset_index()
    c.columns = [col, "count"]
    c["pct"] = (c["count"] / c["count"].sum() * 100).round(1)
    return c.head(top_n)

def section(label):
    st.markdown(f"<h3>{label}</h3>", unsafe_allow_html=True)

# ── HTML card builder (rendered via components.html) ──────────────────────────
def build_xp_card_html(title, tag_text, tag_bg, tag_color, rows_html):
    return f"""<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1c24; font-family:'DM Sans',sans-serif; padding:20px; border-radius:10px; }}
</style>
</head><body>
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
    <span style="font-size:13px;font-weight:500;color:#f0f0f0;">{title}</span>
    <span style="font-size:10px;padding:2px 8px;border-radius:4px;font-weight:500;
                 background:{tag_bg};color:{tag_color};">{tag_text}</span>
  </div>
  {rows_html}
</body></html>"""

def build_progress_rows(data, name_col, val_col, bar_color):
    max_val = data[val_col].max() or 1
    html = ""
    for _, row in data.iterrows():
        pct_bar = row[val_col] / max_val * 100
        html += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
          <span style="font-size:12px;color:#f0f0f0;width:120px;flex-shrink:0;
                       white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{row[name_col]}</span>
          <div style="flex:1;height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;">
            <div style="width:{pct_bar:.1f}%;height:100%;background:{bar_color};border-radius:3px;"></div>
          </div>
          <span style="font-size:11px;color:#9a9aaa;font-family:'DM Mono',monospace;
                       width:36px;text-align:right;">{int(row[val_col])}%</span>
        </div>"""
    return html

def xp_card(title, tag_text, tag_bg, tag_color, data, name_col, val_col, bar_color, height=210):
    rows = build_progress_rows(data, name_col, val_col, bar_color)
    html = build_xp_card_html(title, tag_text, tag_bg, tag_color, rows)
    components.html(html, height=height, scrolling=False)

# ── Zone badge grid (HTML) ─────────────────────────────────────────────────────
def zone_badge_grid(df_in, col, title, color):
    counts = df_in[col].value_counts()
    total  = counts.sum()
    cards  = ""
    for zone, cnt in counts.items():
        pct = cnt / total * 100
        bar_w = pct / counts.max() * 100 * total / total  # normalise width
        cards += f"""
        <div style="background:#1a1c24;border:1px solid rgba(255,255,255,0.07);
                    border-radius:8px;padding:12px;text-align:center;">
          <div style="font-size:10px;color:#5a5a6a;text-transform:uppercase;
                      letter-spacing:0.5px;margin-bottom:4px;">{zone}</div>
          <div style="font-size:20px;font-weight:600;color:#f0f0f0;">{pct:.1f}%</div>
          <div style="font-size:10px;color:#5a5a6a;margin-top:2px;">{cnt:,} attempts</div>
          <div style="height:3px;border-radius:2px;margin-top:8px;
                      background:{color};width:{pct / counts.max() * 100:.0f}%;margin-left:auto;margin-right:auto;"></div>
        </div>"""
    html = f"""<!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>*{{margin:0;padding:0;box-sizing:border-box;}}
    body{{background:#0e0f13;font-family:'DM Sans',sans-serif;padding:4px 0;}}
    .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}}
    </style></head><body>
    <div class="grid">{cards}</div>
    </body></html>"""
    n_rows = -(-len(counts) // 3)
    components.html(html, height=n_rows * 110 + 20, scrolling=False)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:#161820;margin:-0px -2rem 0;padding:14px 2rem;
            border-bottom:1px solid rgba(255,255,255,0.07);
            display:flex;align-items:center;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:32px;height:32px;border-radius:50%;background:#e87c2a;
                display:flex;align-items:center;justify-content:center;
                font-weight:700;font-size:14px;color:#fff;flex-shrink:0;">S</div>
    <div>
      <div style="font-size:16px;font-weight:600;color:#f0f0f0;letter-spacing:-0.3px;
                  font-family:'DM Sans',sans-serif;line-height:1.2;">SwishScore</div>
      <div style="font-size:11px;color:#5a5a6a;font-family:'DM Sans',sans-serif;">
        NBA Shot Outcome Prediction &nbsp;·&nbsp; xP Model
      </div>
    </div>
  </div>
  <a href="https://github.com/jngoh24/swishscore_nba"
     style="font-size:11px;color:#9a9aaa;text-decoration:none;
            border:1px solid rgba(255,255,255,0.1);padding:4px 10px;
            border-radius:6px;font-family:'DM Sans',sans-serif;">
    github.com/jngoh24/swishscore_nba
  </a>
</div>
<div style="height:24px;"></div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab0, tab1, tab2, tab3 = st.tabs([
    "📈  xP Performance",
    "📊  Shooting Stats",
    "🏀  Team Stats",
    "👤  Player Stats",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 · xP PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab0:

    # Filters
    with st.expander("🔍  Filter options", expanded=False):
        unique_teams = sorted(master_xp["TEAM_ABBRV"].unique())
        unique_confs = sorted(master_xp["CONF"].unique())
        unique_divs  = sorted(master_xp["DIVISION"].unique())

        if "xp_reset" not in st.session_state:
            st.session_state.xp_reset = False
        if st.session_state.xp_reset:
            for k, v in [("xp_teams", unique_teams),("xp_confs", unique_confs),("xp_divs", unique_divs)]:
                st.session_state[k] = v
            st.session_state.xp_reset = False
            st.rerun()

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_teams = st.multiselect("Team", unique_teams,
                default=st.session_state.get("xp_teams", unique_teams), key="xp_teams")
        with fc2:
            sel_confs = st.multiselect("Conference", unique_confs,
                default=st.session_state.get("xp_confs", unique_confs), key="xp_confs")
        with fc3:
            sel_divs = st.multiselect("Division", unique_divs,
                default=st.session_state.get("xp_divs", unique_divs), key="xp_divs")
        if st.button("↺  Reset filters"):
            st.session_state.xp_reset = True
            st.rerun()

    fxp = master_xp[
        master_xp["TEAM_ABBRV"].isin(sel_teams) &
        master_xp["CONF"].isin(sel_confs) &
        master_xp["DIVISION"].isin(sel_divs)
    ]

    # Team xP aggregation
    tgs = fxp.groupby(["GAME_ID","TEAM_ABBRV"]).agg(
        total_xP=("xP","sum"), total_pts=("pts","sum")).reset_index()
    tgs["over"] = (tgs["total_pts"] > tgs["total_xP"]).map({True:"yes",False:"no"})
    tp = tgs.groupby("TEAM_ABBRV")["over"].value_counts().unstack(fill_value=0)
    tp = tp.rename(columns={"yes":"outperform","no":"underperform"}).reset_index()
    for c in ["outperform","underperform"]:
        if c not in tp.columns: tp[c] = 0
    tp["total"] = tp["outperform"] + tp["underperform"]
    tp["outperform_pct"]   = (tp["outperform"]   / tp["total"] * 100).round()
    tp["underperform_pct"] = (tp["underperform"] / tp["total"] * 100).round()

    # Player xP aggregation
    pgs = fxp.groupby(["GAME_ID","FULL NAME"]).agg(
        total_xP=("xP","sum"), total_pts=("pts","sum")).reset_index()
    pgs["over"] = (pgs["total_pts"] > pgs["total_xP"]).map({True:"yes",False:"no"})
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
    k1.metric("Teams in view",       len(tp))
    k2.metric("Players (min 10 GP)", len(pp))
    k3.metric("Avg outperform rate", f"{tp['outperform_pct'].mean():.0f}%")
    k4.metric("Avg underperform rate", f"{tp['underperform_pct'].mean():.0f}%")

    st.divider()
    section("Team xP Performance")

    c1, c2 = st.columns(2)
    with c1:
        xp_card("Teams · xP Outperformance %", "Top 5",
                "rgba(34,197,94,0.12)", "#22c55e",
                top_out_t, "TEAM_ABBRV", "outperform_pct", "#22c55e")
    with c2:
        xp_card("Teams · xP Underperformance %", "Bottom 5",
                "rgba(239,68,68,0.12)", "#ef4444",
                top_und_t, "TEAM_ABBRV", "underperform_pct", "#ef4444")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section("Player xP Performance")

    c3, c4 = st.columns(2)
    with c3:
        xp_card("Players · xP Outperformance %", "Min 10 games",
                "rgba(34,197,94,0.12)", "#22c55e",
                top_out_p, "FULL NAME", "outperform_pct", "#22c55e")
    with c4:
        xp_card("Players · xP Underperformance %", "Min 10 games",
                "rgba(239,68,68,0.12)", "#ef4444",
                top_und_p, "FULL NAME", "underperform_pct", "#ef4444")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 · SHOOTING STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    total_shots = len(shots)
    made   = (shots["EVENT_TYPE"].str.lower().str.contains("made")).sum() if "EVENT_TYPE" in shots.columns else 0
    missed = total_shots - made
    fg_pct = round(made / total_shots * 100, 1) if total_shots else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total shots tracked", f"{total_shots:,}")
    k2.metric("Made",                f"{made:,}")
    k3.metric("Missed",              f"{missed:,}")
    k4.metric("FG%",                 f"{fg_pct}%")

    st.divider()
    section("Shot Outcomes & Game Flow")

    c1, c2 = st.columns(2)
    with c1:
        if "EVENT_TYPE" in shots.columns:
            oc = shots["EVENT_TYPE"].value_counts().reset_index()
            oc.columns = ["EVENT_TYPE","count"]
            st.plotly_chart(dark_pie(oc,"EVENT_TYPE","count","Shot outcomes (made vs missed)"),
                            use_container_width=True)
    with c2:
        if "QUARTER" in shots.columns:
            st.plotly_chart(dark_bar(agg_top(shots,"QUARTER",10),
                                     "QUARTER","count","Shots per quarter",
                                     scale="blue", height=320),
                            use_container_width=True)

    st.divider()
    section("Shot Action Types")
    if "ACTION_TYPE" in shots.columns:
        st.plotly_chart(dark_bar(agg_top(shots,"ACTION_TYPE",10),
                                 "ACTION_TYPE","count","Top 10 shot action types",
                                 scale="orange", height=400),
                        use_container_width=True)

    st.divider()
    section("Zone Breakdown")
    if "ZONE_NAME" in shots.columns:
        zone_badge_grid(shots, "ZONE_NAME", "By zone name", "#3b82f6")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    z1, z2 = st.columns(2)
    with z1:
        if "BASIC_ZONE" in shots.columns:
            st.plotly_chart(dark_bar(agg_top(shots,"BASIC_ZONE",8),
                                     "BASIC_ZONE","count","Shot distribution by basic zone",
                                     scale="blue", height=340),
                            use_container_width=True)
    with z2:
        if "ZONE_RANGE" in shots.columns:
            st.plotly_chart(dark_bar(agg_top(shots,"ZONE_RANGE",8),
                                     "ZONE_RANGE","count","Shot range distribution",
                                     scale="blue", height=340),
                            use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 · TEAM STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    top10_shots = shots["TEAM_NAME"].value_counts().head(10).reset_index()
    top10_shots.columns = ["TEAM_NAME","count"]
    bot10_shots = shots["TEAM_NAME"].value_counts().tail(10).reset_index()
    bot10_shots.columns = ["TEAM_NAME","count"]
    top10_oppg  = teams[["TEAM","oPPG"]].drop_duplicates().sort_values("oPPG").head(10)
    top10_deff  = teams[["TEAM","dEFF"]].drop_duplicates().sort_values("dEFF").head(10)
    bot10_oppg  = teams[["TEAM","oPPG"]].drop_duplicates().sort_values("oPPG",ascending=False).head(10)
    bot10_deff  = teams[["TEAM","dEFF"]].drop_duplicates().sort_values("dEFF",ascending=False).head(10)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Most attempts",   f"{top10_shots['count'].iloc[0]:,}",  top10_shots['TEAM_NAME'].iloc[0])
    k2.metric("Fewest attempts", f"{bot10_shots['count'].iloc[0]:,}",  bot10_shots['TEAM_NAME'].iloc[0])
    k3.metric("Best def. oPPG",  f"{top10_oppg['oPPG'].iloc[0]:.1f}", top10_oppg['TEAM'].iloc[0])
    k4.metric("Best dEFF",       f"{top10_deff['dEFF'].iloc[0]:.1f}", top10_deff['TEAM'].iloc[0])

    st.divider()
    section("Shot Volume by Team")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(dark_bar(top10_shots,"TEAM_NAME","count",
                                 "Top 10 shot volume teams", scale="blue"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(dark_bar(bot10_shots,"TEAM_NAME","count",
                                 "Bottom 10 shot volume teams",
                                 ascending=True, scale="red"),
                        use_container_width=True)

    st.divider()
    section("Defensive Efficiency")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(dark_bar(top10_oppg,"TEAM","oPPG",
                                 "Best defensive oPPG (lower = better)",
                                 ascending=True, scale="green"),
                        use_container_width=True)
        st.plotly_chart(dark_bar(bot10_oppg,"TEAM","oPPG",
                                 "Worst defensive oPPG",
                                 ascending=False, scale="red"),
                        use_container_width=True)
    with c4:
        st.plotly_chart(dark_bar(top10_deff,"TEAM","dEFF",
                                 "Best defensive efficiency (lower = better)",
                                 ascending=True, scale="green"),
                        use_container_width=True)
        st.plotly_chart(dark_bar(bot10_deff,"TEAM","dEFF",
                                 "Worst defensive efficiency",
                                 ascending=False, scale="red"),
                        use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 · PLAYER STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.info("Showing players with 50+ games played this season")

    shot_takers = shots["PLAYER_NAME"].value_counts().head(10).reset_index()
    shot_takers.columns = ["PLAYER_NAME","Shot Attempts"]

    stat_cols = [
        ("eFG%", "Top 10 effective FG% (eFG%)",    "green"),
        ("TS%",  "Top 10 true shooting % (TS%)",    "green"),
        ("2P%",  "Top 10 two-point FG% (2P%)",      "blue"),
        ("3P%",  "Top 10 three-point FG% (3P%)",    "orange"),
        ("ORTG", "Top 10 offensive rating (ORTG)",   "blue"),
    ]

    # KPI row
    valid_stats = [(c,l,s) for c,l,s in stat_cols if c in players_filtered.columns]
    kpi_cols = st.columns(len(valid_stats))
    for (col, label, _), kc in zip(valid_stats, kpi_cols):
        best = players_filtered.loc[players_filtered[col].idxmax()]
        suffix = "%" if "%" in col else ""
        kc.metric(col, f"{best[col]:.1f}{suffix}", best["FULL NAME"])

    st.divider()
    section("Shot Volume")
    st.plotly_chart(dark_bar(shot_takers,"PLAYER_NAME","Shot Attempts",
                             "Top 10 shot takers this season",
                             scale="orange", height=380),
                    use_container_width=True)

    st.divider()
    section("Shooting Efficiency")
    c1, c2 = st.columns(2)
    sides = [c1, c2, c1, c2, c1]
    for (col, title, scale), side in zip(valid_stats, sides):
        top_df = players_filtered[["FULL NAME",col]].sort_values(col, ascending=False).head(10)
        with side:
            st.plotly_chart(dark_bar(top_df,"FULL NAME",col,title,
                                     scale=scale, pct=("%" in col), height=380),
                            use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding:16px 0;
            border-top:1px solid rgba(255,255,255,0.06);
            text-align:center;font-size:11px;color:#5a5a6a;
            font-family:'DM Sans',sans-serif;">
  SwishScore &nbsp;·&nbsp; NBA xP Model &nbsp;·&nbsp;
  <a href="https://github.com/jngoh24/swishscore_nba"
     style="color:#e87c2a;text-decoration:none;">jngoh24</a>
</div>
""", unsafe_allow_html=True)
