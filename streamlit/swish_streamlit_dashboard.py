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
    initial_sidebar_state="expanded",
)

# ── CSS — matches HSR dashboard aesthetic ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    color: #1a1a1a;
}
.stApp { background-color: #f7f7f5; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e5e3;
}
section[data-testid="stSidebar"] * { color: #1a1a1a !important; }

/* Headers */
h1 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 600; font-size: 28px; color: #111111;
    letter-spacing: -0.01em; line-height: 1.2;
}
h2, h3, h4 {
    font-family: 'Inter', sans-serif;
    font-weight: 600; color: #111111; letter-spacing: -0.01em;
}
h3 { font-size: 16px; }
h4 { font-size: 14px; font-weight: 500; color: #444; }

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e5e5e3;
    border-radius: 4px;
    padding: 16px 20px;
    box-shadow: none;
}
[data-testid="metric-container"] label {
    font-size: 11px; font-weight: 500; color: #888 !important;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-family: 'Inter', sans-serif;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px; font-weight: 600; color: #111 !important;
    font-family: 'Source Serif 4', serif;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 2px solid #e5e5e3;
    gap: 0; padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent; color: #888;
    font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500;
    border-radius: 0; border-bottom: 2px solid transparent;
    margin-bottom: -2px; padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: #111 !important;
    border-bottom: 2px solid #111 !important;
}

/* Misc */
hr { border-color: #e5e5e3; }
[data-testid="stDataFrame"] {
    border: 1px solid #e5e5e3; border-radius: 4px; background: #fff;
}
[data-testid="stMultiSelect"] label, [data-testid="stSlider"] label {
    font-size: 12px; font-weight: 500; color: #444 !important;
    text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e5e5e3 !important;
    border-radius: 4px !important;
}

/* Utility classes */
.eyebrow {
    font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600;
    color: #888; text-transform: uppercase; letter-spacing: 0.08em;
}
.kicker {
    font-family: 'Source Serif 4', serif; font-size: 14px;
    font-style: italic; color: #555;
}
.stat-label {
    font-family: 'Inter', sans-serif; font-size: 11px; color: #888;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.mono { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

/* Player card */
.player-card {
    background: #ffffff; border: 1px solid #e5e5e3; border-radius: 4px;
    padding: 16px; margin-bottom: 0;
}
.player-card-name {
    font-family: 'Source Serif 4', serif; font-size: 18px;
    font-weight: 600; color: #111; margin: 0 0 2px 0;
}
.player-card-team {
    font-family: 'Inter', sans-serif; font-size: 11px;
    font-weight: 600; color: #888; text-transform: uppercase;
    letter-spacing: 0.06em; margin: 0 0 12px 0;
}
.player-stat-row {
    display: flex; justify-content: space-between;
    border-top: 1px solid #f0f0ee; padding: 6px 0;
}
.player-stat-label {
    font-family: 'Inter', sans-serif; font-size: 11px; color: #888;
}
.player-stat-value {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    font-weight: 500; color: #111;
}
.badge-good {
    display:inline-block; background:#e8f5e9; color:#2e7d32;
    font-size:11px; font-weight:600; padding:2px 8px;
    border-radius:3px; font-family:'Inter',sans-serif;
}
.badge-bad {
    display:inline-block; background:#fce4e4; color:#c62828;
    font-size:11px; font-weight:600; padding:2px 8px;
    border-radius:3px; font-family:'Inter',sans-serif;
}
.badge-neutral {
    display:inline-block; background:#f5f5f5; color:#666;
    font-size:11px; font-weight:600; padding:2px 8px;
    border-radius:3px; font-family:'Inter',sans-serif;
}
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
def load_drive_csv(file_id, filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
        gdown.download(url=url, output=path, quiet=False, fuzzy=True, use_cookies=False)
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_local_csv(path):
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

# ── Plotly theme — matches HSR base_layout ────────────────────────────────────
PLOT_BG    = "#ffffff"
PAPER_BG   = "#f7f7f5"
GRID_COLOR = "#eeeeec"
TEXT_COLOR = "#666666"
ACCENT     = "#1a4b8c"
GREEN      = "#1a6b3c"
RED        = "#c0392b"
AMBER      = "#b7791f"
ORANGE     = "#c2410c"

def base_layout(title="", height=400, xaxis=None, yaxis=None, margin=None):
    default_axis = dict(gridcolor=GRID_COLOR, showline=False, zeroline=False)
    x = {**default_axis, **(xaxis or {})}
    y = {**default_axis, **(yaxis or {})}
    return dict(
        title=dict(text=title, font=dict(family="Inter", size=13, color="#111111")),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Inter", color=TEXT_COLOR, size=11),
        height=height,
        margin=margin or dict(l=40, r=20, t=44, b=60),
        xaxis=x,
        yaxis=y,
        coloraxis_showscale=False,
        showlegend=False,
    )

# ── Chart helpers ──────────────────────────────────────────────────────────────
def hbar(df_in, y_col, x_col, title, color=ACCENT, height=380, ascending=True, pct=False):
    d = df_in.sort_values(x_col, ascending=ascending)
    suffix = "%" if pct else ""
    norm = (d[x_col] - d[x_col].min()) / (d[x_col].max() - d[x_col].min() + 1e-9)
    # Build color gradient from pale to full accent
    import plotly.express as px
    colors = px.colors.sample_colorscale(
        [[0,"#ddeeff"],[1,color]] if color == ACCENT
        else [[0,"#e8f5e9"],[1,GREEN]] if color == GREEN
        else [[0,"#fde8e8"],[1,RED]],
        norm.tolist()
    )
    fig = go.Figure(go.Bar(
        x=d[x_col], y=d[y_col], orientation="h",
        marker=dict(color=colors, line=dict(width=0), opacity=0.85),
        text=[f"{v:.1f}{suffix}" for v in d[x_col]],
        textposition="outside",
        textfont=dict(size=10, color="#444"),
        hovertemplate=f"%{{y}}: %{{x:.1f}}{suffix}<extra></extra>",
    ))
    fig.update_layout(**base_layout(title, height=height,
        yaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False,
                   autorange="reversed", tickfont=dict(size=10)),
        xaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False,
                   tickfont=dict(size=10)),
    ))
    return fig

def lollipop(df_in, y_col, x_col, title, color=ACCENT, height=380, ascending=True, pct=False):
    d = df_in.sort_values(x_col, ascending=ascending)
    suffix = "%" if pct else ""
    fig = go.Figure()
    for _, row in d.iterrows():
        fig.add_shape(type="line",
            x0=0, x1=row[x_col], y0=row[y_col], y1=row[y_col],
            line=dict(color=color, width=1.5), opacity=0.3)
    fig.add_trace(go.Scatter(
        x=d[x_col], y=d[y_col], mode="markers+text",
        marker=dict(color=color, size=9, opacity=0.85,
                    line=dict(color="#ffffff", width=1.5)),
        text=[f"{v:.1f}{suffix}" for v in d[x_col]],
        textposition="middle right",
        textfont=dict(size=10, color="#444"),
        hovertemplate=f"%{{y}}: %{{x:.1f}}{suffix}<extra></extra>",
    ))
    fig.update_layout(**base_layout(title, height=height,
        yaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False,
                   categoryorder="array", categoryarray=list(d[y_col]),
                   tickfont=dict(size=10)),
        xaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False,
                   tickfont=dict(size=10)),
    ))
    return fig

def donut(df_in, names, values, title, height=320):
    fig = px.pie(df_in, names=names, values=values, hole=0.5, height=height,
                 color_discrete_sequence=[ACCENT, RED, GREEN, AMBER, "#888"])
    fig.update_traces(
        textinfo="percent+label", pull=[0.02]*len(df_in),
        marker=dict(line=dict(color=PAPER_BG, width=2)),
        textfont=dict(size=11, color="#111"),
    )
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Inter", color=TEXT_COLOR, size=11),
        title=dict(text=title, font=dict(family="Inter", size=13, color="#111"), x=0),
        margin=dict(l=12, r=12, t=44, b=12),
        showlegend=True, height=height,
        coloraxis_showscale=False,
        legend=dict(font=dict(size=10, color="#666"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig

def vbar(df_in, x_col, y_col, title, color=ACCENT, height=320):
    d = df_in.sort_values(x_col)
    fig = go.Figure(go.Bar(
        x=d[x_col].astype(str), y=d[y_col],
        marker=dict(color=color, opacity=0.8, line=dict(width=0)),
        text=[f"{v:,}" for v in d[y_col]],
        textposition="outside",
        textfont=dict(size=10, color="#444"),
    ))
    fig.update_layout(**base_layout(title, height=height,
        xaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False, tickfont=dict(size=10)),
    ))
    return fig

def scatter_chart(df_in, x_col, y_col, name_col, title, color=ACCENT, height=420):
    fig = px.scatter(df_in, x=x_col, y=y_col, text=name_col, height=height,
                     color=y_col,
                     color_continuous_scale=[[0,"#ddeeff"],[0.5,ACCENT],[1,GREEN]])
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=8, color="#888"),
        marker=dict(size=8, opacity=0.8, line=dict(color="#fff", width=1)),
    )
    fig.update_layout(**base_layout(title, height=height,
        xaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False,
                   title=dict(text=x_col, font=dict(size=11, color="#888")),
                   tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False,
                   title=dict(text=y_col, font=dict(size=11, color="#888")),
                   tickfont=dict(size=10)),
    ))
    return fig

def agg_top(df_in, col, top_n=10):
    c = df_in[col].value_counts().reset_index()
    c.columns = [col, "count"]
    c["pct"] = (c["count"] / c["count"].sum() * 100).round(1)
    return c.head(top_n)

def section(label):
    st.markdown(f"### {label}")

def eyebrow(text):
    st.markdown(f'<p class="eyebrow">{text}</p>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="eyebrow" style="margin-bottom:2px;">SwishScore</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Inter;font-size:12px;color:#666;margin:0 0 16px 0;">NBA xP Prediction Model</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### Filters")

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

    sel_teams = st.multiselect("Teams", unique_teams,
        default=st.session_state.get("xp_teams", unique_teams), key="xp_teams")
    sel_confs = st.multiselect("Conference", unique_confs,
        default=st.session_state.get("xp_confs", unique_confs), key="xp_confs")
    sel_divs  = st.multiselect("Division", unique_divs,
        default=st.session_state.get("xp_divs", unique_divs),  key="xp_divs")

    min_games_xp = st.slider("Min games (xP)", 1, 82, 10)

    if st.button("↺  Reset filters"):
        st.session_state.xp_reset = True
        st.rerun()

    st.divider()
    st.markdown('<p style="font-family:Inter;font-size:11px;color:#888;line-height:1.6;">'
                '<strong>xP outperformance</strong> — % of games where actual pts > expected pts.<br><br>'
                '<strong>Avg delta</strong> — mean (actual − xP) per game.<br><br>'
                '<strong>Delta %</strong> — delta normalised by xP baseline.</p>',
                unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
eyebrow("NBA · SwishScore xP Prediction Model")
st.markdown('<h1 style="margin:0 0 4px 0;">Expected Points Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="kicker">Shot-level xP model measuring how often teams and players outscore their expected output.</p>', unsafe_allow_html=True)
st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab0, tab_player, tab_shoot, tab_team = st.tabs([
    "📈  xP Performance",
    "👤  Player Stats",
    "📊  Shooting Stats",
    "🏀  Team Stats",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 · xP PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab0:

    fxp = master_xp[
        master_xp["TEAM_ABBRV"].isin(sel_teams) &
        master_xp["CONF"].isin(sel_confs) &
        master_xp["DIVISION"].isin(sel_divs)
    ]

    # ── Team aggregation ───────────────────────────────────────────────────────
    tgs = fxp.groupby(["GAME_ID","TEAM_ABBRV"]).agg(
        total_xP=("xP","sum"), total_pts=("pts","sum")).reset_index()
    tgs["over"]      = (tgs["total_pts"] > tgs["total_xP"]).map({True:"yes",False:"no"})
    tgs["delta"]     = tgs["total_pts"] - tgs["total_xP"]
    tgs["delta_pct"] = (tgs["delta"] / tgs["total_xP"].replace(0, float("nan")) * 100)

    tp = tgs.groupby("TEAM_ABBRV")["over"].value_counts().unstack(fill_value=0)
    tp = tp.rename(columns={"yes":"outperform","no":"underperform"}).reset_index()
    for c in ["outperform","underperform"]:
        if c not in tp.columns: tp[c] = 0
    tp["total"] = tp["outperform"] + tp["underperform"]
    tp["outperform_pct"]   = (tp["outperform"]   / tp["total"] * 100).round()
    tp["underperform_pct"] = (tp["underperform"] / tp["total"] * 100).round()
    tp_delta = tgs.groupby("TEAM_ABBRV").agg(
        avg_delta=("delta","mean"), avg_delta_pct=("delta_pct","mean")).reset_index()
    tp_delta["avg_delta"]     = tp_delta["avg_delta"].round(1)
    tp_delta["avg_delta_pct"] = tp_delta["avg_delta_pct"].round(1)
    tp = tp.merge(tp_delta, on="TEAM_ABBRV", how="left")

    # ── Player aggregation ─────────────────────────────────────────────────────
    pgs = fxp.groupby(["GAME_ID","FULL NAME"]).agg(
        total_xP=("xP","sum"), total_pts=("pts","sum")).reset_index()
    pgs["over"]      = (pgs["total_pts"] > pgs["total_xP"]).map({True:"yes",False:"no"})
    pgs["delta"]     = pgs["total_pts"] - pgs["total_xP"]
    pgs["delta_pct"] = (pgs["delta"] / pgs["total_xP"].replace(0, float("nan")) * 100)

    pp = pgs.groupby("FULL NAME")["over"].value_counts().unstack(fill_value=0)
    pp = pp.rename(columns={"yes":"outperform","no":"underperform"}).reset_index()
    for c in ["outperform","underperform"]:
        if c not in pp.columns: pp[c] = 0
    pp["total"] = pp["outperform"] + pp["underperform"]
    pp = pp[pp["total"] >= min_games_xp]
    pp["outperform_pct"]   = (pp["outperform"]   / pp["total"] * 100).round()
    pp["underperform_pct"] = (pp["underperform"] / pp["total"] * 100).round()
    pp_delta = pgs.groupby("FULL NAME").agg(
        avg_delta=("delta","mean"), avg_delta_pct=("delta_pct","mean")).reset_index()
    pp_delta["avg_delta"]     = pp_delta["avg_delta"].round(1)
    pp_delta["avg_delta_pct"] = pp_delta["avg_delta_pct"].round(1)
    pp = pp.merge(pp_delta, on="FULL NAME", how="left")

    top_out_t  = tp.sort_values("outperform_pct",   ascending=False).head(5)
    top_out_p  = pp.sort_values("outperform_pct",   ascending=False).head(5)
    top_und_t  = tp.sort_values("underperform_pct", ascending=False).head(5)
    top_und_p  = pp.sort_values("underperform_pct", ascending=False).head(5)

    # ── KPI row ────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Teams in view",         len(tp))
    k2.metric("Players (filtered)",    len(pp))
    k3.metric("Avg outperform rate",   f"{tp['outperform_pct'].mean():.0f}%")
    k4.metric("Avg underperform rate", f"{tp['underperform_pct'].mean():.0f}%")
    best_td = tp.loc[tp["avg_delta"].idxmax()]
    k5.metric("Best team avg Δ",       f"+{best_td['avg_delta']:.1f} pts", best_td["TEAM_ABBRV"])
    best_pd = pp.loc[pp["avg_delta"].idxmax()]
    k6.metric("Best player avg Δ",     f"+{best_pd['avg_delta']:.1f} pts", best_pd["FULL NAME"])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── xP outperformance % ────────────────────────────────────────────────────
    st.divider()
    section("xP Outperformance Rate")
    st.markdown('<p class="kicker">% of games in which actual pts exceeded expected pts (xP)</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(hbar(top_out_t, "TEAM_ABBRV", "outperform_pct",
                             "Teams — top outperformance %",
                             color=GREEN, height=300, ascending=True, pct=True),
                        width='stretch')
    with c2:
        st.plotly_chart(hbar(top_out_p, "FULL NAME", "outperform_pct",
                             "Players — top outperformance %",
                             color=GREEN, height=300, ascending=True, pct=True),
                        width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(hbar(top_und_t, "TEAM_ABBRV", "underperform_pct",
                             "Teams — top underperformance %",
                             color=RED, height=300, ascending=True, pct=True),
                        width='stretch')
    with c4:
        st.plotly_chart(hbar(top_und_p, "FULL NAME", "underperform_pct",
                             "Players — top underperformance %",
                             color=RED, height=300, ascending=True, pct=True),
                        width='stretch')

    # ── Avg delta ──────────────────────────────────────────────────────────────
    st.divider()
    section("Avg Points Above / Below xP per Game")
    st.markdown('<p class="kicker">Mean of (actual pts − xP) per game. Positive = consistently scores above shot quality.</p>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(hbar(tp.nlargest(5,"avg_delta")[["TEAM_ABBRV","avg_delta"]],
                             "TEAM_ABBRV", "avg_delta", "Teams — best avg Δ pts",
                             color=GREEN, height=280, ascending=True),
                        width='stretch')
    with d2:
        st.plotly_chart(hbar(pp.nlargest(5,"avg_delta")[["FULL NAME","avg_delta"]],
                             "FULL NAME", "avg_delta", "Players — best avg Δ pts",
                             color=GREEN, height=280, ascending=True),
                        width='stretch')

    d3, d4 = st.columns(2)
    with d3:
        st.plotly_chart(hbar(tp.nsmallest(5,"avg_delta")[["TEAM_ABBRV","avg_delta"]],
                             "TEAM_ABBRV", "avg_delta", "Teams — worst avg Δ pts",
                             color=RED, height=280, ascending=False),
                        width='stretch')
    with d4:
        st.plotly_chart(hbar(pp.nsmallest(5,"avg_delta")[["FULL NAME","avg_delta"]],
                             "FULL NAME", "avg_delta", "Players — worst avg Δ pts",
                             color=RED, height=280, ascending=False),
                        width='stretch')

    # ── Delta % ────────────────────────────────────────────────────────────────
    st.divider()
    section("Avg xP Delta % per Game")
    st.markdown('<p class="kicker">((actual pts − xP) / xP) × 100. Normalised for playing time and role.</p>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(hbar(tp.nlargest(5,"avg_delta_pct")[["TEAM_ABBRV","avg_delta_pct"]],
                             "TEAM_ABBRV","avg_delta_pct","Teams — best avg xP delta %",
                             color=GREEN, height=280, ascending=True, pct=True),
                        width='stretch')
    with e2:
        st.plotly_chart(hbar(pp.nlargest(5,"avg_delta_pct")[["FULL NAME","avg_delta_pct"]],
                             "FULL NAME","avg_delta_pct","Players — best avg xP delta %",
                             color=GREEN, height=280, ascending=True, pct=True),
                        width='stretch')

    e3, e4 = st.columns(2)
    with e3:
        st.plotly_chart(hbar(tp.nsmallest(5,"avg_delta_pct")[["TEAM_ABBRV","avg_delta_pct"]],
                             "TEAM_ABBRV","avg_delta_pct","Teams — worst avg xP delta %",
                             color=RED, height=280, ascending=False, pct=True),
                        width='stretch')
    with e4:
        st.plotly_chart(hbar(pp.nsmallest(5,"avg_delta_pct")[["FULL NAME","avg_delta_pct"]],
                             "FULL NAME","avg_delta_pct","Players — worst avg xP delta %",
                             color=RED, height=280, ascending=False, pct=True),
                        width='stretch')

    # ── Scatter ────────────────────────────────────────────────────────────────
    st.divider()
    section("Outperformance Rate vs Avg Points Above xP")
    st.markdown('<p class="kicker">Top-right = frequently beats xP by a large margin. The truest elite performers.</p>', unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.plotly_chart(scatter_chart(tp,"outperform_pct","avg_delta","TEAM_ABBRV",
                                      "Teams — rate vs avg Δ", height=400),
                        width='stretch')
    with sc2:
        top30p = pp.nlargest(30,"avg_delta")
        st.plotly_chart(scatter_chart(top30p,"outperform_pct","avg_delta","FULL NAME",
                                      "Players (top 30) — rate vs avg Δ", height=400),
                        width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB_TEMP_SHOOT · SHOOTING STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shoot:

    total_shots = len(shots)
    made   = (shots["EVENT_TYPE"].str.lower().str.contains("made")).sum() if "EVENT_TYPE" in shots.columns else 0
    missed = total_shots - made
    fg_pct = round(made / total_shots * 100, 1) if total_shots else 0
    top_zone = shots["ZONE_NAME"].mode()[0] if "ZONE_NAME" in shots.columns else "—"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total shots tracked", f"{total_shots:,}")
    k2.metric("Made",                f"{made:,}")
    k3.metric("Missed",              f"{missed:,}")
    k4.metric("FG%",                 f"{fg_pct}%")
    k5.metric("Top zone",            top_zone)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.divider()
    section("Shot Outcomes & Game Flow")
    c1, c2 = st.columns(2)
    with c1:
        if "EVENT_TYPE" in shots.columns:
            oc = shots["EVENT_TYPE"].value_counts().reset_index()
            oc.columns = ["EVENT_TYPE","count"]
            st.plotly_chart(donut(oc,"EVENT_TYPE","count","Shot outcomes — made vs missed"),
                            width='stretch')
    with c2:
        if "QUARTER" in shots.columns:
            st.plotly_chart(vbar(agg_top(shots,"QUARTER",6),"QUARTER","count",
                                 "Shots per quarter", color=ACCENT, height=320),
                            width='stretch')

    st.divider()
    section("Shot Action Types")
    if "ACTION_TYPE" in shots.columns:
        st.plotly_chart(hbar(agg_top(shots,"ACTION_TYPE",10),"ACTION_TYPE","count",
                             "Top 10 shot action types", color=ORANGE, height=420, ascending=True),
                        width='stretch')

    st.divider()
    section("Zone Distribution")

    if "ZONE_NAME" in shots.columns:
        zone_data = shots["ZONE_NAME"].value_counts().reset_index()
        zone_data.columns = ["Zone","Count"]
        zone_data["Pct"] = (zone_data["Count"] / zone_data["Count"].sum() * 100).round(1)
        cols = st.columns(len(zone_data))
        palette = [ACCENT, ORANGE, GREEN, RED, AMBER, "#7c3aed","#0891b2","#be185d","#854d0e"]
        for i, (_, row) in enumerate(zone_data.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid #e5e5e3;border-radius:4px;
                            padding:14px 10px;text-align:center;border-top:3px solid {palette[i % len(palette)]};">
                  <p style="font-family:Inter;font-size:10px;color:#888;text-transform:uppercase;
                             letter-spacing:0.5px;margin:0 0 4px;">{row['Zone']}</p>
                  <p style="font-family:'Source Serif 4',serif;font-size:22px;font-weight:600;
                             color:#111;margin:0;">{row['Pct']:.1f}%</p>
                  <p style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#888;
                             margin:3px 0 0;">{row['Count']:,} attempts</p>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    z1, z2 = st.columns(2)
    with z1:
        if "BASIC_ZONE" in shots.columns:
            st.plotly_chart(hbar(agg_top(shots,"BASIC_ZONE",8),"BASIC_ZONE","count",
                                 "By basic zone", color=ACCENT, height=340, ascending=True),
                            width='stretch')
    with z2:
        if "ZONE_RANGE" in shots.columns:
            st.plotly_chart(lollipop(agg_top(shots,"ZONE_RANGE",8),"ZONE_RANGE","count",
                                     "By range", color=ACCENT, height=340, ascending=True),
                            width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB_TEMP_TEAM · TEAM STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_team:

    top10_shots = shots["TEAM_NAME"].value_counts().head(10).reset_index()
    top10_shots.columns = ["TEAM_NAME","count"]
    bot10_shots = shots["TEAM_NAME"].value_counts().tail(10).reset_index()
    bot10_shots.columns = ["TEAM_NAME","count"]
    top10_oppg  = teams[["TEAM","oPPG"]].drop_duplicates().sort_values("oPPG").head(10)
    top10_deff  = teams[["TEAM","dEFF"]].drop_duplicates().sort_values("dEFF").head(10)
    bot10_oppg  = teams[["TEAM","oPPG"]].drop_duplicates().sort_values("oPPG",ascending=False).head(10)
    bot10_deff  = teams[["TEAM","dEFF"]].drop_duplicates().sort_values("dEFF",ascending=False).head(10)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Most attempts",  f"{top10_shots['count'].iloc[0]:,}",  top10_shots['TEAM_NAME'].iloc[0])
    k2.metric("Fewest attempts",f"{bot10_shots['count'].iloc[0]:,}",  bot10_shots['TEAM_NAME'].iloc[0])
    k3.metric("Best def. oPPG", f"{top10_oppg['oPPG'].iloc[0]:.1f}", top10_oppg['TEAM'].iloc[0])
    k4.metric("Best dEFF",      f"{top10_deff['dEFF'].iloc[0]:.1f}", top10_deff['TEAM'].iloc[0])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.divider()
    section("Shot Volume by Team")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(hbar(top10_shots,"TEAM_NAME","count","Top 10 shot volume teams",
                             color=ACCENT, height=380, ascending=True), width='stretch')
    with c2:
        st.plotly_chart(lollipop(bot10_shots,"TEAM_NAME","count","Bottom 10 shot volume teams",
                                 color=RED, height=380, ascending=True), width='stretch')

    st.divider()
    section("Defensive Efficiency")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(hbar(top10_oppg,"TEAM","oPPG","Best defensive oPPG (lower = better)",
                             color=GREEN, height=360, ascending=True), width='stretch')
        st.plotly_chart(hbar(bot10_oppg,"TEAM","oPPG","Worst defensive oPPG",
                             color=RED, height=360, ascending=False), width='stretch')
    with c4:
        st.plotly_chart(hbar(top10_deff,"TEAM","dEFF","Best defensive efficiency (dEFF)",
                             color=GREEN, height=360, ascending=True), width='stretch')
        st.plotly_chart(hbar(bot10_deff,"TEAM","dEFF","Worst defensive efficiency (dEFF)",
                             color=RED, height=360, ascending=False), width='stretch')

    st.divider()
    section("Defensive Profile — oPPG vs dEFF")
    all_teams_df = teams[["TEAM","oPPG","dEFF"]].drop_duplicates()
    st.plotly_chart(scatter_chart(all_teams_df,"oPPG","dEFF","TEAM",
                                  "Team defensive profile — oPPG allowed vs defensive efficiency",
                                  height=480), width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB_TEMP_PLAYER · PLAYER STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_player:

    st.markdown(
        '<p style="font-family:Inter;font-size:12px;color:#888;margin-bottom:16px;">'
        'Showing players with 50+ games played this season</p>',
        unsafe_allow_html=True)

    shot_takers = shots["PLAYER_NAME"].value_counts().head(10).reset_index()
    shot_takers.columns = ["PLAYER_NAME","Shot Attempts"]

    stat_cols = [
        ("eFG%", "Top 10 effective FG%",    GREEN),
        ("TS%",  "Top 10 true shooting %",  GREEN),
        ("2P%",  "Top 10 two-point FG%",    ACCENT),
        ("3P%",  "Top 10 three-point FG%",  ORANGE),
        ("ORTG", "Top 10 offensive rating", ACCENT),
    ]
    valid = [(c,l,col) for c,l,col in stat_cols if c in players_filtered.columns]

    # ── Player selector (drives KPIs + cards) ────────────────────────────────
    st.divider()
    section("Player Cards")
    st.markdown('<p class="kicker">Select a player to view their full shooting profile and see how they compare to league averages.</p>', unsafe_allow_html=True)

    player_search = st.selectbox(
        "Search player",
        options=sorted(players_filtered["FULL NAME"].dropna().unique()),
        key="player_card_selector"
    )

    # ── KPI row — selected player's stats vs league avg ────────────────────────
    league_avgs = {col: players_filtered[col].mean() for col, _, _ in valid}

    sel_for_kpi = players_filtered[players_filtered["FULL NAME"] == player_search]
    kpi_cols = st.columns(len(valid))
    for (col, label, _), kc in zip(valid, kpi_cols):
        if not sel_for_kpi.empty:
            val      = sel_for_kpi.iloc[0][col]
            lg_avg   = league_avgs[col]
            delta    = val - lg_avg
            suffix   = "%" if "%" in col else ""
            # delta label: show vs league avg with arrow direction
            delta_str = f"{delta:+.1f}{suffix} vs league avg"
            kc.metric(col, f"{val:.1f}{suffix}", delta_str)
        else:
            best   = players_filtered.loc[players_filtered[col].idxmax()]
            suffix = "%" if "%" in col else ""
            kc.metric(col, f"{best[col]:.1f}{suffix}", best["FULL NAME"])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    sel_player = players_filtered[players_filtered["FULL NAME"] == player_search]

    if not sel_player.empty:
        pm = sel_player.iloc[0]

        # Also pull xP data for this player if available
        xp_row = pp[pp["FULL NAME"] == player_search] if not pp.empty else pd.DataFrame()

        pc1, pc2, pc3 = st.columns([1, 1, 1])

        with pc1:
            # Identity card
            team_name = pm.get("TEAM", pm.get("Team", "—"))
            pos_name  = pm.get("POS", pm.get("Pos", "—"))
            gp        = int(pm.get("GP", 0))
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e5e5e3;border-radius:4px;
                        padding:20px;border-top:3px solid {ACCENT};">
              <p style="font-family:Inter;font-size:11px;font-weight:600;color:#888;
                        text-transform:uppercase;letter-spacing:0.08em;margin:0 0 4px;">{team_name} &nbsp;·&nbsp; {pos_name}</p>
              <p style="font-family:'Source Serif 4',serif;font-size:24px;font-weight:600;
                        color:#111;margin:0 0 12px;">{player_search}</p>
              <div style="border-top:1px solid #f0f0ee;padding-top:10px;">
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                  <span style="font-family:Inter;font-size:11px;color:#888;">Games played</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{gp}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                  <span style="font-family:Inter;font-size:11px;color:#888;">eFG%</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{pm.get('eFG%',0):.1f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                  <span style="font-family:Inter;font-size:11px;color:#888;">TS%</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{pm.get('TS%',0):.1f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                  <span style="font-family:Inter;font-size:11px;color:#888;">2P%</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{pm.get('2P%',0):.1f}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;">
                  <span style="font-family:Inter;font-size:11px;color:#888;">3P%</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{pm.get('3P%',0):.1f}%</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        with pc2:
            # Advanced stats card
            ortg = pm.get("ORTG", 0)
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e5e5e3;border-radius:4px;
                        padding:20px;border-top:3px solid {GREEN};">
              <p style="font-family:Inter;font-size:11px;font-weight:600;color:#888;
                        text-transform:uppercase;letter-spacing:0.08em;margin:0 0 16px;">Advanced Stats</p>
              <div style="border-top:1px solid #f0f0ee;padding-top:10px;">
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                  <span style="font-family:Inter;font-size:11px;color:#888;">Offensive Rating</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{ortg:.1f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                  <span style="font-family:Inter;font-size:11px;color:#888;">Shot attempts (season)</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{shots[shots['PLAYER_NAME']==player_search].shape[0]:,}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        with pc3:
            # xP card
            if not xp_row.empty:
                xr = xp_row.iloc[0]
                out_pct  = xr.get("outperform_pct", 0)
                und_pct  = xr.get("underperform_pct", 0)
                avg_d    = xr.get("avg_delta", 0)
                avg_dpct = xr.get("avg_delta_pct", 0)
                total_g  = int(xr.get("total", 0))
                delta_badge = f'<span class="badge-good">+{avg_d:.1f} pts</span>' if avg_d >= 0 else f'<span class="badge-bad">{avg_d:.1f} pts</span>'
                st.markdown(f"""
                <div style="background:#fff;border:1px solid #e5e5e3;border-radius:4px;
                            padding:20px;border-top:3px solid {ORANGE};">
                  <p style="font-family:Inter;font-size:11px;font-weight:600;color:#888;
                            text-transform:uppercase;letter-spacing:0.08em;margin:0 0 16px;">xP Profile</p>
                  <div style="border-top:1px solid #f0f0ee;padding-top:10px;">
                    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                      <span style="font-family:Inter;font-size:11px;color:#888;">Games analysed</span>
                      <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{total_g}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                      <span style="font-family:Inter;font-size:11px;color:#888;">Outperformance rate</span>
                      <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{out_pct:.0f}%</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                      <span style="font-family:Inter;font-size:11px;color:#888;">Underperformance rate</span>
                      <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{und_pct:.0f}%</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #f9f9f7;">
                      <span style="font-family:Inter;font-size:11px;color:#888;">Avg pts above xP</span>
                      {delta_badge}
                    </div>
                    <div style="display:flex;justify-content:space-between;padding:5px 0;">
                      <span style="font-family:Inter;font-size:11px;color:#888;">Avg delta %</span>
                      <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#111;font-weight:500;">{avg_dpct:+.1f}%</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid #e5e5e3;border-radius:4px;
                            padding:20px;border-top:3px solid {ORANGE};">
                  <p style="font-family:Inter;font-size:11px;font-weight:600;color:#888;
                            text-transform:uppercase;letter-spacing:0.08em;margin:0 0 12px;">xP Profile</p>
                  <p style="font-family:Inter;font-size:12px;color:#888;">
                    Not enough games to compute xP profile (min {min_games_xp} games required).
                    Adjust the sidebar filter to include this player.</p>
                </div>""", unsafe_allow_html=True)

    st.divider()
    section("Shot Volume")
    st.plotly_chart(hbar(shot_takers,"PLAYER_NAME","Shot Attempts",
                         "Top 10 shot takers this season",
                         color=ORANGE, height=380, ascending=True), width='stretch')

    st.divider()
    section("Shooting Efficiency")

    if "eFG%" in players_filtered.columns and "TS%" in players_filtered.columns:
        top30 = players_filtered.nlargest(30,"eFG%")
        st.plotly_chart(scatter_chart(top30,"eFG%","TS%","FULL NAME",
                                      "eFG% vs TS% — top 30 players by eFG%", height=460),
                        width='stretch')

    st.divider()
    c1, c2 = st.columns(2)
    sides = [c1, c2, c1, c2, c1]
    for (col, title, color), side in zip(valid, sides):
        top_df = players_filtered[["FULL NAME",col]].sort_values(col,ascending=False).head(10)
        with side:
            st.plotly_chart(hbar(top_df,"FULL NAME",col,title,
                                 color=color, height=360, ascending=True, pct=("%" in col)),
                            width='stretch')

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;font-family:Inter,sans-serif;font-size:11px;color:#aaa;">'
    'SwishScore · NBA xP Prediction Model · github.com/jngoh24/swishscore_nba</p>',
    unsafe_allow_html=True)
