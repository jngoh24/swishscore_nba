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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .block-container {
    background-color: #0e0f13 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #f0f0f0 !important;
}
.block-container { padding: 2rem 2rem 4rem !important; max-width: 100% !important; }
[data-testid="stHeader"] { background: #0e0f13 !important; border-bottom: none !important; }
[data-testid="stToolbar"], [data-testid="stToolbarActions"],
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

[data-testid="stTabs"] [role="tablist"] {
    background: #161820 !important;
    border-radius: 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    padding: 0 8px !important; gap: 2px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important; font-weight: 500 !important;
    color: #9a9aaa !important; padding: 14px 18px 12px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #e87c2a !important;
    border-bottom: 2px solid #e87c2a !important;
}
[data-testid="stTabContent"] { background: #0e0f13 !important; padding-top: 28px !important; }

[data-testid="stMetric"] {
    background: #1a1c24 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important; padding: 16px 18px !important;
}
[data-testid="stMetricLabel"] p { color: #9a9aaa !important; font-size: 11px !important; letter-spacing: 0.3px !important; }
[data-testid="stMetricValue"]   { color: #f0f0f0 !important; font-size: 24px !important; font-weight: 600 !important; letter-spacing: -0.5px !important; }
[data-testid="stMetricDelta"]   { font-size: 11px !important; }

[data-testid="stInfo"] {
    background: rgba(59,130,246,0.08) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 6px !important; padding: 8px 12px !important;
}
[data-testid="stInfo"] p { color: #9a9aaa !important; font-size: 11px !important; }

/* filter panel */
.filter-panel {
    background: #1a1c24;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}

hr { border-color: rgba(255,255,255,0.06) !important; margin: 20px 0 !important; }
p, span, label, li { color: #9a9aaa !important; font-family: 'DM Sans', sans-serif !important; }
h1 { color: #f0f0f0 !important; font-size: 20px !important; font-weight: 600 !important; }
h3 { color: #5a5a6a !important; font-size: 10px !important; font-weight: 600 !important;
     letter-spacing: 1.2px !important; text-transform: uppercase !important; margin: 18px 0 12px !important; }
iframe { border: none !important; }

[data-testid="stButton"] button {
    background: #1a1c24 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #9a9aaa !important;
    font-size: 11px !important;
    padding: 4px 12px !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stButton"] button:hover {
    border-color: #e87c2a !important;
    color: #e87c2a !important;
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

# ── Chart helpers ──────────────────────────────────────────────────────────────
BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9a9aaa", size=11),
    margin=dict(l=16, r=16, t=44, b=16),
    coloraxis_showscale=False,
    title_font=dict(size=13, color="#f0f0f0", family="DM Sans"),
    title_x=0,
    showlegend=False,
)

def xaxis(tickangle=-35):
    return dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=10, color="#9a9aaa"), tickangle=tickangle,
                showgrid=False, zeroline=False)

def yaxis():
    return dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=10, color="#9a9aaa"), showgrid=True, zeroline=False)

def agg_top(df_in, col, top_n=10):
    c = df_in[col].value_counts().reset_index()
    c.columns = [col, "count"]
    c["pct"] = (c["count"] / c["count"].sum() * 100).round(1)
    return c.head(top_n)

def section(label):
    st.markdown(f"<h3>{label}</h3>", unsafe_allow_html=True)

# ── Horizontal lollipop chart ──────────────────────────────────────────────────
def lollipop(df_in, y_col, x_col, title, color="#3b82f6", height=380, ascending=True, pct=False):
    d = df_in.sort_values(x_col, ascending=ascending)
    suffix = "%" if pct else ""
    fig = go.Figure()
    for _, row in d.iterrows():
        fig.add_shape(type="line",
            x0=0, x1=row[x_col], y0=row[y_col], y1=row[y_col],
            line=dict(color=color, width=2), opacity=0.35)
    fig.add_trace(go.Scatter(
        x=d[x_col], y=d[y_col], mode="markers+text",
        marker=dict(color=color, size=10, line=dict(color="#0e0f13", width=2)),
        text=[f"{v:.1f}{suffix}" for v in d[x_col]],
        textposition="middle right",
        textfont=dict(size=10, color="#f0f0f0"),
        hovertemplate=f"%{{y}}: %{{x:.1f}}{suffix}<extra></extra>",
    ))
    fig.update_layout(**BASE, title=title, height=height,
                      xaxis={**xaxis(0), "showgrid": True},
                      yaxis={**yaxis(), "showgrid": False,
                             "categoryorder": "array",
                             "categoryarray": list(d[y_col])})
    return fig

# ── Gradient horizontal bar ────────────────────────────────────────────────────
def hbar(df_in, y_col, x_col, title, color_scale, height=380, ascending=True, pct=False):
    d = df_in.sort_values(x_col, ascending=ascending)
    suffix = "%" if pct else ""
    norm = (d[x_col] - d[x_col].min()) / (d[x_col].max() - d[x_col].min() + 1e-9)
    colors = px.colors.sample_colorscale(color_scale, norm.tolist())
    fig = go.Figure(go.Bar(
        x=d[x_col], y=d[y_col], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}{suffix}" for v in d[x_col]],
        textposition="inside",
        textfont=dict(size=10, color="rgba(255,255,255,0.9)"),
        hovertemplate=f"%{{y}}: %{{x:.1f}}{suffix}<extra></extra>",
    ))
    fig.update_layout(**BASE, title=title, height=height,
                      xaxis={**xaxis(0), "showgrid": True},
                      yaxis={**yaxis(), "showgrid": False,
                             "categoryorder": "array",
                             "categoryarray": list(d[y_col])})
    return fig

# ── Donut ──────────────────────────────────────────────────────────────────────
def donut(df_in, names, values, title, height=340):
    fig = px.pie(df_in, names=names, values=values, hole=0.5, height=height,
                 color_discrete_sequence=["#3b82f6","#ef4444","#22c55e","#e87c2a","#a855f7"])
    fig.update_traces(
        textinfo="percent+label", pull=[0.03]*len(df_in),
        marker=dict(line=dict(color="#0e0f13", width=2)),
        textfont=dict(size=11, color="#f0f0f0"),
    )
    fig.update_layout(**{**BASE, "margin": dict(l=12,r=12,t=44,b=12)}, title=title, showlegend=True,
                      legend=dict(font=dict(color="#9a9aaa", size=10),
                                  bgcolor="rgba(0,0,0,0)", x=1, y=0.5))
    return fig

# ── Grouped bar (for quarter data) ────────────────────────────────────────────
def vbar(df_in, x_col, y_col, title, color_scale="Blues", height=320, pct=False):
    d = df_in.sort_values(x_col)
    suffix = "%" if pct else ""
    norm = (d[y_col] - d[y_col].min()) / (d[y_col].max() - d[y_col].min() + 1e-9)
    colors = px.colors.sample_colorscale(color_scale, norm.tolist())
    fig = go.Figure(go.Bar(
        x=d[x_col].astype(str), y=d[y_col],
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:,}{suffix}" for v in d[y_col]],
        textposition="outside",
        textfont=dict(size=10, color="#9a9aaa"),
        hovertemplate=f"%{{x}}: %{{y:,}}{suffix}<extra></extra>",
    ))
    fig.update_layout(**BASE, title=title, height=height,
                      xaxis={**xaxis(0)},
                      yaxis={**yaxis()},
                      margin=dict(l=16, r=16, t=44, b=16))
    return fig

# ── Scatter (eFG% vs ORTG) ────────────────────────────────────────────────────
def scatter(df_in, x_col, y_col, name_col, title, height=420):
    fig = px.scatter(df_in, x=x_col, y=y_col, text=name_col, height=height,
                     color=y_col,
                     color_continuous_scale=[[0,"#1d4ed8"],[0.5,"#e87c2a"],[1,"#22c55e"]])
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=9, color="#9a9aaa"),
        marker=dict(size=9, line=dict(color="#0e0f13", width=1)),
    )
    fig.update_layout(**BASE, title=title,
                      xaxis={**xaxis(0), "title": dict(text=x_col, font=dict(size=11, color="#9a9aaa"))},
                      yaxis={**yaxis(), "title": dict(text=y_col, font=dict(size=11, color="#9a9aaa"))})
    return fig

# ── xP progress-bar card ──────────────────────────────────────────────────────
def xp_card(title, tag_text, tag_bg, tag_fg, data, name_col, val_col, bar_color, height=210):
    max_val = data[val_col].max() or 1
    rows = ""
    for _, row in data.iterrows():
        w = row[val_col] / max_val * 100
        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
          <span style="font-size:12px;color:#f0f0f0;width:130px;flex-shrink:0;
                       white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                       font-family:'DM Sans',sans-serif;">{row[name_col]}</span>
          <div style="flex:1;height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;">
            <div style="width:{w:.1f}%;height:100%;background:{bar_color};border-radius:3px;"></div>
          </div>
          <span style="font-size:11px;color:#9a9aaa;font-family:'DM Mono',monospace;
                       width:36px;text-align:right;">{int(row[val_col])}%</span>
        </div>"""
    html = f"""<!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
    <style>*{{margin:0;padding:0;box-sizing:border-box;}}</style>
    </head>
    <body style="background:#1a1c24;padding:20px;border-radius:10px;border:1px solid rgba(255,255,255,0.07);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
        <span style="font-size:13px;font-weight:500;color:#f0f0f0;font-family:'DM Sans',sans-serif;">{title}</span>
        <span style="font-size:10px;padding:2px 8px;border-radius:4px;font-weight:500;
                     background:{tag_bg};color:{tag_fg};font-family:'DM Sans',sans-serif;">{tag_text}</span>
      </div>
      {rows}
    </body></html>"""
    components.html(html, height=height, scrolling=False)

# ── Zone badge grid ────────────────────────────────────────────────────────────
def zone_grid(df_in, col):
    counts = df_in[col].value_counts()
    total  = counts.sum()
    max_c  = counts.max()
    cards  = ""
    palette = ["#3b82f6","#e87c2a","#22c55e","#a855f7","#ef4444","#06b6d4","#84cc16","#f59e0b","#ec4899"]
    for i, (zone, cnt) in enumerate(counts.items()):
        pct   = cnt / total * 100
        bar_w = cnt / max_c * 100
        col_c = palette[i % len(palette)]
        cards += f"""
        <div style="background:#1a1c24;border:1px solid rgba(255,255,255,0.07);
                    border-radius:8px;padding:14px 12px;text-align:center;">
          <div style="font-size:9px;color:#5a5a6a;text-transform:uppercase;
                      letter-spacing:0.6px;margin-bottom:6px;font-family:'DM Sans',sans-serif;">{zone}</div>
          <div style="font-size:22px;font-weight:600;color:#f0f0f0;
                      font-family:'DM Sans',sans-serif;letter-spacing:-0.5px;">{pct:.1f}%</div>
          <div style="font-size:10px;color:#5a5a6a;margin-top:3px;
                      font-family:'DM Sans',sans-serif;">{cnt:,} attempts</div>
          <div style="height:3px;border-radius:2px;margin-top:10px;
                      background:{col_c};width:{bar_w:.0f}%;margin-left:auto;margin-right:auto;"></div>
        </div>"""
    n_rows = -(-len(counts) // 3)
    html = f"""<!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>*{{margin:0;padding:0;box-sizing:border-box;}}
    body{{background:#0e0f13;font-family:'DM Sans',sans-serif;padding:4px 0;}}
    .g{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}}
    </style></head><body>
    <div class="g">{cards}</div>
    </body></html>"""
    components.html(html, height=n_rows * 115 + 16, scrolling=False)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:0 0 20px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:6px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:34px;height:34px;border-radius:50%;background:#e87c2a;
                display:flex;align-items:center;justify-content:center;
                font-weight:700;font-size:15px;color:#fff;flex-shrink:0;">S</div>
    <div>
      <div style="font-size:17px;font-weight:600;color:#f0f0f0;letter-spacing:-0.4px;
                  font-family:'DM Sans',sans-serif;">SwishScore</div>
      <div style="font-size:11px;color:#5a5a6a;font-family:'DM Sans',sans-serif;">
        NBA Shot Outcome Prediction &nbsp;·&nbsp; xP Model
      </div>
    </div>
  </div>
  <a href="https://github.com/jngoh24/swishscore_nba"
     style="font-size:11px;color:#9a9aaa;text-decoration:none;
            border:1px solid rgba(255,255,255,0.1);padding:5px 12px;
            border-radius:6px;font-family:'DM Sans',sans-serif;">
    github.com/jngoh24/swishscore_nba
  </a>
</div>
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

    # Filter toggle — NO st.expander
    if "show_filters" not in st.session_state:
        st.session_state.show_filters = False

    if st.button("🔍  Filter options  ▾" if not st.session_state.show_filters else "🔍  Filter options  ▴"):
        st.session_state.show_filters = not st.session_state.show_filters

    unique_teams = sorted(master_xp["TEAM_ABBRV"].unique())
    unique_confs = sorted(master_xp["CONF"].unique())
    unique_divs  = sorted(master_xp["DIVISION"].unique())

    if st.session_state.show_filters:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
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
        if st.button("↺  Reset"):
            for k, v in [("xp_teams", unique_teams),("xp_confs", unique_confs),("xp_divs", unique_divs)]:
                st.session_state[k] = v
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        sel_teams = st.session_state.get("xp_teams", unique_teams)
        sel_confs = st.session_state.get("xp_confs", unique_confs)
        sel_divs  = st.session_state.get("xp_divs",  unique_divs)

    fxp = master_xp[
        master_xp["TEAM_ABBRV"].isin(sel_teams) &
        master_xp["CONF"].isin(sel_confs) &
        master_xp["DIVISION"].isin(sel_divs)
    ]

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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Teams in view",         len(tp))
    k2.metric("Players (min 10 GP)",   len(pp))
    k3.metric("Avg outperform rate",   f"{tp['outperform_pct'].mean():.0f}%")
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

    # xP scatter: pts vs xP per team
    st.divider()
    section("Actual Points vs xP by Team")
    team_scatter = fxp.groupby("TEAM_ABBRV").agg(
        avg_xP=("xP","mean"), avg_pts=("pts","mean")).reset_index()
    st.plotly_chart(
        scatter(team_scatter, "avg_xP", "avg_pts", "TEAM_ABBRV",
                "Avg actual pts vs avg xP per team (above diagonal = outperforming)", height=440),
        use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 · SHOOTING STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    total_shots = len(shots)
    made   = (shots["EVENT_TYPE"].str.lower().str.contains("made")).sum() if "EVENT_TYPE" in shots.columns else 0
    missed = total_shots - made
    fg_pct = round(made / total_shots * 100, 1) if total_shots else 0
    top_zone = shots["ZONE_NAME"].mode()[0] if "ZONE_NAME" in shots.columns else "—"

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
            st.plotly_chart(donut(oc,"EVENT_TYPE","count","Shot outcomes — made vs missed"),
                            use_container_width=True)
    with c2:
        if "QUARTER" in shots.columns:
            qd = agg_top(shots,"QUARTER",6)
            st.plotly_chart(vbar(qd,"QUARTER","count","Shots per quarter", color_scale="Blues", height=340),
                            use_container_width=True)

    st.divider()
    section("Shot Action Types — top 10")
    if "ACTION_TYPE" in shots.columns:
        act = agg_top(shots,"ACTION_TYPE",10)
        st.plotly_chart(hbar(act,"ACTION_TYPE","count","Top 10 shot action types",
                             color_scale="Oranges", height=420, ascending=True),
                        use_container_width=True)

    st.divider()
    section("Zone Distribution")
    if "ZONE_NAME" in shots.columns:
        zone_grid(shots, "ZONE_NAME")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    z1, z2 = st.columns(2)
    with z1:
        if "BASIC_ZONE" in shots.columns:
            st.plotly_chart(hbar(agg_top(shots,"BASIC_ZONE",8),"BASIC_ZONE","count",
                                 "Shot distribution by basic zone",
                                 color_scale="Blues", height=340, ascending=True),
                            use_container_width=True)
    with z2:
        if "ZONE_RANGE" in shots.columns:
            st.plotly_chart(lollipop(agg_top(shots,"ZONE_RANGE",8),"ZONE_RANGE","count",
                                     "Shot range distribution",
                                     color="#3b82f6", height=340),
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
        st.plotly_chart(hbar(top10_shots,"TEAM_NAME","count",
                             "Top 10 shot volume teams",
                             color_scale="Blues", height=380, ascending=True),
                        use_container_width=True)
    with c2:
        st.plotly_chart(lollipop(bot10_shots,"TEAM_NAME","count",
                                 "Bottom 10 shot volume teams",
                                 color="#ef4444", height=380, ascending=True),
                        use_container_width=True)

    st.divider()
    section("Defensive Efficiency")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(hbar(top10_oppg,"TEAM","oPPG",
                             "Best defensive oPPG (lower = better)",
                             color_scale="Greens", height=360, ascending=True),
                        use_container_width=True)
        st.plotly_chart(hbar(bot10_oppg,"TEAM","oPPG",
                             "Worst defensive oPPG",
                             color_scale="Reds", height=360, ascending=False),
                        use_container_width=True)
    with c4:
        st.plotly_chart(hbar(top10_deff,"TEAM","dEFF",
                             "Best defensive efficiency (dEFF)",
                             color_scale="Greens", height=360, ascending=True),
                        use_container_width=True)
        st.plotly_chart(hbar(bot10_deff,"TEAM","dEFF",
                             "Worst defensive efficiency (dEFF)",
                             color_scale="Reds", height=360, ascending=False),
                        use_container_width=True)

    # Scatter: oPPG vs dEFF
    st.divider()
    section("Defensive Profile — oPPG vs dEFF")
    all_teams = teams[["TEAM","oPPG","dEFF"]].drop_duplicates()
    st.plotly_chart(scatter(all_teams,"oPPG","dEFF","TEAM",
                            "Team defensive profile — oPPG allowed vs defensive efficiency",
                            height=480),
                    use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 · PLAYER STATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.info("Showing players with 50+ games played this season")

    shot_takers = shots["PLAYER_NAME"].value_counts().head(10).reset_index()
    shot_takers.columns = ["PLAYER_NAME","Shot Attempts"]

    stat_cols = [
        ("eFG%", "Top 10 effective FG% (eFG%)",   "Greens"),
        ("TS%",  "Top 10 true shooting % (TS%)",   "Greens"),
        ("2P%",  "Top 10 two-point FG% (2P%)",     "Blues"),
        ("3P%",  "Top 10 three-point FG% (3P%)",   "Oranges"),
        ("ORTG", "Top 10 offensive rating (ORTG)",  "Blues"),
    ]
    valid = [(c,l,s) for c,l,s in stat_cols if c in players_filtered.columns]

    # KPI row
    kpi_cols = st.columns(len(valid))
    for (col, label, _), kc in zip(valid, kpi_cols):
        best = players_filtered.loc[players_filtered[col].idxmax()]
        suffix = "%" if "%" in col else ""
        kc.metric(col, f"{best[col]:.1f}{suffix}", best["FULL NAME"])

    st.divider()
    section("Shot Volume")
    st.plotly_chart(hbar(shot_takers,"PLAYER_NAME","Shot Attempts",
                         "Top 10 shot takers this season",
                         color_scale="Oranges", height=380, ascending=True),
                    use_container_width=True)

    st.divider()
    section("Shooting Efficiency")

    # eFG% vs TS% scatter
    if "eFG%" in players_filtered.columns and "TS%" in players_filtered.columns:
        top30 = players_filtered.nlargest(30,"eFG%")
        st.plotly_chart(scatter(top30,"eFG%","TS%","FULL NAME",
                                "eFG% vs TS% — top 30 players by eFG%", height=460),
                        use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    sides = [c1, c2, c1, c2, c1]
    for (col, title, scale), side in zip(valid, sides):
        top_df = players_filtered[["FULL NAME",col]].sort_values(col,ascending=False).head(10)
        with side:
            st.plotly_chart(hbar(top_df,"FULL NAME",col,title,
                                 color_scale=scale, height=360,
                                 ascending=True, pct=("%" in col)),
                            use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding:16px 0;border-top:1px solid rgba(255,255,255,0.06);
            text-align:center;font-size:11px;color:#5a5a6a;font-family:'DM Sans',sans-serif;">
  SwishScore &nbsp;·&nbsp; NBA xP Model &nbsp;·&nbsp;
  <a href="https://github.com/jngoh24/swishscore_nba" style="color:#e87c2a;text-decoration:none;">jngoh24</a>
</div>
""", unsafe_allow_html=True)
