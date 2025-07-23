import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

url1 = "https://drive.google.com/uc?export=download&id=1TjmaUe1fTPNNtB1STGMq9af0JntkvCM0"
url2 = "https://drive.google.com/uc?export=download&id=11_iTfCLqvOjnR_OmDQnGZMu1-_0WnhQV"

shots = load_data(url1)
df = load_data(url2)

master_xp = pd.read_csv("streamlit/data/master_xp.csv")
teams = pd.read_csv("streamlit/data/team_data.csv")
players_original = pd.read_csv("streamlit/data/player_data.csv")
players_original['Team'] = players_original['TEAM'].str.upper()
players = players_original.dropna()
players_filtered = players[players["GP"] >= 50]

st.title("SwishScore - NBA Shot Outcome Prediction Model (xP)")

# Utility Functions
def make_bar_chart(df, category_col, count_col, title):
    fig = px.bar(
        df.sort_values(by=count_col, ascending=False),
        x=category_col,
        y=count_col,
        text=df["percentage"].astype(str) + "%",
        title=title,
        color=count_col,
        color_continuous_scale='blues'
    )
    fig.update_traces(textposition='inside')
    fig.update_layout(
        xaxis_title=category_col.replace("_", " ").title(),
        yaxis_title="Count",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    return fig

def make_pie_chart(df, category_col, title):
    counts = df[category_col].value_counts().reset_index()
    counts.columns = [category_col, "count"]
    counts["percentage"] = round((counts["count"] / counts["count"].sum()) * 100, 2)

    norm_counts = (counts["count"] - counts["count"].min()) / (counts["count"].max() - counts["count"].min() + 1e-9)
    blues_scale = px.colors.sequential.Blues
    colors = [blues_scale[int(x * (len(blues_scale) - 1))] for x in norm_counts]
    colors = list(reversed(colors))

    fig = px.pie(
        counts,
        names=category_col,
        values="count",
        title=title,
        hole=0.4,
    )
    fig.update_traces(
        textinfo='percent+label',
        pull=[0.05] * len(counts),
        marker=dict(colors=colors)
    )
    fig.update_layout(showlegend=True)

    st.plotly_chart(fig)

def agg_and_plot(df, group_col, title, top_n=10):
    counts = df[group_col].value_counts().reset_index()
    counts.columns = [group_col, "count"]
    counts["percentage"] = round((counts["count"] / counts["count"].sum()) * 100, 2)
    counts = counts.head(top_n)
    st.plotly_chart(make_bar_chart(counts, group_col, "count", title))


def plot_bar(df, x_col, y_col, title, ascending=False, percent_labels=False):
    df_sorted = df.sort_values(by=y_col, ascending=ascending)

    # Only apply "%" suffix if specified
    text_values = df_sorted[y_col].astype(str) + "%" if percent_labels else df_sorted[y_col]

    fig = px.bar(
        df_sorted,
        x=x_col,
        y=y_col,
        text=text_values,
        title=title,
        color=y_col,
        color_continuous_scale='blues'
    )
    fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig.update_layout(
        yaxis_title=y_col,
        xaxis_title=x_col,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(b=150),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig)

# ---------------------------
# Create tabs
tabs = st.tabs(["📈 xP Performance Summary", "📊 Shooting Stats", "🏀 Team Stats", "👤 Player Stats"])

# Tab 0: xP Performance Summary
with tabs[0]:
    st.title("📈 xP Performance Summary")

    with st.expander("🔍 Filter Options", expanded=False):
        unique_teams = sorted(master_xp["TEAM_ABBRV"].unique())
        unique_confs = sorted(master_xp["CONF"].unique())
        unique_divs = sorted(master_xp["DIVISION"].unique())

        # Store defaults
        if "default_teams" not in st.session_state:
            st.session_state.default_teams = unique_teams
        if "default_confs" not in st.session_state:
            st.session_state.default_confs = unique_confs
        if "default_divs" not in st.session_state:
            st.session_state.default_divs = unique_divs

        # Handle reset before widget creation
        if "filters_reset" not in st.session_state:
            st.session_state.filters_reset = False

        if st.session_state.filters_reset:
            # Set values before rendering widgets
            st.session_state["team_filter"] = st.session_state.default_teams
            st.session_state["conf_filter"] = st.session_state.default_confs
            st.session_state["div_filter"] = st.session_state.default_divs
            st.session_state.filters_reset = False  # prevent infinite rerun loop
            st.rerun()  # rerun with updated state

        # Create widgets with state values
        selected_teams = st.multiselect("Team", unique_teams, default=st.session_state.get("team_filter", unique_teams),
                                        key="team_filter")
        selected_confs = st.multiselect("Conference", unique_confs,
                                        default=st.session_state.get("conf_filter", unique_confs), key="conf_filter")
        selected_divs = st.multiselect("Division", unique_divs, default=st.session_state.get("div_filter", unique_divs),
                                       key="div_filter")

        if st.button("🔄 Reset Filters"):
            st.session_state.filters_reset = True
            st.rerun()

    # ===== TEAM-LEVEL ANALYSIS =====
    filtered_xp = master_xp[
        master_xp["TEAM_ABBRV"].isin(selected_teams) &
        master_xp["CONF"].isin(selected_confs) &
        master_xp["DIVISION"].isin(selected_divs)
        ]

    team_game_summary = filtered_xp.groupby(["GAME_ID", "TEAM_ABBRV"]).agg(
        total_xP=("xP", "sum"),
        total_pts=("pts", "sum")
    ).reset_index()

    team_game_summary["xP_performance"] = team_game_summary["total_pts"] > team_game_summary["total_xP"]
    team_game_summary["xP_performance"] = team_game_summary["xP_performance"].map({True: "yes", False: "no"})

    team_perf = team_game_summary.groupby("TEAM_ABBRV")["xP_performance"].value_counts().unstack().fillna(0)
    team_perf = team_perf.rename(columns={"yes": "outperform", "no": "underperform"}).reset_index()
    team_perf["total_games"] = team_perf["outperform"] + team_perf["underperform"]
    team_perf["outperform_pct"] = round((team_perf["outperform"] / team_perf["total_games"]) * 100)
    team_perf["underperform_pct"] = round((team_perf["underperform"] / team_perf["total_games"]) * 100)

    top_outperform_teams = team_perf.sort_values("outperform_pct", ascending=False).head(5)
    top_underperform_teams = team_perf.sort_values("underperform_pct", ascending=False).head(5)

    # ===== PLAYER-LEVEL ANALYSIS =====
    player_game_summary = filtered_xp.groupby(["GAME_ID", "FULL NAME"]).agg(
        total_xP=("xP", "sum"),
        total_pts=("pts", "sum")
    ).reset_index()

    player_game_summary["xP_performance"] = player_game_summary["total_pts"] > player_game_summary["total_xP"]
    player_game_summary["xP_performance"] = player_game_summary["xP_performance"].map({True: "yes", False: "no"})

    player_perf = player_game_summary.groupby("FULL NAME")["xP_performance"].value_counts().unstack().fillna(0)
    player_perf = player_perf.rename(columns={"yes": "outperform", "no": "underperform"}).reset_index()
    player_perf["total_games"] = player_perf["outperform"] + player_perf["underperform"]
    player_perf = player_perf[player_perf["total_games"] >= 10]  # Optional: Filter out low-sample players
    player_perf["outperform_pct"] = round((player_perf["outperform"] / player_perf["total_games"]) * 100)
    player_perf["underperform_pct"] = round((player_perf["underperform"] / player_perf["total_games"]) * 100)

    top_outperform_players = player_perf.sort_values("outperform_pct", ascending=False).head(5)
    top_underperform_players = player_perf.sort_values("underperform_pct", ascending=False).head(5)

    plot_bar(top_outperform_teams, "TEAM_ABBRV", "outperform_pct", "Teams: xP Outperformance %", percent_labels=True)

    plot_bar(top_outperform_players, "FULL NAME", "outperform_pct", "Players: xP Outperformance %", percent_labels=True)

    plot_bar(top_underperform_teams, "TEAM_ABBRV", "outperform_pct", "Teams: xP Underperformance %", ascending=True,
             percent_labels=True)

    plot_bar(top_underperform_players, "FULL NAME", "outperform_pct", "Players: xP Underperformance %",
             ascending=True, percent_labels=True)

# Tab 1: Shooting Stats
with tabs[1]:
    make_pie_chart(df, "EVENT_TYPE", "Shot Outcomes (event_type)")
    agg_and_plot(shots, "ACTION_TYPE", "Top 10 Shot Actions")
    agg_and_plot(shots, "ZONE_NAME", "Shot Distribution by Zone Name")
    agg_and_plot(shots, "BASIC_ZONE", "Shot Distribution by Basic Zone")
    agg_and_plot(shots, "ZONE_RANGE", "Shot Range Distribution")
    agg_and_plot(shots, "QUARTER", "Shots Per Quarter")

# Tab 2: Team Stats
with tabs[2]:
    top_10_shot_teams = shots['TEAM_NAME'].value_counts().head(10).reset_index()
    top_10_shot_teams.columns = ['TEAM_NAME', 'count']

    bottom_10_shot_teams = shots['TEAM_NAME'].value_counts().tail(10).reset_index()
    bottom_10_shot_teams.columns = ['TEAM_NAME', 'count']

    top10_oPPg = teams[['TEAM', 'oPPG']].drop_duplicates().sort_values('oPPG').head(10)
    top10_dEFF = teams[['TEAM', 'dEFF']].drop_duplicates().sort_values('dEFF').head(10)
    bottom10_oPPg = teams[['TEAM', 'oPPG']].drop_duplicates().sort_values('oPPG', ascending=False).head(10)
    bottom10_dEFF = teams[['TEAM', 'dEFF']].drop_duplicates().sort_values('dEFF', ascending=False).head(10)

    plot_bar(top_10_shot_teams, "TEAM_NAME", "count", "Top 10 Shot Teams")
    plot_bar(bottom_10_shot_teams, "TEAM_NAME", "count", "Bottom 10 Shot Teams", ascending=True)
    plot_bar(top10_oPPg, "TEAM", "oPPG", "Top 10 Teams by Defensive oPPG", ascending=True)
    plot_bar(top10_dEFF, "TEAM", "dEFF", "Top 10 Teams by Defensive Efficiency", ascending=True)
    plot_bar(bottom10_oPPg, "TEAM", "oPPG", "Bottom 10 Teams by Defensive oPPG", ascending=False)
    plot_bar(bottom10_dEFF, "TEAM", "dEFF", "Bottom 10 Teams by Defensive Efficiency", ascending=False)

# Tab 3: Player Stats
with tabs[3]:
    st.markdown("###### Includes players who played 50+ games this season")

    top_10_shot_takers_df = df["PLAYER_NAME"].value_counts().head(10).reset_index()
    top_10_shot_takers_df.columns = ["PLAYER_NAME", "Shot Attempts"]

    top10_eFG_df = players_filtered[["FULL NAME", "eFG%"]].sort_values(by="eFG%", ascending=False).head(10)
    top10_TS_df = players_filtered[["FULL NAME", "TS%"]].sort_values(by="TS%", ascending=False).head(10)
    top10_two_pt_df = players_filtered[["FULL NAME", "2P%"]].sort_values(by="2P%", ascending=False).head(10)
    top10_three_pt_df = players_filtered[["FULL NAME", "3P%"]].sort_values(by="3P%", ascending=False).head(10)
    top10_ORTG_df = players_filtered[["FULL NAME", "ORTG"]].sort_values(by="ORTG", ascending=False).head(10)

    plot_bar(top_10_shot_takers_df, "PLAYER_NAME", "Shot Attempts", "Top 10 Shot Takers")
    plot_bar(top10_eFG_df, "FULL NAME", "eFG%", "Top 10 Effective FG%")
    plot_bar(top10_TS_df, "FULL NAME", "TS%", "Top 10 True Shooting %")
    plot_bar(top10_two_pt_df, "FULL NAME", "2P%", "Top 10 2-Point FG%")
    plot_bar(top10_three_pt_df, "FULL NAME", "3P%", "Top 10 3-Point FG%")
    plot_bar(top10_ORTG_df, "FULL NAME", "ORTG", "Top 10 ORTG")
