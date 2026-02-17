"""
07_dashboard.py - NBA Win Probability Explorer

Interactive Streamlit dashboard for exploring historical NBA win probability data.
Select teams, game situations, and team strength filters to see how often the
home team won in similar historical scenarios, with model predictions for comparison.

Usage:
    streamlit run 07_dashboard.py
"""

import pickle
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, NBA_TEAM_NAMES

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NBA Win Probability Explorer",
    page_icon="🏀",
    layout="wide",
)

SEASONS = [2022, 2023, 2024]
TEAM_OPTIONS = ["Any"] + sorted(NBA_TEAM_NAMES.keys())
TEAM_DISPLAY = {tc: f"{tc} ({names[0]})" for tc, names in NBA_TEAM_NAMES.items()}
TEAM_DISPLAY["Any"] = "Any"


# ── Cached Data Loading ──────────────────────────────────────────────────────


@st.cache_data
def load_game_states(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Load game-state snapshots for selected seasons."""
    dfs = []
    for s in seasons:
        path = DATA_DIR / f"game_states_{s}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


@st.cache_resource
def load_models() -> dict | None:
    """Load trained model pipelines."""
    path = DATA_DIR / "win_prob_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def model_predict(models: dict, margin: float, seconds_remaining: float,
                  row: pd.Series | None = None) -> dict:
    """Get model predictions at all available tiers. Mirrors 06_backtest.py."""
    time_frac = seconds_remaining / 2880
    result = {}

    if models.get("baseline"):
        X = np.array([[margin, time_frac, margin * time_frac]])
        result["Baseline"] = float(models["baseline"].predict_proba(X)[0, 1])

    if models.get("enhanced") and row is not None:
        cols = ["home_win_pct", "away_win_pct", "home_home_win_pct",
                "home_last10", "away_last10", "strength_diff"]
        if all(pd.notna(row.get(c)) for c in cols):
            X = np.array([[
                margin, time_frac, margin * time_frac,
                row["home_win_pct"], row["away_win_pct"], row["home_home_win_pct"],
                row["home_last10"], row["away_last10"], row["strength_diff"],
                row["strength_diff"] * time_frac,
                margin * row["strength_diff"],
            ]])
            result["Enhanced"] = float(models["enhanced"].predict_proba(X)[0, 1])

    if models.get("full") and row is not None:
        ext_cols = ["home_net_rating", "away_net_rating", "net_rating_diff",
                    "home_pace", "away_pace", "expected_pace"]
        if all(pd.notna(row.get(c)) for c in ext_cols):
            sd = row.get("strength_diff", 0)
            nrd = row["net_rating_diff"]
            ep = row["expected_pace"]
            X = np.array([[
                margin, time_frac, margin * time_frac,
                row["home_win_pct"], row["away_win_pct"], row["home_home_win_pct"],
                row["home_last10"], row["away_last10"], sd,
                row["home_net_rating"], row["away_net_rating"], nrd,
                row["home_pace"], row["away_pace"], ep,
                sd * time_frac,
                margin * sd,
                nrd * time_frac,
                margin * nrd,
                ep / 210.0 - 1.0,
            ]])
            result["Full"] = float(models["full"].predict_proba(X)[0, 1])

    if models.get("momentum") and row is not None:
        mom_cols = ["run_team", "run_length", "run_points",
                    "momentum", "scoring_burst", "margin_change_last5"]
        ext_cols = ["home_net_rating", "away_net_rating", "net_rating_diff",
                    "home_pace", "away_pace", "expected_pace"]
        if (all(c in row.index and pd.notna(row.get(c)) for c in mom_cols) and
                all(c in row.index and pd.notna(row.get(c)) for c in ext_cols)):
            sd = row.get("strength_diff", 0)
            nrd = row["net_rating_diff"]
            ep = row["expected_pace"]
            X = np.array([[
                margin, time_frac, margin * time_frac,
                row["home_win_pct"], row["away_win_pct"], row["home_home_win_pct"],
                row["home_last10"], row["away_last10"], sd,
                row["home_net_rating"], row["away_net_rating"], nrd,
                row["home_pace"], row["away_pace"], ep,
                sd * time_frac, margin * sd,
                nrd * time_frac, margin * nrd,
                ep / 210.0 - 1.0,
                row["run_team"], row["run_length"], row["run_points"],
                row["momentum"], row["scoring_burst"], row["margin_change_last5"],
                row["momentum"] * (1 - time_frac),
                row["run_points"] * (1 - time_frac),
                row["run_team"] * row["run_points"] * margin,
            ]])
            result["Momentum"] = float(models["momentum"].predict_proba(X)[0, 1])

    result["best"] = result.get("Momentum", result.get("Full", result.get("Enhanced", result.get("Baseline", 0.5))))
    return result


@st.cache_data
def load_sim_trades() -> pd.DataFrame:
    """Load paper trading simulation trades."""
    path = DATA_DIR / "kalshi_sim_trades.jsonl"
    if not path.exists():
        return pd.DataFrame()
    import json
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Filter game-state snapshots to those matching all criteria."""
    mask = pd.Series(True, index=df.index)

    # Team filters
    if f["home_team"] != "Any":
        mask &= df["home_team"] == f["home_team"]
    if f["away_team"] != "Any":
        mask &= df["away_team"] == f["away_team"]

    # Margin range
    mask &= df["home_margin"].between(f["margin_lo"], f["margin_hi"])

    # Time remaining (slider is in minutes, data is in seconds)
    secs_lo = int(f["time_lo"] * 60)
    secs_hi = int(f["time_hi"] * 60)
    mask &= df["seconds_remaining"].between(secs_lo, secs_hi)

    # L10 filters (slider is 0-10 integer, data is 0.0-1.0 fraction)
    eps = 0.01
    mask &= df["home_last10"].between(f["home_l10_lo"] / 10 - eps,
                                       f["home_l10_hi"] / 10 + eps)
    mask &= df["away_last10"].between(f["away_l10_lo"] / 10 - eps,
                                       f["away_l10_hi"] / 10 + eps)

    # Win% filters
    mask &= df["home_win_pct"].between(f["home_wpct_lo"], f["home_wpct_hi"])
    mask &= df["away_win_pct"].between(f["away_wpct_lo"], f["away_wpct_hi"])

    return df[mask]


def get_game_outcomes(filtered: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate snapshots to one row per game, keeping the first matching snapshot."""
    if filtered.empty:
        return pd.DataFrame()
    return filtered.groupby("game_id").first().reset_index()


def build_filter_dict_either(team_a: str, team_b: str, base: dict) -> tuple[dict, dict]:
    """Build two filter dicts for 'either direction' team search."""
    f1 = {**base, "home_team": team_a, "away_team": team_b}
    f2 = {**base, "home_team": team_b, "away_team": team_a}
    # Swap L10 and win% filters for the reversed matchup
    f2["home_l10_lo"], f2["away_l10_lo"] = base["away_l10_lo"], base["home_l10_lo"]
    f2["home_l10_hi"], f2["away_l10_hi"] = base["away_l10_hi"], base["home_l10_hi"]
    f2["home_wpct_lo"], f2["away_wpct_lo"] = base["away_wpct_lo"], base["home_wpct_lo"]
    f2["home_wpct_hi"], f2["away_wpct_hi"] = base["away_wpct_hi"], base["home_wpct_hi"]
    # Reverse margin for the swapped matchup
    f2["margin_lo"], f2["margin_hi"] = -base["margin_hi"], -base["margin_lo"]
    return f1, f2


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar() -> dict:
    st.sidebar.title("NBA Win Probability Explorer")

    # -- Teams --
    st.sidebar.subheader("Teams")
    team_a = st.sidebar.selectbox("Team A", TEAM_OPTIONS,
                                  format_func=lambda t: TEAM_DISPLAY[t])
    team_b = st.sidebar.selectbox("Team B", TEAM_OPTIONS,
                                  format_func=lambda t: TEAM_DISPLAY[t])
    home_choice = st.sidebar.radio(
        "Home team",
        ["Team A is Home", "Team B is Home", "Either direction"],
        index=0,
    )

    # -- Game Situation --
    st.sidebar.subheader("Game Situation")
    margin_range = st.sidebar.slider("Score Margin (Home - Away)", -40, 40, (-5, 5))
    time_range = st.sidebar.slider("Minutes Remaining", 0.0, 48.0, (4.0, 8.0), step=0.5)

    # -- Team Strength --
    st.sidebar.subheader("Team Strength")
    with st.sidebar.expander("L10 Record"):
        home_l10 = st.slider("Home Team L10 wins (out of 10)", 0, 10, (0, 10),
                              key="home_l10")
        away_l10 = st.slider("Away Team L10 wins (out of 10)", 0, 10, (0, 10),
                              key="away_l10")
    with st.sidebar.expander("Overall Win %"):
        home_wpct = st.slider("Home Team Win %", 0.0, 1.0, (0.0, 1.0), step=0.05,
                               key="home_wpct")
        away_wpct = st.slider("Away Team Win %", 0.0, 1.0, (0.0, 1.0), step=0.05,
                               key="away_wpct")

    # -- Seasons --
    st.sidebar.subheader("Seasons")
    seasons = st.sidebar.multiselect(
        "Include seasons",
        SEASONS,
        default=SEASONS,
        format_func=lambda s: f"{s}-{str(s+1)[-2:]}",
    )
    if not seasons:
        seasons = SEASONS

    return {
        "team_a": team_a,
        "team_b": team_b,
        "home_choice": home_choice,
        "margin_lo": margin_range[0],
        "margin_hi": margin_range[1],
        "time_lo": time_range[0],
        "time_hi": time_range[1],
        "home_l10_lo": home_l10[0],
        "home_l10_hi": home_l10[1],
        "away_l10_lo": away_l10[0],
        "away_l10_hi": away_l10[1],
        "home_wpct_lo": home_wpct[0],
        "home_wpct_hi": home_wpct[1],
        "away_wpct_lo": away_wpct[0],
        "away_wpct_hi": away_wpct[1],
        "seasons": tuple(seasons),
    }


# ── Charts ────────────────────────────────────────────────────────────────────


def chart_win_rate_by_margin(df: pd.DataFrame, margin_lo: int, margin_hi: int) -> alt.Chart:
    """Bar chart of empirical win rate by home margin value."""
    games = get_game_outcomes(df)
    if games.empty:
        return alt.Chart(pd.DataFrame()).mark_bar()

    grouped = (
        games.groupby("home_margin")
        .agg(wins=("home_win", "sum"), games=("home_win", "count"))
        .reset_index()
    )
    grouped["win_rate"] = grouped["wins"] / grouped["games"]
    grouped["in_range"] = grouped["home_margin"].between(margin_lo, margin_hi)
    # Trim to reasonable display range
    grouped = grouped[grouped["home_margin"].between(-35, 35)]

    base = alt.Chart(grouped).mark_bar(opacity=0.3, color="steelblue").encode(
        x=alt.X("home_margin:Q", title="Home Margin"),
        y=alt.Y("win_rate:Q", title="Home Win Rate", scale=alt.Scale(domain=[0, 1])),
        tooltip=["home_margin", "win_rate", "games"],
    ).transform_filter(alt.datum.in_range == False)  # noqa: E712

    highlight = alt.Chart(grouped).mark_bar(color="steelblue").encode(
        x=alt.X("home_margin:Q"),
        y=alt.Y("win_rate:Q"),
        tooltip=["home_margin", "win_rate", "games"],
    ).transform_filter(alt.datum.in_range == True)  # noqa: E712

    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        strokeDash=[4, 4], color="gray"
    ).encode(y="y:Q")

    return (base + highlight + rule).properties(
        title="Win Rate by Score Margin (highlighted = your filter)",
        height=300,
    )


def chart_win_rate_by_time(df: pd.DataFrame, time_lo_min: float, time_hi_min: float) -> alt.Chart:
    """Line chart of empirical win rate by time remaining (2-min buckets)."""
    games = get_game_outcomes(df)
    if games.empty:
        return alt.Chart(pd.DataFrame()).mark_line()

    games = games[games["seconds_remaining"] >= 0].copy()
    games["minutes_bucket"] = (games["seconds_remaining"] / 120).round() * 2

    grouped = (
        games.groupby("minutes_bucket")
        .agg(wins=("home_win", "sum"), games=("home_win", "count"))
        .reset_index()
    )
    grouped["win_rate"] = grouped["wins"] / grouped["games"]
    grouped["in_range"] = grouped["minutes_bucket"].between(time_lo_min, time_hi_min)

    line = alt.Chart(grouped).mark_line(point=True, color="steelblue").encode(
        x=alt.X("minutes_bucket:Q", title="Minutes Remaining", sort="descending"),
        y=alt.Y("win_rate:Q", title="Home Win Rate", scale=alt.Scale(domain=[0, 1])),
        tooltip=["minutes_bucket", "win_rate", "games"],
    )

    band = alt.Chart(pd.DataFrame({
        "x": [time_lo_min, time_hi_min],
    })).transform_fold(
        ["x"], as_=["key", "val"]
    )

    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        strokeDash=[4, 4], color="gray"
    ).encode(y="y:Q")

    # Highlight selected range with a rect
    rect = alt.Chart(pd.DataFrame({
        "x": [time_lo_min], "x2": [time_hi_min],
    })).mark_rect(opacity=0.15, color="orange").encode(
        x="x:Q", x2="x2:Q",
    )

    return (rect + line + rule).properties(
        title="Win Rate by Time Remaining (shaded = your filter)",
        height=300,
    )


def chart_outcomes(n_wins: int, n_games: int) -> alt.Chart:
    """Horizontal bar showing home wins vs away wins."""
    n_losses = n_games - n_wins
    data = pd.DataFrame({
        "outcome": ["Home Win", "Away Win"],
        "count": [n_wins, n_losses],
        "pct": [n_wins / n_games * 100 if n_games else 0,
                n_losses / n_games * 100 if n_games else 0],
    })
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("count:Q", title="Number of Games"),
        y=alt.Y("outcome:N", title=""),
        color=alt.Color("outcome:N", scale=alt.Scale(
            domain=["Home Win", "Away Win"],
            range=["#2ecc71", "#e74c3c"],
        ), legend=None),
        tooltip=["outcome", "count", alt.Tooltip("pct:Q", format=".1f", title="Pct")],
    ).properties(height=120, title="Outcome Breakdown")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    filters = render_sidebar()

    # Load data
    df = load_game_states(filters["seasons"])
    if df.empty:
        st.error("No game data found. Run 01_fetch_pbp_data.py and 02_build_win_prob_model.py first.")
        st.stop()

    models = load_models()

    # Resolve team A/B → home/away
    ta, tb = filters["team_a"], filters["team_b"]
    choice = filters["home_choice"]

    if choice == "Either direction" and ta != "Any" and tb != "Any":
        base_filters = {k: v for k, v in filters.items()
                        if k not in ("team_a", "team_b", "home_choice", "seasons")}
        base_filters["home_team"] = "Any"
        base_filters["away_team"] = "Any"
        f1, f2 = build_filter_dict_either(ta, tb, base_filters)
        filtered = pd.concat([apply_filters(df, f1), apply_filters(df, f2)], ignore_index=True)
        # For "either direction", flip home_win for reversed matchup so we track Team A's win
        # Actually, keep it as home perspective since that's what the data records
    else:
        if choice == "Team A is Home":
            home, away = ta, tb
        elif choice == "Team B is Home":
            home, away = tb, ta
        else:
            home, away = ta, tb

        f = {k: v for k, v in filters.items()
             if k not in ("team_a", "team_b", "home_choice", "seasons")}
        f["home_team"] = home
        f["away_team"] = away
        filtered = apply_filters(df, f)

    game_outcomes = get_game_outcomes(filtered)
    n_games = len(game_outcomes)
    n_wins = int(game_outcomes["home_win"].sum()) if n_games > 0 else 0
    win_rate = n_wins / n_games if n_games > 0 else 0.0

    # ── Hero Metrics ──────────────────────────────────────────────────────

    st.header("Win Probability Explorer")

    if n_games == 0:
        st.warning("No historical games match these filters. Try widening your ranges.")
        st.stop()

    ci_lo, ci_hi = wilson_ci(n_wins, n_games)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Empirical Home Win %", f"{win_rate:.1%}",
                delta=f"{win_rate - 0.5:+.1%} vs 50%")
    col2.metric("Matched Games", f"{n_games:,}")
    col3.metric("95% Confidence Interval", f"{ci_lo:.1%} – {ci_hi:.1%}")

    # Model prediction using median values from filtered data
    if models:
        median_row = filtered.median(numeric_only=True)
        mid_margin = median_row.get("home_margin", 0)
        mid_secs = median_row.get("seconds_remaining", 720)
        preds = model_predict(models, mid_margin, mid_secs, median_row)
        col4.metric("Model Prediction (best)", f"{preds['best']:.1%}",
                    delta=f"{preds['best'] - win_rate:+.1%} vs empirical")
    else:
        preds = {}
        col4.metric("Model Prediction", "N/A")

    if n_games < 10:
        st.warning(f"Only **{n_games}** games match — results may not be statistically meaningful. "
                   "Consider widening your filters.")
    elif n_games < 30:
        st.info(f"**{n_games}** games match — a reasonable sample, but confidence intervals are wide.")

    # ── Tabs ──────────────────────────────────────────────────────────────

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Historical Data", "Model Comparison", "Momentum Impact",
        "Kalshi Simulation", "Game Details",
    ])

    # -- Tab 1: Historical Data --
    with tab1:
        # Context charts: filter by time window only (varying margin)
        # and by margin window only (varying time)
        f_time_only = {k: v for k, v in filters.items()
                       if k not in ("team_a", "team_b", "home_choice", "seasons",
                                    "margin_lo", "margin_hi")}
        f_time_only["home_team"] = "Any"
        f_time_only["away_team"] = "Any"
        f_time_only["margin_lo"] = -60
        f_time_only["margin_hi"] = 60
        df_time_context = apply_filters(df, f_time_only)

        f_margin_only = {k: v for k, v in filters.items()
                         if k not in ("team_a", "team_b", "home_choice", "seasons",
                                      "time_lo", "time_hi")}
        f_margin_only["home_team"] = "Any"
        f_margin_only["away_team"] = "Any"
        f_margin_only["time_lo"] = 0.0
        f_margin_only["time_hi"] = 48.0
        df_margin_context = apply_filters(df, f_margin_only)

        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                chart_win_rate_by_margin(df_time_context,
                                         filters["margin_lo"], filters["margin_hi"]),
                use_container_width=True,
            )
        with c2:
            st.altair_chart(
                chart_win_rate_by_time(df_margin_context,
                                       filters["time_lo"], filters["time_hi"]),
                use_container_width=True,
            )

        st.altair_chart(chart_outcomes(n_wins, n_games), use_container_width=True)

        st.caption(
            f"Showing results from **{n_games:,}** unique games across "
            f"**{filtered.shape[0]:,}** game-state snapshots."
        )

    # -- Tab 2: Model Comparison --
    with tab2:
        if not models:
            st.warning("No trained models found. Run 02_build_win_prob_model.py first.")
        else:
            st.subheader("Empirical vs Model Predictions")
            st.caption(
                "Model inputs use the **median** values from your filtered game states. "
                "Empirical rate comes from actual game outcomes."
            )

            rows = [{"Source": f"Empirical ({n_games} games)",
                     "Win Probability": f"{win_rate:.1%}",
                     "": ""}]
            for tier in ["Baseline", "Enhanced", "Full"]:
                if tier in preds:
                    diff = preds[tier] - win_rate
                    rows.append({
                        "Source": f"Model: {tier}",
                        "Win Probability": f"{preds[tier]:.1%}",
                        "": f"({diff:+.1%} vs empirical)",
                    })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Visual comparison bars
            st.subheader("Visual Comparison")
            bar_data = [{"source": f"Empirical (N={n_games})", "prob": win_rate}]
            for tier in ["Baseline", "Enhanced", "Full"]:
                if tier in preds:
                    bar_data.append({"source": f"Model: {tier}", "prob": preds[tier]})

            bar_df = pd.DataFrame(bar_data)
            chart = alt.Chart(bar_df).mark_bar().encode(
                x=alt.X("prob:Q", title="Win Probability",
                         scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("source:N", title="", sort=None),
                color=alt.condition(
                    alt.datum.source == f"Empirical (N={n_games})",
                    alt.value("#2ecc71"),
                    alt.value("steelblue"),
                ),
                tooltip=[alt.Tooltip("prob:Q", format=".1%", title="Probability")],
            ).properties(height=40 * len(bar_data) + 40)

            st.altair_chart(chart, use_container_width=True)

            # Calibration note
            diff = abs(win_rate - preds.get("best", 0.5))
            if n_games >= 30 and diff < 0.03:
                st.success("Model and historical data are well-aligned (within 3 percentage points).")
            elif n_games >= 30 and diff < 0.10:
                st.info(f"Model diverges from empirical by {diff:.1%}. "
                        "This may reflect sample specifics or model smoothing.")
            elif n_games >= 30:
                st.warning(f"Model diverges from empirical by {diff:.1%}. "
                           "Investigate whether the model is miscalibrated for this scenario.")

            # Confidence interval context
            st.caption(
                f"95% CI for empirical rate: **{ci_lo:.1%} – {ci_hi:.1%}**. "
                f"Model prediction {'falls within' if ci_lo <= preds.get('best', 0.5) <= ci_hi else 'falls outside'} "
                "this interval."
            )

    # -- Tab 3: Momentum Impact --
    with tab3:
        st.subheader("Momentum & Scoring Run Analysis")

        has_mom = "momentum" in filtered.columns and filtered["momentum"].notna().any()
        has_mom_model = models and models.get("momentum") is not None

        if not has_mom:
            st.warning(
                "No momentum features in data. Re-run `01_fetch_pbp_data.py` and "
                "`02_build_win_prob_model.py` to generate momentum features."
            )
        else:
            st.caption(
                "Momentum features capture intra-game scoring runs. A team on a 10-0 run "
                "has positive momentum (+1 direction) which shifts win probability."
            )

            # Momentum distribution
            col_a, col_b = st.columns(2)
            with col_a:
                mom_games = get_game_outcomes(filtered)
                if not mom_games.empty and "run_points" in mom_games.columns:
                    # Win rate by scoring run magnitude
                    mom_games["run_bucket"] = pd.cut(
                        mom_games["run_points"].fillna(0).astype(float),
                        bins=[-1, 0, 4, 8, 12, 100],
                        labels=["No run", "1-4 pts", "5-8 pts", "9-12 pts", "13+"],
                    )
                    run_wr = (
                        mom_games.groupby("run_bucket", observed=True)
                        .agg(wins=("home_win", "sum"), games=("home_win", "count"))
                        .reset_index()
                    )
                    run_wr["win_rate"] = run_wr["wins"] / run_wr["games"]
                    run_wr["label"] = run_wr.apply(
                        lambda r: f"{r['run_bucket']} (n={r['games']})", axis=1
                    )

                    chart = alt.Chart(run_wr).mark_bar(color="coral").encode(
                        x=alt.X("label:N", title="Scoring Run Size", sort=None),
                        y=alt.Y("win_rate:Q", title="Home Win Rate", scale=alt.Scale(domain=[0, 1])),
                        tooltip=["label", "win_rate", "games"],
                    ).properties(title="Win Rate by Scoring Run Size", height=300)

                    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
                        strokeDash=[4, 4], color="gray"
                    ).encode(y="y:Q")

                    st.altair_chart(chart + rule, use_container_width=True)

            with col_b:
                if not mom_games.empty and "momentum" in mom_games.columns:
                    # Win rate by momentum direction
                    mom_games["mom_direction"] = mom_games["momentum"].apply(
                        lambda x: "Home momentum" if x > 0.2 else (
                            "Away momentum" if x < -0.2 else "Neutral"
                        ) if pd.notna(x) else "Neutral"
                    )
                    dir_wr = (
                        mom_games.groupby("mom_direction")
                        .agg(wins=("home_win", "sum"), games=("home_win", "count"))
                        .reset_index()
                    )
                    dir_wr["win_rate"] = dir_wr["wins"] / dir_wr["games"]

                    chart = alt.Chart(dir_wr).mark_bar().encode(
                        x=alt.X("mom_direction:N", title="Momentum Direction"),
                        y=alt.Y("win_rate:Q", title="Home Win Rate", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("mom_direction:N", scale=alt.Scale(
                            domain=["Away momentum", "Neutral", "Home momentum"],
                            range=["#e74c3c", "#95a5a6", "#2ecc71"],
                        ), legend=None),
                        tooltip=["mom_direction", "win_rate", "games"],
                    ).properties(title="Win Rate by Momentum Direction", height=300)

                    st.altair_chart(chart + rule, use_container_width=True)

            # Model comparison with and without momentum
            if has_mom_model:
                st.subheader("Momentum Model Impact")
                st.caption(
                    "Compare how the Momentum model tier differs from the Full model. "
                    "Larger differences mean the current scoring run is significantly "
                    "shifting the probability."
                )

                # Show model comparison for scenarios with big runs
                scenarios = []
                for rt, rp, mom_val, label in [
                    (0, 0, 0.0, "No run"),
                    (1, 6, 0.5, "Home 6-0 run"),
                    (1, 10, 0.8, "Home 10-0 run"),
                    (1, 15, 1.0, "Home 15-0 run"),
                    (-1, 6, -0.5, "Away 6-0 run"),
                    (-1, 10, -0.8, "Away 10-0 run"),
                    (-1, 15, -1.0, "Away 15-0 run"),
                ]:
                    mid_margin = median_row.get("home_margin", 0)
                    mid_secs = median_row.get("seconds_remaining", 720)

                    # Build a row with momentum features
                    mom_row = median_row.copy()
                    mom_row["run_team"] = rt
                    mom_row["run_length"] = abs(rp) // 2
                    mom_row["run_points"] = rp
                    mom_row["momentum"] = mom_val
                    mom_row["scoring_burst"] = int(rp >= 8)
                    mom_row["margin_change_last5"] = rp * rt

                    full_pred = preds.get("Full", preds.get("best", 0.5))
                    mom_pred = model_predict(models, mid_margin, mid_secs, mom_row)
                    scenarios.append({
                        "Scenario": label,
                        "Full Model": f"{full_pred:.1%}",
                        "Momentum Model": f"{mom_pred.get('Momentum', full_pred):.1%}",
                        "Difference": f"{(mom_pred.get('Momentum', full_pred) - full_pred):+.1%}",
                    })

                st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

    # -- Tab 4: Kalshi Simulation --
    with tab4:
        st.subheader("Kalshi Paper Trading Performance")

        sim_trades = load_sim_trades()

        if sim_trades.empty:
            st.info(
                "No paper trading data yet. Start collecting on Thursday when games resume:\n\n"
                "```bash\n"
                "# Start live paper trading (run during games)\n"
                "python 08_kalshi_sim.py --live\n\n"
                "# Or replay from logged data\n"
                "python 08_kalshi_sim.py --replay\n\n"
                "# Generate report\n"
                "python 08_kalshi_sim.py --report\n"
                "```\n\n"
                "The simulator tracks:\n"
                "- **Score delay detection**: skips trades when our feed is behind the market\n"
                "- **Timeout signals**: upweights trades during timeouts (known score)\n"
                "- **Momentum-aware edge**: uses scoring run features in the model\n"
                "- **Paper P&L**: tracks hypothetical profit/loss with Kelly sizing"
            )
        else:
            settled = sim_trades[sim_trades["status"] == "settled"].copy()

            if settled.empty:
                st.info("Trades recorded but none settled yet (games still in progress).")
            else:
                # Hero metrics
                c1, c2, c3, c4 = st.columns(4)
                total_pnl = settled["realized_pnl"].sum()
                win_rate = (settled["realized_pnl"] > 0).mean()
                c1.metric("Total Trades", f"{len(settled)}")
                c2.metric("Win Rate", f"{win_rate:.1%}")
                c3.metric("Total P&L", f"${total_pnl:+.2f}")
                c4.metric("Avg Edge", f"{settled['abs_edge'].mean()*100:.1f}¢")

                # Cumulative P&L chart
                settled = settled.sort_values("timestamp")
                settled["cum_pnl"] = settled["realized_pnl"].cumsum()

                pnl_chart = alt.Chart(settled.reset_index()).mark_line(color="green").encode(
                    x=alt.X("index:Q", title="Trade #"),
                    y=alt.Y("cum_pnl:Q", title="Cumulative P&L ($)"),
                    tooltip=["index", "cum_pnl", "home_team", "away_team",
                             "direction", "realized_pnl"],
                ).properties(title="Paper Trading P&L Curve", height=300)

                zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                    strokeDash=[4, 4], color="red"
                ).encode(y="y:Q")

                st.altair_chart(pnl_chart + zero, use_container_width=True)

                # By confidence level
                st.subheader("Performance by Signal Type")
                st.caption(
                    "**Timeout** = score is known (highest confidence). "
                    "**Fresh** = score just changed. "
                    "**Normal** = routine poll. "
                    "**Stale** = suspected feed delay (no trades taken)."
                )

                if "confidence" in settled.columns:
                    conf_stats = (
                        settled.groupby("confidence")
                        .agg(
                            trades=("realized_pnl", "count"),
                            wins=("realized_pnl", lambda x: (x > 0).sum()),
                            pnl=("realized_pnl", "sum"),
                            avg_edge=("abs_edge", "mean"),
                        )
                        .reset_index()
                    )
                    conf_stats["win_rate"] = conf_stats["wins"] / conf_stats["trades"]
                    conf_stats["pnl"] = conf_stats["pnl"].round(2)
                    conf_stats["avg_edge"] = (conf_stats["avg_edge"] * 100).round(1)
                    conf_stats["win_rate"] = conf_stats["win_rate"].apply(lambda x: f"{x:.0%}")
                    conf_stats.columns = ["Signal", "Trades", "Wins", "P&L ($)", "Avg Edge (¢)", "Win Rate"]
                    st.dataframe(conf_stats, use_container_width=True, hide_index=True)

                # Recent trades table
                st.subheader("Recent Trades")
                recent = settled.tail(20).sort_values("timestamp", ascending=False)
                display = recent[[
                    "timestamp", "home_team", "away_team", "margin",
                    "direction", "kalshi_mid", "model_prob", "edge",
                    "confidence", "bet_size", "realized_pnl",
                ]].copy()
                display["model_prob"] = display["model_prob"].apply(lambda x: f"{x:.1%}")
                display["edge"] = display["edge"].apply(lambda x: f"{x*100:+.1f}¢")
                display["realized_pnl"] = display["realized_pnl"].apply(lambda x: f"${x:+.2f}")
                display.columns = [
                    "Time", "Home", "Away", "Margin", "Dir", "Kalshi (¢)",
                    "Model", "Edge", "Signal", "Bet ($)", "P&L",
                ]
                st.dataframe(display, use_container_width=True, hide_index=True)

    # -- Tab 5: Game Details --
    with tab5:
        st.subheader(f"Matching Games ({n_games:,})")

        display_cols = [
            "game_id", "home_team", "away_team",
            "home_score", "away_score", "home_margin",
            "period", "clock", "seconds_remaining",
            "home_last10", "away_last10",
            "home_win_pct", "away_win_pct",
            "home_win",
        ]
        available = [c for c in display_cols if c in game_outcomes.columns]
        display_df = game_outcomes[available].copy()

        # Format for readability
        rename = {
            "game_id": "Game ID",
            "home_team": "Home",
            "away_team": "Away",
            "home_score": "Home Pts",
            "away_score": "Away Pts",
            "home_margin": "Margin",
            "period": "Period",
            "clock": "Clock",
            "seconds_remaining": "Secs Left",
            "home_last10": "Home L10",
            "away_last10": "Away L10",
            "home_win_pct": "Home Win%",
            "away_win_pct": "Away Win%",
            "home_win": "Home Won",
        }
        display_df = display_df.rename(columns=rename)
        if "Home Won" in display_df.columns:
            display_df["Home Won"] = display_df["Home Won"].map({1: "Yes", 0: "No"})
        if "Home L10" in display_df.columns:
            display_df["Home L10"] = display_df["Home L10"].apply(lambda x: f"{x:.0%}")
        if "Away L10" in display_df.columns:
            display_df["Away L10"] = display_df["Away L10"].apply(lambda x: f"{x:.0%}")
        if "Home Win%" in display_df.columns:
            display_df["Home Win%"] = display_df["Home Win%"].apply(lambda x: f"{x:.0%}")
        if "Away Win%" in display_df.columns:
            display_df["Away Win%"] = display_df["Away Win%"].apply(lambda x: f"{x:.0%}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download
        csv = game_outcomes[available].to_csv(index=False)
        st.download_button("Download CSV", csv, "filtered_games.csv", "text/csv")


if __name__ == "__main__":
    main()
