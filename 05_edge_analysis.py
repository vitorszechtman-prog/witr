"""
05_edge_analysis.py
===================
Analyze the spread between our win probability model and Kalshi's live prices.

This script:
1. Loads logged Kalshi prices + NBA game states (synced by timestamp)
2. Looks up model fair value for each game state
3. Computes edge = model_prob - kalshi_price
4. Analyzes: How often is there edge? How large? Would trading it be profitable?
5. Kelly criterion sizing for optimal bet allocation
6. Team-aware market matching (maps Kalshi tickers to specific NBA games)

Usage:
    python 05_edge_analysis.py
    python 05_edge_analysis.py --threshold 5 --visualize
    python 05_edge_analysis.py --kelly --bankroll 1000

Output:
    analysis/edge_distribution.png
    analysis/edge_summary.csv
    analysis/pnl_simulation.png
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from utils import DATA_DIR, PROJECT_ROOT, read_jsonl, TEAM_NAME_TO_TRICODE


def load_kalshi_prices() -> pd.DataFrame:
    """Load logged Kalshi prices."""
    filepath = DATA_DIR / "kalshi_nba_prices.jsonl"
    if not filepath.exists():
        logger.error(f"No Kalshi price data found at {filepath}")
        logger.info("Run 03_kalshi_live_logger.py during live games first.")
        sys.exit(1)
    
    records = read_jsonl(filepath)
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded {len(df):,} Kalshi price snapshots")
    return df


def load_game_states() -> pd.DataFrame:
    """Load logged NBA game states."""
    filepath = DATA_DIR / "nba_game_state.jsonl"
    if not filepath.exists():
        logger.error(f"No NBA game state data found at {filepath}")
        logger.info("Run 04_nba_live_game_state.py during live games first.")
        sys.exit(1)
    
    records = read_jsonl(filepath)
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded {len(df):,} NBA game-state snapshots")
    return df


def load_win_prob_table() -> pd.DataFrame:
    """Load the empirical win probability lookup table."""
    filepath = DATA_DIR / "win_prob_empirical.parquet"
    if not filepath.exists():
        logger.error("Win probability table not found. Run 02_build_win_prob_model.py first.")
        sys.exit(1)
    return pd.read_parquet(filepath)


def load_win_prob_model():
    """Load the fitted logistic model."""
    filepath = DATA_DIR / "win_prob_model.pkl"
    if not filepath.exists():
        return None
    with open(filepath, "rb") as f:
        return pickle.load(f)


def lookup_fair_value(
    table: pd.DataFrame,
    margin: int,
    seconds_remaining: int,
    time_bucket_size: int = 30,
) -> float | None:
    """
    Look up fair value from the empirical table.
    Uses nearest-bucket matching.
    """
    # Round to nearest buckets
    margin_clipped = np.clip(margin, -30, 30)
    time_bucket = round(seconds_remaining / time_bucket_size) * time_bucket_size
    
    match = table[
        (table["home_margin"] == margin_clipped)
        & (table["seconds_remaining"] == time_bucket)
    ]
    
    if len(match) == 0:
        return None
    return float(match.iloc[0]["win_prob"])


def model_fair_value(model, margin: int, seconds_remaining: int) -> float:
    """Get fair value from the logistic model."""
    time_frac = seconds_remaining / 2880
    X = np.array([[margin, time_frac, margin * time_frac]])
    return float(model.predict_proba(X)[0, 1])


def extract_teams_from_kalshi(row: pd.Series) -> tuple[str, str] | None:
    """Extract team tricodes from a Kalshi market row."""
    # Check if already annotated
    if "home_team" in row and pd.notna(row.get("home_team")):
        return (str(row["home_team"]), str(row.get("away_team", "")))

    search_text = f"{row.get('title', '')} {row.get('ticker', '')}".upper()

    matched = []
    for name, tricode in TEAM_NAME_TO_TRICODE.items():
        if name in search_text and tricode not in matched:
            matched.append(tricode)

    if len(matched) >= 2:
        return (matched[0], matched[1])
    return None


def merge_kalshi_and_game_state(
    kalshi_df: pd.DataFrame,
    game_state_df: pd.DataFrame,
    max_time_diff_sec: int = 15,
) -> pd.DataFrame:
    """
    Merge Kalshi prices with NBA game states by timestamp + team matching.

    Two-pass strategy:
    1. If Kalshi markets have team annotations, merge by team + closest timestamp
    2. Fallback: merge by closest timestamp only (for single-game windows)
    """
    logger.info("Merging Kalshi prices with game states...")

    kalshi_df = kalshi_df.sort_values("timestamp").copy()
    game_state_df = game_state_df.sort_values("timestamp").copy()

    # Try to extract teams from Kalshi data
    kalshi_df["_extracted_teams"] = kalshi_df.apply(extract_teams_from_kalshi, axis=1)
    has_teams = kalshi_df["_extracted_teams"].notna()

    merged_parts = []

    # Pass 1: Team-aware merge
    if has_teams.any():
        team_kalshi = kalshi_df[has_teams].copy()
        # For each Kalshi row with team info, find the matching NBA game
        matched_rows = []
        for _, krow in team_kalshi.iterrows():
            teams = krow["_extracted_teams"]
            ts = krow["timestamp"]

            # Find NBA game state for these teams near this timestamp
            team_mask = (
                (game_state_df["home_team"].isin(teams))
                & (game_state_df["away_team"].isin(teams))
            )
            candidates = game_state_df[team_mask]
            if len(candidates) == 0:
                continue

            # Find nearest timestamp
            time_diffs = (candidates["timestamp"] - ts).abs()
            nearest_idx = time_diffs.idxmin()
            if time_diffs[nearest_idx].total_seconds() > max_time_diff_sec:
                continue

            nba_row = candidates.loc[nearest_idx]
            merged_row = {**krow.to_dict(), **{f"{k}_nba": v for k, v in nba_row.to_dict().items()}}
            # Use NBA data directly for key fields
            merged_row["home_margin"] = nba_row["home_margin"]
            merged_row["seconds_remaining"] = nba_row["seconds_remaining"]
            merged_row["home_team_matched"] = nba_row["home_team"]
            merged_row["away_team_matched"] = nba_row["away_team"]
            matched_rows.append(merged_row)

        if matched_rows:
            team_merged = pd.DataFrame(matched_rows)
            merged_parts.append(team_merged)
            logger.info(f"  Team-aware merge: {len(team_merged):,} rows")

    # Pass 2: Timestamp-only merge for unmatched rows
    unmatched_kalshi = kalshi_df[~has_teams] if has_teams.any() else kalshi_df
    if len(unmatched_kalshi) > 0:
        ts_merged = pd.merge_asof(
            unmatched_kalshi,
            game_state_df,
            on="timestamp",
            tolerance=pd.Timedelta(seconds=max_time_diff_sec),
            direction="nearest",
            suffixes=("_kalshi", "_nba"),
        )
        ts_merged = ts_merged.dropna(subset=["home_margin", "seconds_remaining"])
        if len(ts_merged) > 0:
            merged_parts.append(ts_merged)
            logger.info(f"  Timestamp-only merge: {len(ts_merged):,} rows")

    if not merged_parts:
        logger.warning("  No rows merged!")
        return pd.DataFrame()

    merged = pd.concat(merged_parts, ignore_index=True)
    merged.drop(columns=["_extracted_teams"], errors="ignore", inplace=True)
    logger.info(f"  Total merged: {len(merged):,} rows")

    return merged


def compute_edge(
    merged_df: pd.DataFrame,
    table: pd.DataFrame,
    model=None,
) -> pd.DataFrame:
    """
    Compute edge (model_prob - kalshi_price) for each snapshot.
    """
    logger.info("Computing edge for each snapshot...")
    
    edges = []
    for _, row in merged_df.iterrows():
        margin = int(row["home_margin"])
        secs = int(row["seconds_remaining"])
        
        # Kalshi price (convert from cents to probability)
        kalshi_mid = row.get("yes_mid")
        if kalshi_mid is None or pd.isna(kalshi_mid):
            continue
        kalshi_prob = kalshi_mid / 100.0  # cents → probability
        
        # Model fair values
        emp_fv = lookup_fair_value(table, margin, secs)
        model_fv = model_fair_value(model, margin, secs) if model else None
        
        # Use the average of both models as fair value
        if emp_fv is not None and model_fv is not None:
            fair_value = (emp_fv + model_fv) / 2
        elif emp_fv is not None:
            fair_value = emp_fv
        elif model_fv is not None:
            fair_value = model_fv
        else:
            continue
        
        edge = fair_value - kalshi_prob
        
        edges.append({
            "timestamp": row["timestamp"],
            "ticker": row.get("ticker", ""),
            "margin": margin,
            "seconds_remaining": secs,
            "kalshi_prob": kalshi_prob,
            "empirical_fv": emp_fv,
            "model_fv": model_fv,
            "fair_value": fair_value,
            "edge": edge,
            "abs_edge": abs(edge),
            "kalshi_bid": row.get("yes_bid"),
            "kalshi_ask": row.get("yes_ask"),
        })
    
    edges_df = pd.DataFrame(edges)
    logger.info(f"  Computed edge for {len(edges_df):,} snapshots")
    return edges_df


def analyze_edge(edges_df: pd.DataFrame, threshold_cents: float = 5.0):
    """
    Analyze edge distribution and simulate PnL.
    """
    threshold = threshold_cents / 100.0
    
    logger.info(f"\n{'='*60}")
    logger.info("EDGE ANALYSIS SUMMARY")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nTotal observations: {len(edges_df):,}")
    logger.info(f"Mean edge: {edges_df['edge'].mean():.4f} ({edges_df['edge'].mean()*100:.2f}¢)")
    logger.info(f"Std edge:  {edges_df['edge'].std():.4f}")
    logger.info(f"Mean |edge|: {edges_df['abs_edge'].mean():.4f} ({edges_df['abs_edge'].mean()*100:.2f}¢)")
    
    # Edge distribution by bucket
    logger.info(f"\nEdge distribution:")
    for bucket in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]:
        count = (edges_df["abs_edge"] >= bucket).sum()
        pct = count / len(edges_df) * 100
        logger.info(f"  |Edge| >= {bucket*100:.0f}¢: {count:,} ({pct:.1f}%)")
    
    # Opportunities above threshold
    opps = edges_df[edges_df["abs_edge"] >= threshold]
    logger.info(f"\nOpportunities (|edge| >= {threshold_cents:.0f}¢): {len(opps):,}")
    
    if len(opps) > 0:
        logger.info(f"  Avg edge when triggered: {opps['edge'].mean()*100:.2f}¢")
        logger.info(f"  Avg |edge| when triggered: {opps['abs_edge'].mean()*100:.2f}¢")
        
        # Where do opportunities cluster?
        logger.info(f"\n  By game phase:")
        phase_bins = [(0, 600, "Last 10 min"), (600, 1440, "Mid game"), (1440, 2880, "First half")]
        for lo, hi, label in phase_bins:
            phase_opps = opps[(opps["seconds_remaining"] >= lo) & (opps["seconds_remaining"] < hi)]
            if len(phase_opps) > 0:
                logger.info(f"    {label}: {len(phase_opps):,} opps, avg |edge| = {phase_opps['abs_edge'].mean()*100:.1f}¢")
        
        # By margin range
        logger.info(f"\n  By margin range:")
        margin_bins = [(-30, -10, "Down big"), (-10, -3, "Down moderate"), (-3, 3, "Close game"), (3, 10, "Up moderate"), (10, 30, "Up big")]
        for lo, hi, label in margin_bins:
            margin_opps = opps[(opps["margin"] >= lo) & (opps["margin"] < hi)]
            if len(margin_opps) > 0:
                logger.info(f"    {label}: {len(margin_opps):,} opps, avg edge = {margin_opps['edge'].mean()*100:+.1f}¢")
    
    # Tail analysis (this is your core strategy)
    logger.info(f"\n{'='*60}")
    logger.info("TAIL ANALYSIS (Kalshi price near 0 or 100)")
    logger.info(f"{'='*60}")
    
    tails = edges_df[
        (edges_df["kalshi_prob"] <= 0.10) | (edges_df["kalshi_prob"] >= 0.90)
    ]
    
    if len(tails) > 0:
        logger.info(f"Tail observations (Kalshi < 10¢ or > 90¢): {len(tails):,}")
        logger.info(f"  Avg edge in tails: {tails['edge'].mean()*100:+.2f}¢")
        logger.info(f"  Avg |edge| in tails: {tails['abs_edge'].mean()*100:.2f}¢")
        
        # Extreme tails
        extreme = edges_df[
            (edges_df["kalshi_prob"] <= 0.05) | (edges_df["kalshi_prob"] >= 0.95)
        ]
        if len(extreme) > 0:
            logger.info(f"\nExtreme tails (< 5¢ or > 95¢): {len(extreme):,}")
            logger.info(f"  Avg edge: {extreme['edge'].mean()*100:+.2f}¢")
    
    return edges_df


def kelly_fraction(edge: float, odds: float) -> float:
    """
    Compute Kelly criterion fraction for a binary bet.

    For a binary market at price p (implied probability):
        - If we think true prob is q, edge = q - p
        - Odds of YES contract: payout = 1/p, cost = p → net odds = (1-p)/p
        - Kelly fraction f* = (q * (1-p)/p - (1-q)) / ((1-p)/p)
                            = (q - p) / (1 - p)  [for YES bets]
        - For NO bets: f* = (p - q) / p

    Returns fraction of bankroll to bet (0 = no bet, capped at 0.25).
    """
    if edge > 0:
        # Buy YES: model says underpriced
        p = odds  # market price
        q = odds + edge  # our probability
        if p >= 1.0 or p <= 0.0:
            return 0.0
        f = (q - p) / (1 - p)
    else:
        # Buy NO: model says overpriced
        p = odds
        q = odds + edge  # our probability (lower than market)
        if p >= 1.0 or p <= 0.0:
            return 0.0
        f = (p - q) / p

    # Half-Kelly for safety, cap at 25%
    f = max(0.0, f) * 0.5
    return min(f, 0.25)


def simulate_pnl(
    edges_df: pd.DataFrame,
    threshold_cents: float = 5.0,
    bet_size: float = 10.0,  # dollars per trade (flat sizing)
    bankroll: float = None,  # if set, use Kelly sizing
):
    """
    PnL simulation with flat sizing and optional Kelly criterion.

    Two modes:
    1. Flat: bet fixed $ per trade when |edge| > threshold
    2. Kelly: bet Kelly-optimal fraction of bankroll per trade

    NOTE: This is a rough simulation. Real PnL depends on:
    - Execution at bid/ask (not mid)
    - Timing of entry and exit
    - Kalshi fees (typically ~2% on profits)
    - Contract resolution
    """
    threshold = threshold_cents / 100.0

    trades = edges_df[edges_df["abs_edge"] >= threshold].copy()

    if len(trades) == 0:
        logger.info("No trades triggered at this threshold")
        return None

    # Flat sizing
    trades["theoretical_pnl"] = trades["edge"] * bet_size
    trades["conservative_pnl"] = trades["edge"] * bet_size * 0.5

    logger.info(f"\n{'='*60}")
    logger.info("PnL SIMULATION — FLAT SIZING")
    logger.info(f"{'='*60}")
    logger.info(f"Threshold: {threshold_cents:.0f}¢ | Bet size: ${bet_size:.0f}")
    logger.info(f"Total trades: {len(trades):,}")
    logger.info(f"Theoretical total PnL: ${trades['theoretical_pnl'].sum():.2f}")
    logger.info(f"Conservative total PnL: ${trades['conservative_pnl'].sum():.2f}")
    logger.info(f"Avg PnL per trade: ${trades['conservative_pnl'].mean():.4f}")

    # Kelly sizing
    if bankroll is not None and bankroll > 0:
        logger.info(f"\n{'='*60}")
        logger.info("PnL SIMULATION — KELLY CRITERION")
        logger.info(f"{'='*60}")
        logger.info(f"Starting bankroll: ${bankroll:.0f}")

        trades["kelly_f"] = trades.apply(
            lambda r: kelly_fraction(r["edge"], r["kalshi_prob"]), axis=1
        )
        trades["kelly_bet"] = trades["kelly_f"] * bankroll
        trades["kelly_pnl"] = trades["edge"] * trades["kelly_bet"] * 0.5  # conservative

        # Simulate sequential growth
        running_bankroll = bankroll
        bankroll_history = [bankroll]
        for _, trade in trades.iterrows():
            f = kelly_fraction(trade["edge"], trade["kalshi_prob"])
            size = f * running_bankroll
            pnl = trade["edge"] * size * 0.5  # conservative capture
            running_bankroll += pnl
            bankroll_history.append(running_bankroll)

        trades["cumulative_kelly_pnl"] = trades["kelly_pnl"].cumsum()

        logger.info(f"Total Kelly trades: {len(trades):,}")
        logger.info(f"Avg Kelly fraction: {trades['kelly_f'].mean():.2%}")
        logger.info(f"Max Kelly fraction: {trades['kelly_f'].max():.2%}")
        logger.info(f"Total Kelly PnL: ${trades['kelly_pnl'].sum():.2f}")
        logger.info(f"Final bankroll (sequential): ${running_bankroll:.2f}")
        logger.info(f"Return: {(running_bankroll / bankroll - 1):.1%}")

    return trades


def visualize_edge(edges_df: pd.DataFrame, output_dir: Path):
    """Generate edge analysis visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(exist_ok=True)

    # 1. Edge distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(edges_df["edge"] * 100, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Edge (cents)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Edge Distribution (Model - Kalshi)")
    
    axes[1].scatter(
        edges_df["kalshi_prob"] * 100,
        edges_df["fair_value"] * 100,
        alpha=0.1, s=2,
    )
    axes[1].plot([0, 100], [0, 100], "r--", alpha=0.5)
    axes[1].set_xlabel("Kalshi Price (cents)")
    axes[1].set_ylabel("Model Fair Value (cents)")
    axes[1].set_title("Model vs Kalshi: Scatter")
    
    plt.tight_layout()
    plt.savefig(output_dir / "edge_distribution.png", dpi=150)
    logger.info(f"Saved edge distribution → {output_dir / 'edge_distribution.png'}")
    plt.close()

    # 2. Edge heatmap by margin and time
    fig, ax = plt.subplots(figsize=(14, 8))
    
    edges_df["margin_bucket"] = (edges_df["margin"] // 3) * 3
    edges_df["time_bucket_min"] = (edges_df["seconds_remaining"] // 120) * 2
    
    pivot = edges_df.pivot_table(
        values="edge",
        index="margin_bucket",
        columns="time_bucket_min",
        aggfunc="mean",
    ) * 100  # convert to cents
    
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdBu_r", vmin=-10, vmax=10,
        origin="lower",
    )
    ax.set_xlabel("Minutes Remaining")
    ax.set_ylabel("Home Margin")
    ax.set_title("Average Edge (cents) by Game State\n(Blue = Kalshi underprices home, Red = overprices)")
    plt.colorbar(im, label="Edge (cents)")
    plt.tight_layout()
    plt.savefig(output_dir / "edge_heatmap.png", dpi=150)
    logger.info(f"Saved edge heatmap → {output_dir / 'edge_heatmap.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze model vs Kalshi edge")
    parser.add_argument(
        "--threshold", type=float, default=5.0, help="Edge threshold in cents"
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--kelly", action="store_true", help="Run Kelly criterion sizing simulation"
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Starting bankroll for Kelly simulation (default: $1000)",
    )
    args = parser.parse_args()

    # Load data
    kalshi_df = load_kalshi_prices()
    game_state_df = load_game_states()
    table = load_win_prob_table()
    model = load_win_prob_model()

    # Merge
    merged = merge_kalshi_and_game_state(kalshi_df, game_state_df)

    if len(merged) == 0:
        logger.error("No merged data available. Need both Kalshi and NBA game logs.")
        sys.exit(1)

    # Compute edge
    edges_df = compute_edge(merged, table, model)

    # Analyze
    analyze_edge(edges_df, threshold_cents=args.threshold)
    trades = simulate_pnl(
        edges_df,
        threshold_cents=args.threshold,
        bankroll=args.bankroll if args.kelly else None,
    )

    # Save
    analysis_dir = PROJECT_ROOT / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    edges_df.to_csv(analysis_dir / "edge_data.csv", index=False)
    logger.info(f"Saved edge data → {analysis_dir / 'edge_data.csv'}")

    if trades is not None:
        trades.to_csv(analysis_dir / "trades.csv", index=False)
        logger.info(f"Saved trades → {analysis_dir / 'trades.csv'}")

    if args.visualize:
        visualize_edge(edges_df, analysis_dir)

    logger.success("Edge analysis complete!")


if __name__ == "__main__":
    main()
