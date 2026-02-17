"""
06_backtest.py
==============
Historical backtesting framework for the win probability model.

Replays historical NBA games tick-by-tick, computing:
1. Model win probability at each game state
2. Simulated "market" prices (using lagged empirical table as proxy)
3. Edge opportunities and theoretical PnL

This lets us evaluate the model WITHOUT needing historical Kalshi data.
The simulated market uses the empirical table with noise/lag as a proxy
for how a naive market might price games.

Usage:
    python 06_backtest.py                           # backtest all seasons
    python 06_backtest.py --season 2024             # single season
    python 06_backtest.py --season 2024 --visualize # with charts
    python 06_backtest.py --kelly --bankroll 1000   # Kelly sizing

Output:
    analysis/backtest_results.csv
    analysis/backtest_summary.txt
    analysis/backtest_results.png
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from utils import DATA_DIR, PROJECT_ROOT


def load_game_states(season: int = None) -> pd.DataFrame:
    """Load historical game state data."""
    if season:
        files = [DATA_DIR / f"game_states_{season}.parquet"]
    else:
        files = sorted(DATA_DIR.glob("game_states_*.parquet"))

    if not files or not files[0].exists():
        logger.error("No game state files found. Run 01_fetch_pbp_data.py first.")
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in files if f.exists()]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} game states from {len(dfs)} season(s)")
    return df


def load_models():
    """Load the trained win probability models."""
    model_path = DATA_DIR / "win_prob_model.pkl"
    if not model_path.exists():
        logger.error("No model found. Run 02_build_win_prob_model.py first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_empirical_table() -> pd.DataFrame:
    """Load the empirical win probability lookup table."""
    table_path = DATA_DIR / "win_prob_empirical.parquet"
    if not table_path.exists():
        logger.error("No empirical table found. Run 02_build_win_prob_model.py first.")
        sys.exit(1)
    return pd.read_parquet(table_path)


def simulate_market_price(
    margin: int,
    seconds_remaining: int,
    table: pd.DataFrame,
    noise_std: float = 0.03,
    lag_seconds: int = 30,
) -> float:
    """
    Simulate what a market price might look like.

    Uses the empirical table (a different model than the logistic)
    with added noise and a time lag to simulate market inefficiency.
    """
    lagged_seconds = seconds_remaining + lag_seconds

    margin_clipped = np.clip(margin, -30, 30)
    time_bucket = round(lagged_seconds / 30) * 30

    match = table[
        (table["home_margin"] == margin_clipped)
        & (table["seconds_remaining"] == time_bucket)
    ]

    if len(match) == 0:
        if seconds_remaining <= 0:
            base = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        else:
            k = 0.15 + 0.85 * (1 - seconds_remaining / 2880)
            z = margin * k * 0.3
            base = 1 / (1 + np.exp(-z))
    else:
        base = float(match.iloc[0]["win_prob"])

    noisy = base + np.random.normal(0, noise_std)
    return np.clip(noisy, 0.01, 0.99)


def model_predict(models: dict, margin: int, seconds_remaining: int, row: pd.Series = None) -> dict:
    """Get model predictions at all available tiers."""
    time_frac = seconds_remaining / 2880
    result = {}

    # Baseline
    if models.get("baseline"):
        X = np.array([[margin, time_frac, margin * time_frac]])
        result["baseline"] = float(models["baseline"].predict_proba(X)[0, 1])

    # Enhanced
    if models.get("enhanced") and row is not None:
        cols = ["home_win_pct", "away_win_pct", "home_home_win_pct",
                "home_last10", "away_last10", "strength_diff"]
        if all(c in row.index and pd.notna(row.get(c)) for c in cols):
            X = np.array([[
                margin, time_frac, margin * time_frac,
                row["home_win_pct"], row["away_win_pct"], row["home_home_win_pct"],
                row["home_last10"], row["away_last10"], row["strength_diff"],
                row["strength_diff"] * time_frac,
                margin * row["strength_diff"],
            ]])
            result["enhanced"] = float(models["enhanced"].predict_proba(X)[0, 1])

    # Full
    if models.get("full") and row is not None:
        ext_cols = ["home_net_rating", "away_net_rating", "net_rating_diff",
                    "home_pace", "away_pace", "expected_pace"]
        if all(c in row.index and pd.notna(row.get(c)) for c in ext_cols):
            strength_diff = row.get("strength_diff", 0)
            net_rating_diff = row["net_rating_diff"]
            expected_pace = row["expected_pace"]
            X = np.array([[
                margin, time_frac, margin * time_frac,
                row["home_win_pct"], row["away_win_pct"], row["home_home_win_pct"],
                row["home_last10"], row["away_last10"], strength_diff,
                row["home_net_rating"], row["away_net_rating"], net_rating_diff,
                row["home_pace"], row["away_pace"], expected_pace,
                strength_diff * time_frac,
                margin * strength_diff,
                net_rating_diff * time_frac,
                margin * net_rating_diff,
                expected_pace / 210.0 - 1.0,
            ]])
            result["full"] = float(models["full"].predict_proba(X)[0, 1])

    # Momentum
    if models.get("momentum") and row is not None:
        mom_cols = ["run_team", "run_length", "run_points",
                    "momentum", "scoring_burst", "margin_change_last5"]
        ext_cols = ["home_net_rating", "away_net_rating", "net_rating_diff",
                    "home_pace", "away_pace", "expected_pace"]
        if (all(c in row.index and pd.notna(row.get(c)) for c in mom_cols) and
                all(c in row.index and pd.notna(row.get(c)) for c in ext_cols)):
            strength_diff = row.get("strength_diff", 0)
            net_rating_diff = row["net_rating_diff"]
            expected_pace = row["expected_pace"]
            X = np.array([[
                margin, time_frac, margin * time_frac,
                row["home_win_pct"], row["away_win_pct"], row["home_home_win_pct"],
                row["home_last10"], row["away_last10"], strength_diff,
                row["home_net_rating"], row["away_net_rating"], net_rating_diff,
                row["home_pace"], row["away_pace"], expected_pace,
                strength_diff * time_frac,
                margin * strength_diff,
                net_rating_diff * time_frac,
                margin * net_rating_diff,
                expected_pace / 210.0 - 1.0,
                row["run_team"], row["run_length"], row["run_points"],
                row["momentum"], row["scoring_burst"], row["margin_change_last5"],
                row["momentum"] * (1 - time_frac),
                row["run_points"] * (1 - time_frac),
                row["run_team"] * row["run_points"] * margin,
            ]])
            result["momentum"] = float(models["momentum"].predict_proba(X)[0, 1])

    result["best"] = result.get("momentum", result.get("full", result.get("enhanced", result.get("baseline", 0.5))))
    return result


def kelly_fraction(edge: float, market_price: float) -> float:
    """Half-Kelly fraction, capped at 25%."""
    if edge > 0:
        if market_price >= 1.0 or market_price <= 0.0:
            return 0.0
        f = edge / (1 - market_price)
    else:
        if market_price >= 1.0 or market_price <= 0.0:
            return 0.0
        f = (-edge) / market_price
    f = max(0.0, f) * 0.5
    return min(f, 0.25)


def run_backtest(
    df: pd.DataFrame,
    models: dict,
    table: pd.DataFrame,
    threshold_cents: float = 5.0,
    noise_std: float = 0.03,
    market_lag_sec: int = 30,
    sample_interval: int = 5,
) -> pd.DataFrame:
    """
    Run the full backtest across all games in the dataset.

    For each sampled game state:
    1. Compute model fair value
    2. Simulate a market price
    3. Compute edge
    4. Record whether a trade would have been triggered
    """
    threshold = threshold_cents / 100.0

    game_ids = df["game_id"].unique()
    logger.info(f"Backtesting {len(game_ids)} games, sampling every {sample_interval}th play...")
    logger.info(f"  Market sim: noise_std={noise_std}, lag={market_lag_sec}s")
    logger.info(f"  Edge threshold: {threshold_cents:.0f} cents")

    np.random.seed(42)
    records = []

    for gi, game_id in enumerate(game_ids):
        game_df = df[df["game_id"] == game_id].sort_values("seconds_remaining", ascending=False)

        if len(game_df) == 0:
            continue

        sampled = game_df.iloc[::sample_interval]
        home_win = int(sampled.iloc[0]["home_win"])

        for _, play in sampled.iterrows():
            margin = int(play["home_margin"])
            secs = int(play["seconds_remaining"])

            if secs < 0:
                continue

            preds = model_predict(models, margin, secs, play)

            market_price = simulate_market_price(
                margin, secs, table,
                noise_std=noise_std,
                lag_seconds=market_lag_sec,
            )

            model_prob = preds["best"]
            edge = model_prob - market_price

            trade = abs(edge) >= threshold
            trade_direction = "YES" if edge > 0 else "NO" if edge < 0 else None

            if trade and trade_direction:
                if trade_direction == "YES":
                    realized_pnl = (1.0 - market_price) if home_win else (-market_price)
                else:
                    realized_pnl = market_price if not home_win else (-(1.0 - market_price))
            else:
                realized_pnl = 0.0

            records.append({
                "game_id": game_id,
                "home_team": play.get("home_team", ""),
                "away_team": play.get("away_team", ""),
                "margin": margin,
                "seconds_remaining": secs,
                "home_win": home_win,
                "model_prob": model_prob,
                "model_baseline": preds.get("baseline"),
                "model_enhanced": preds.get("enhanced"),
                "model_full": preds.get("full"),
                "model_momentum": preds.get("momentum"),
                "market_price": market_price,
                "edge": edge,
                "abs_edge": abs(edge),
                "trade": trade,
                "trade_direction": trade_direction if trade else None,
                "realized_pnl": realized_pnl if trade else 0.0,
                "run_team": play.get("run_team", 0),
                "run_length": play.get("run_length", 0),
                "run_points": play.get("run_points", 0),
                "momentum": play.get("momentum", 0.0),
                "scoring_burst": play.get("scoring_burst", 0),
            })

        if (gi + 1) % 500 == 0:
            logger.info(f"  Processed {gi + 1}/{len(game_ids)} games...")

    results = pd.DataFrame(records)
    logger.info(f"Backtest complete: {len(results):,} snapshots from {len(game_ids)} games")
    return results


def analyze_backtest(results: pd.DataFrame, bankroll: float = None):
    """Print backtest analysis."""
    trades = results[results["trade"]].copy()

    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST RESULTS")
    logger.info(f"{'='*60}")

    logger.info(f"Total snapshots: {len(results):,}")
    logger.info(f"Trades triggered: {len(trades):,} ({len(trades)/max(1,len(results))*100:.1f}%)")

    if len(trades) == 0:
        logger.info("No trades triggered.")
        return trades

    winning = trades[trades["realized_pnl"] > 0]
    logger.info(f"Winning trades: {len(winning):,} ({len(winning)/len(trades)*100:.1f}%)")

    logger.info(f"\nPnL Statistics (per $1 contract):")
    logger.info(f"  Mean PnL per trade: ${trades['realized_pnl'].mean():.4f}")
    logger.info(f"  Median PnL per trade: ${trades['realized_pnl'].median():.4f}")
    logger.info(f"  Std PnL: ${trades['realized_pnl'].std():.4f}")
    logger.info(f"  Total PnL: ${trades['realized_pnl'].sum():.2f}")

    if trades['realized_pnl'].std() > 0:
        sharpe = trades['realized_pnl'].mean() / trades['realized_pnl'].std()
        logger.info(f"  PnL ratio (mean/std): {sharpe:.3f}")

    logger.info(f"\nEdge Statistics:")
    logger.info(f"  Mean edge when trading: {trades['edge'].mean()*100:.2f} cents")
    logger.info(f"  Mean |edge| when trading: {trades['abs_edge'].mean()*100:.2f} cents")

    logger.info(f"\nBy Game Phase:")
    phases = [
        (0, 300, "Final 5 min"),
        (300, 720, "4th quarter"),
        (720, 1440, "3rd quarter"),
        (1440, 2160, "2nd quarter"),
        (2160, 2880, "1st quarter"),
    ]
    for lo, hi, label in phases:
        phase_trades = trades[
            (trades["seconds_remaining"] >= lo) & (trades["seconds_remaining"] < hi)
        ]
        if len(phase_trades) > 0:
            wr = len(phase_trades[phase_trades["realized_pnl"] > 0]) / len(phase_trades)
            logger.info(
                f"  {label:15s}: {len(phase_trades):5d} trades, "
                f"win rate {wr:.1%}, "
                f"avg PnL ${phase_trades['realized_pnl'].mean():.4f}"
            )

    # Model tier comparison
    if "model_full" in trades.columns and trades["model_full"].notna().any():
        logger.info(f"\nModel Tier Comparison (avg edge in trades):")
        for tier in ["model_baseline", "model_enhanced", "model_full"]:
            if tier in trades.columns and trades[tier].notna().any():
                tier_edge = trades[tier] - trades["market_price"]
                logger.info(f"  {tier:20s}: avg edge = {tier_edge.mean()*100:.2f} cents")

    # Kelly simulation
    if bankroll is not None and bankroll > 0:
        logger.info(f"\n{'='*60}")
        logger.info("KELLY CRITERION BACKTEST")
        logger.info(f"{'='*60}")
        logger.info(f"Starting bankroll: ${bankroll:.0f}")

        running = bankroll
        peak = bankroll
        max_drawdown = 0.0
        bankroll_curve = []

        for _, trade in trades.iterrows():
            f = kelly_fraction(trade["edge"], trade["market_price"])
            # Cap bet at 5% of bankroll and $500 max per trade for realism
            bet = min(f * running, running * 0.05, 500.0)
            pnl = trade["realized_pnl"] * bet
            running += pnl
            running = max(running, 1.0)  # floor at $1 to prevent collapse
            if np.isnan(running) or np.isinf(running):
                running = max(bankroll_curve[-1] if bankroll_curve else bankroll, 1.0)
            peak = max(peak, running)
            dd = (peak - running) / peak
            max_drawdown = max(max_drawdown, dd)
            bankroll_curve.append(running)

        trades["bankroll"] = bankroll_curve

        logger.info(f"Final bankroll: ${running:.2f}")
        logger.info(f"Total return: {(running / bankroll - 1):.1%}")
        logger.info(f"Max drawdown: {max_drawdown:.1%}")

    return trades


def visualize_backtest(results: pd.DataFrame, trades: pd.DataFrame, output_dir: Path):
    """Generate backtest visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Cumulative PnL curve
    ax = axes[0, 0]
    cum_pnl = trades["realized_pnl"].cumsum()
    ax.plot(cum_pnl.values, linewidth=0.8)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Cumulative PnL (Flat $1 Sizing)")

    # 2. Edge vs realized PnL scatter
    ax = axes[0, 1]
    ax.scatter(trades["edge"] * 100, trades["realized_pnl"], alpha=0.05, s=3)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Edge (cents)")
    ax.set_ylabel("Realized PnL ($)")
    ax.set_title("Edge vs Realized PnL")

    # 3. Win rate by edge bucket
    ax = axes[1, 0]
    trades_copy = trades.copy()
    trades_copy["edge_bucket"] = pd.cut(trades_copy["abs_edge"] * 100, bins=[0, 3, 5, 8, 10, 15, 20, 50])
    wr_by_edge = trades_copy.groupby("edge_bucket", observed=True)["realized_pnl"].apply(
        lambda x: (x > 0).mean()
    )
    wr_by_edge.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("|Edge| Bucket (cents)")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Edge Size")
    ax.tick_params(axis="x", rotation=45)

    # 4. PnL by game phase
    ax = axes[1, 1]
    trades_copy["phase"] = pd.cut(
        trades_copy["seconds_remaining"],
        bins=[0, 300, 720, 1440, 2160, 2880],
        labels=["Final 5m", "Q4", "Q3", "Q2", "Q1"],
    )
    phase_pnl = trades_copy.groupby("phase", observed=True)["realized_pnl"].mean()
    phase_pnl.plot(kind="bar", ax=ax, color="darkgreen", edgecolor="black")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Game Phase")
    ax.set_ylabel("Avg PnL per Trade ($)")
    ax.set_title("Average PnL by Game Phase")
    ax.tick_params(axis="x", rotation=0)

    plt.suptitle("Win Probability Model Backtest Results", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "backtest_results.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved backtest charts -> {output_dir / 'backtest_results.png'}")
    plt.close()

    # Bankroll curve if Kelly was run
    if "bankroll" in trades.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(trades["bankroll"].values, linewidth=0.8)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Bankroll ($)")
        ax.set_title("Kelly Criterion Bankroll Curve")
        ax.axhline(trades["bankroll"].iloc[0], color="red", linestyle="--", alpha=0.3, label="Starting")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "backtest_kelly_curve.png", dpi=150)
        logger.info(f"Saved Kelly curve -> {output_dir / 'backtest_kelly_curve.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Backtest win probability model")
    parser.add_argument("--season", type=int, default=None, help="Single season to backtest")
    parser.add_argument("--threshold", type=float, default=5.0, help="Edge threshold in cents")
    parser.add_argument("--noise", type=float, default=0.03, help="Market noise std dev")
    parser.add_argument("--lag", type=int, default=30, help="Market lag in seconds")
    parser.add_argument("--sample-interval", type=int, default=5, help="Sample every Nth play")
    parser.add_argument("--kelly", action="store_true", help="Run Kelly criterion simulation")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll")
    parser.add_argument("--visualize", action="store_true", help="Generate charts")
    args = parser.parse_args()

    df = load_game_states(season=args.season)
    models = load_models()
    table = load_empirical_table()

    tiers = [k for k in ["baseline", "enhanced", "full"] if models.get(k)]
    logger.info(f"Model tiers available: {', '.join(tiers)}")

    results = run_backtest(
        df, models, table,
        threshold_cents=args.threshold,
        noise_std=args.noise,
        market_lag_sec=args.lag,
        sample_interval=args.sample_interval,
    )

    bankroll = args.bankroll if args.kelly else None
    trades = analyze_backtest(results, bankroll=bankroll)

    analysis_dir = PROJECT_ROOT / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    results.to_csv(analysis_dir / "backtest_results.csv", index=False)
    logger.info(f"Saved backtest results -> {analysis_dir / 'backtest_results.csv'}")

    if args.visualize and trades is not None and len(trades) > 0:
        visualize_backtest(results, trades, analysis_dir)

    logger.success("Backtest complete!")


if __name__ == "__main__":
    main()
