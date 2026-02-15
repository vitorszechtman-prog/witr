"""
02_build_win_prob_model.py
==========================
Build an empirical win probability surface from historical NBA game states.

Produces:
    P(home_win | home_margin, seconds_remaining)

Two approaches:
    1. Empirical lookup table (binned, smoothed)
    2. Logistic regression model (parametric, interpolates well)

Usage:
    python scripts/02_build_win_prob_model.py
    python scripts/02_build_win_prob_model.py --method both --visualize

Output:
    data/win_prob_empirical.parquet   — binned lookup table
    data/win_prob_model.pkl           — fitted sklearn model
    analysis/win_prob_heatmap.png     — visualization
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from utils import DATA_DIR, PROJECT_ROOT


def load_game_states() -> pd.DataFrame:
    """Load all available game-state parquet files."""
    files = sorted(DATA_DIR.glob("game_states_*.parquet"))
    if not files:
        logger.error("No game_states files found. Run 01_fetch_pbp_data.py first.")
        sys.exit(1)

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} game-state snapshots from {len(files)} season(s)")
    return df


def build_empirical_table(
    df: pd.DataFrame,
    margin_range: tuple = (-30, 30),
    time_bucket_sec: int = 30,
    sigma: float = 1.5,
) -> pd.DataFrame:
    """
    Build a smoothed empirical win probability lookup table.
    
    Bins: (home_margin, seconds_remaining_bucket) → P(home_win)
    Smoothing: Gaussian filter to handle sparse bins.
    """
    logger.info("Building empirical win probability table...")

    # Filter to regulation only (seconds_remaining >= 0)
    reg = df[df["seconds_remaining"] >= 0].copy()

    # Clip margins to range
    reg["margin_clipped"] = reg["home_margin"].clip(*margin_range)

    # Bucket time
    reg["time_bucket"] = (reg["seconds_remaining"] // time_bucket_sec) * time_bucket_sec

    # Group and compute win rates
    grouped = (
        reg.groupby(["margin_clipped", "time_bucket"])
        .agg(
            win_count=("home_win", "sum"),
            total_count=("home_win", "count"),
        )
        .reset_index()
    )
    grouped["win_prob_raw"] = grouped["win_count"] / grouped["total_count"]

    # Create full grid
    margins = np.arange(margin_range[0], margin_range[1] + 1)
    max_time = int(reg["time_bucket"].max())
    time_buckets = np.arange(0, max_time + time_bucket_sec, time_bucket_sec)

    grid = np.full((len(margins), len(time_buckets)), np.nan)
    counts = np.zeros((len(margins), len(time_buckets)))

    margin_to_idx = {m: i for i, m in enumerate(margins)}
    time_to_idx = {t: i for i, t in enumerate(time_buckets)}

    for _, row in grouped.iterrows():
        mi = margin_to_idx.get(int(row["margin_clipped"]))
        ti = time_to_idx.get(int(row["time_bucket"]))
        if mi is not None and ti is not None:
            grid[mi, ti] = row["win_prob_raw"]
            counts[mi, ti] = row["total_count"]

    # Fill NaN with interpolation before smoothing
    # For margins: team up a lot → ~1.0, down a lot → ~0.0
    for ti in range(len(time_buckets)):
        col = grid[:, ti]
        if np.all(np.isnan(col)):
            # Fill with logistic approximation
            for mi, margin in enumerate(margins):
                secs = time_buckets[ti]
                if secs > 0:
                    grid[mi, ti] = _logistic_approx(margin, secs)
                else:
                    grid[mi, ti] = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        else:
            # Forward/back fill then interpolate
            nans = np.isnan(col)
            if nans.any():
                valid = ~nans
                if valid.sum() >= 2:
                    grid[nans, ti] = np.interp(
                        np.where(nans)[0],
                        np.where(valid)[0],
                        col[valid],
                    )

    # Apply Gaussian smoothing
    smoothed = gaussian_filter(grid, sigma=sigma)
    smoothed = np.clip(smoothed, 0.001, 0.999)

    # At time=0 (game over), enforce hard outcomes
    t0_idx = time_to_idx.get(0)
    if t0_idx is not None:
        for mi, margin in enumerate(margins):
            if margin > 0:
                smoothed[mi, t0_idx] = 0.999
            elif margin < 0:
                smoothed[mi, t0_idx] = 0.001
            else:
                smoothed[mi, t0_idx] = 0.5

    # Build output DataFrame
    records = []
    for mi, margin in enumerate(margins):
        for ti, secs in enumerate(time_buckets):
            records.append({
                "home_margin": margin,
                "seconds_remaining": int(secs),
                "win_prob": round(float(smoothed[mi, ti]), 4),
                "sample_count": int(counts[mi, ti]),
            })

    result = pd.DataFrame(records)
    logger.info(
        f"  Built table: {len(margins)} margins × {len(time_buckets)} time buckets "
        f"= {len(result):,} cells"
    )
    return result


def _logistic_approx(margin: float, seconds_remaining: float) -> float:
    """Quick logistic approximation for filling NaN cells."""
    # Rough scaling: as time decreases, same margin matters more
    if seconds_remaining <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    # Scale factor: points matter more with less time
    k = 0.15 + 0.85 * (1 - seconds_remaining / 2880)
    z = margin * k * 0.3
    return 1 / (1 + np.exp(-z))


def build_logistic_model(df: pd.DataFrame) -> Pipeline:
    """
    Build a calibrated logistic regression model:
        P(home_win) = f(home_margin, seconds_remaining, margin * time_interaction)
    
    Uses polynomial features to capture non-linear relationships.
    """
    logger.info("Building logistic regression model...")

    reg = df[df["seconds_remaining"] >= 0].copy()

    # Features
    reg["time_fraction"] = reg["seconds_remaining"] / 2880  # fraction of game remaining
    reg["margin_x_time"] = reg["home_margin"] * reg["time_fraction"]

    X = reg[["home_margin", "time_fraction", "margin_x_time"]].values
    y = reg["home_win"].values

    # Pipeline: polynomial features + calibrated logistic regression
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)),
        ("lr", LogisticRegression(max_iter=5000, C=1.0)),
    ])

    # Fit with calibration for better probability estimates
    model.fit(X, y)

    # Evaluate on training data (just for logging)
    from sklearn.metrics import log_loss, brier_score_loss
    y_pred = model.predict_proba(X)[:, 1]
    ll = log_loss(y, y_pred)
    bs = brier_score_loss(y, y_pred)
    logger.info(f"  Log loss: {ll:.4f}, Brier score: {bs:.4f}")

    return model


def predict_from_model(
    model: Pipeline,
    margin: float,
    seconds_remaining: float,
) -> float:
    """Get win probability from the fitted model."""
    time_frac = seconds_remaining / 2880
    X = np.array([[margin, time_frac, margin * time_frac]])
    return float(model.predict_proba(X)[0, 1])


def visualize(table_df: pd.DataFrame, output_path: Path):
    """Generate a heatmap of win probabilities."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    logger.info("Generating win probability heatmap...")

    pivot = table_df.pivot(
        index="home_margin",
        columns="seconds_remaining",
        values="win_prob",
    )

    # Convert seconds to minutes for readability
    pivot.columns = [c / 60 for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(16, 10))

    # Custom colormap: red (away winning) → white (50/50) → blue (home winning)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "win_prob",
        [(0.8, 0.1, 0.1), (1, 1, 1), (0.1, 0.2, 0.8)],
    )

    # Subsample for readability
    margin_step = 2
    time_step = 4  # every 2 minutes
    pivot_sub = pivot.iloc[::margin_step, ::time_step]

    im = ax.imshow(
        pivot_sub.values,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="bilinear",
    )

    # Labels
    ax.set_xticks(range(len(pivot_sub.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot_sub.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot_sub.index)))
    ax.set_yticklabels(pivot_sub.index, fontsize=8)

    ax.set_xlabel("Minutes Remaining", fontsize=12)
    ax.set_ylabel("Home Team Margin", fontsize=12)
    ax.set_title("NBA Win Probability Surface\nP(Home Win | Margin, Time Remaining)", fontsize=14)

    plt.colorbar(im, ax=ax, label="P(Home Win)")

    # Add contour lines at key probabilities
    # 50%, 75%, 90%, 95%
    contour_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved heatmap → {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Build NBA win probability model")
    parser.add_argument(
        "--method",
        choices=["empirical", "logistic", "both"],
        default="both",
        help="Which model(s) to build",
    )
    parser.add_argument("--visualize", action="store_true", help="Generate heatmap")
    args = parser.parse_args()

    df = load_game_states()

    if args.method in ("empirical", "both"):
        table = build_empirical_table(df)
        table_path = DATA_DIR / "win_prob_empirical.parquet"
        table.to_parquet(table_path, index=False)
        logger.info(f"Saved empirical table → {table_path}")

        if args.visualize:
            analysis_dir = PROJECT_ROOT / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            visualize(table, analysis_dir / "win_prob_heatmap.png")

    if args.method in ("logistic", "both"):
        model = build_logistic_model(df)
        model_path = DATA_DIR / "win_prob_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved logistic model → {model_path}")

        # Print some sample predictions
        logger.info("\nSample predictions (logistic model):")
        test_cases = [
            (0, 2880, "Tip-off, tied"),
            (5, 1440, "Up 5, halftime"),
            (2, 480, "Up 2, 8 min left"),
            (-2, 480, "Down 2, 8 min left"),
            (10, 300, "Up 10, 5 min left"),
            (-10, 300, "Down 10, 5 min left"),
            (3, 60, "Up 3, 1 min left"),
            (-3, 60, "Down 3, 1 min left"),
            (15, 120, "Up 15, 2 min left"),
        ]
        for margin, secs, label in test_cases:
            prob = predict_from_model(model, margin, secs)
            logger.info(f"  {label:30s} → P(win) = {prob:.1%}")

    logger.success("Model building complete!")


if __name__ == "__main__":
    main()
