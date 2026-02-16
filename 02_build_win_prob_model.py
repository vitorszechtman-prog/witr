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
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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


TEAM_FEATURE_COLS = [
    "home_win_pct", "away_win_pct", "home_home_win_pct",
    "home_last10", "away_last10", "strength_diff",
]


def build_logistic_model(df: pd.DataFrame) -> dict:
    """
    Build two models:
        1. Baseline: P(home_win) = f(margin, time)
        2. Enhanced: P(home_win) = f(margin, time, team_strength, L10, home_court_adv)

    Returns dict with both models and feature metadata.
    """
    from sklearn.metrics import log_loss, brier_score_loss

    reg = df[df["seconds_remaining"] >= 0].copy()

    # Core features (same as before)
    reg["time_fraction"] = reg["seconds_remaining"] / 2880
    reg["margin_x_time"] = reg["home_margin"] * reg["time_fraction"]

    base_features = ["home_margin", "time_fraction", "margin_x_time"]

    # --- Baseline model ---
    logger.info("Building BASELINE logistic model (margin + time only)...")
    X_base = reg[base_features].values
    y = reg["home_win"].values

    baseline = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, C=1.0)),
    ])
    baseline.fit(X_base, y)

    y_pred_base = baseline.predict_proba(X_base)[:, 1]
    ll_base = log_loss(y, y_pred_base)
    bs_base = brier_score_loss(y, y_pred_base)
    logger.info(f"  Baseline — Log loss: {ll_base:.4f}, Brier score: {bs_base:.4f}")

    # --- Enhanced model ---
    has_team_features = all(c in reg.columns for c in TEAM_FEATURE_COLS)
    enhanced = None

    if has_team_features:
        logger.info("Building ENHANCED logistic model (+ team strength, L10, home court)...")

        # Filter to games where teams have played >= 10 games (features are meaningful)
        has_data = reg.copy()

        # Enhanced features
        enhanced_features = base_features + TEAM_FEATURE_COLS + [
            "strength_x_time",  # team quality matters more early
            "margin_x_strength",  # margin means more when teams are mismatched
        ]
        has_data["strength_x_time"] = has_data["strength_diff"] * has_data["time_fraction"]
        has_data["margin_x_strength"] = has_data["home_margin"] * has_data["strength_diff"]

        X_enh = has_data[enhanced_features].values
        y_enh = has_data["home_win"].values

        enhanced = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, C=1.0)),
        ])
        enhanced.fit(X_enh, y_enh)

        y_pred_enh = enhanced.predict_proba(X_enh)[:, 1]
        ll_enh = log_loss(y_enh, y_pred_enh)
        bs_enh = brier_score_loss(y_enh, y_pred_enh)
        logger.info(f"  Enhanced — Log loss: {ll_enh:.4f}, Brier score: {bs_enh:.4f}")
        logger.info(f"  Improvement — Log loss: {ll_base - ll_enh:+.4f}, Brier: {bs_base - bs_enh:+.4f}")
    else:
        logger.warning("  Team features not found in data. Run 01_fetch_pbp_data.py to regenerate.")

    return {
        "baseline": baseline,
        "enhanced": enhanced,
        "base_features": base_features,
        "enhanced_features": enhanced_features if enhanced else None,
    }


def predict_from_model(
    models: dict,
    margin: float,
    seconds_remaining: float,
    home_win_pct: float = None,
    away_win_pct: float = None,
    home_home_win_pct: float = None,
    home_last10: float = None,
    away_last10: float = None,
) -> float:
    """
    Get win probability. Uses enhanced model if team features are provided,
    otherwise falls back to baseline.
    """
    time_frac = seconds_remaining / 2880

    use_enhanced = (
        models.get("enhanced") is not None
        and home_win_pct is not None
        and away_win_pct is not None
    )

    if use_enhanced:
        strength_diff = home_win_pct - away_win_pct
        if home_home_win_pct is None:
            home_home_win_pct = home_win_pct
        if home_last10 is None:
            home_last10 = home_win_pct
        if away_last10 is None:
            away_last10 = away_win_pct

        X = np.array([[
            margin, time_frac, margin * time_frac,
            home_win_pct, away_win_pct, home_home_win_pct,
            home_last10, away_last10, strength_diff,
            strength_diff * time_frac,
            margin * strength_diff,
        ]])
        return float(models["enhanced"].predict_proba(X)[0, 1])
    else:
        X = np.array([[margin, time_frac, margin * time_frac]])
        return float(models["baseline"].predict_proba(X)[0, 1])


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
        models = build_logistic_model(df)
        model_path = DATA_DIR / "win_prob_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(models, f)
        logger.info(f"Saved models → {model_path}")

        # Print sample predictions comparing baseline vs enhanced
        logger.info("\nSample predictions:")
        logger.info(f"  {'Scenario':40s} {'Baseline':>10s} {'Enhanced':>10s}")
        logger.info(f"  {'—'*40} {'—'*10} {'—'*10}")

        test_cases = [
            # (margin, secs, label, home_wpct, away_wpct, home_home_wpct, h_l10, a_l10)
            (0, 2880, "Tip-off, tied (even teams)", 0.50, 0.50, 0.55, 0.50, 0.50),
            (0, 2880, "Tip-off, 70% team hosting 30%", 0.70, 0.30, 0.80, 0.70, 0.30),
            (0, 2880, "Tip-off, 30% team hosting 70%", 0.30, 0.70, 0.35, 0.30, 0.70),
            (5, 1440, "Up 5, halftime (even)", 0.50, 0.50, 0.55, 0.50, 0.50),
            (5, 1440, "Up 5, halftime (fav hosting)", 0.65, 0.40, 0.70, 0.70, 0.40),
            (-5, 1440, "Down 5, halftime (fav hosting)", 0.65, 0.40, 0.70, 0.70, 0.40),
            (2, 480, "Up 2, 8 min left (even)", 0.50, 0.50, 0.55, 0.50, 0.50),
            (2, 480, "Up 2, 8 min (hot team)", 0.55, 0.45, 0.60, 0.90, 0.30),
            (-2, 480, "Down 2, 8 min (hot team)", 0.55, 0.45, 0.60, 0.90, 0.30),
            (10, 300, "Up 10, 5 min left", 0.50, 0.50, 0.55, 0.50, 0.50),
            (3, 60, "Up 3, 1 min left", 0.50, 0.50, 0.55, 0.50, 0.50),
            (-3, 60, "Down 3, 1 min left", 0.50, 0.50, 0.55, 0.50, 0.50),
        ]
        for margin, secs, label, h_wp, a_wp, h_hwp, h_l10, a_l10 in test_cases:
            base = predict_from_model(models, margin, secs)
            enh = predict_from_model(
                models, margin, secs,
                home_win_pct=h_wp, away_win_pct=a_wp,
                home_home_win_pct=h_hwp, home_last10=h_l10, away_last10=a_l10,
            )
            logger.info(f"  {label:40s} {base:9.1%} {enh:9.1%}")

    logger.success("Model building complete!")


if __name__ == "__main__":
    main()
