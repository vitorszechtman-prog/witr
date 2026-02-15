"""
01_fetch_pbp_data.py
====================
Process NBA play-by-play CSV data (from shufinskiy/nba_data on GitHub) and
extract game-state snapshots (score margin, seconds remaining) with known
outcomes for building a win probability model.

Data source: https://github.com/shufinskiy/nba_data
  - Pre-downloaded CSV files in data/raw/nbastats_{season}.csv
  - If not present, downloads automatically from GitHub.

Usage:
    python 01_fetch_pbp_data.py --seasons 2022 2023 2024
    python 01_fetch_pbp_data.py --seasons 2024 --sample 50  # quick test

Output:
    data/game_states_{season}.parquet  — one row per play with game state
    data/game_outcomes_{season}.parquet — game-level outcomes
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from utils import DATA_DIR, seconds_remaining_in_game

RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)

DOWNLOAD_URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/nbastats_{season}.tar.xz"


def ensure_csv(season_year: int) -> Path:
    """Download and extract the season CSV if not already present."""
    csv_path = RAW_DIR / f"nbastats_{season_year}.csv"
    if csv_path.exists():
        logger.info(f"  Found existing CSV: {csv_path}")
        return csv_path

    tarball = RAW_DIR / f"nbastats_{season_year}.tar.xz"
    url = DOWNLOAD_URL.format(season=season_year)

    if not tarball.exists():
        logger.info(f"  Downloading {url} ...")
        subprocess.run(
            ["curl", "-sL", url, "-o", str(tarball)],
            check=True,
        )

    logger.info(f"  Extracting {tarball} ...")
    subprocess.run(
        ["tar", "-xJf", str(tarball), "-C", str(RAW_DIR)],
        check=True,
    )
    return csv_path


def identify_teams_per_game(df: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """
    For each GAME_ID, identify (home_team, away_team) abbreviations
    by looking at which team's plays appear in HOMEDESCRIPTION vs VISITORDESCRIPTION.
    """
    teams = {}

    home_plays = df[df["HOMEDESCRIPTION"].notna() & df["PLAYER1_TEAM_ABBREVIATION"].notna()]
    away_plays = df[df["VISITORDESCRIPTION"].notna() & df["PLAYER1_TEAM_ABBREVIATION"].notna()]

    home_teams = home_plays.groupby("GAME_ID")["PLAYER1_TEAM_ABBREVIATION"].agg(lambda x: x.mode().iloc[0])
    away_teams = away_plays.groupby("GAME_ID")["PLAYER1_TEAM_ABBREVIATION"].agg(lambda x: x.mode().iloc[0])

    for game_id in home_teams.index:
        if game_id in away_teams.index:
            teams[game_id] = (str(home_teams[game_id]), str(away_teams[game_id]))

    return teams


def process_season(season_year: int, sample: int | None = None):
    """
    Process an entire season's play-by-play CSV into game states and outcomes.
    """
    csv_path = ensure_csv(season_year)

    logger.info(f"  Loading {csv_path} ...")
    df = pd.read_csv(
        csv_path,
        usecols=[
            "GAME_ID", "PERIOD", "PCTIMESTRING", "SCORE", "SCOREMARGIN",
            "HOMEDESCRIPTION", "VISITORDESCRIPTION", "PLAYER1_TEAM_ABBREVIATION",
        ],
        dtype={"PCTIMESTRING": str, "SCORE": str, "SCOREMARGIN": str},
    )
    logger.info(f"  Loaded {len(df):,} play-by-play events across {df['GAME_ID'].nunique()} games")

    # Identify home/away teams per game
    logger.info("  Identifying home/away teams per game...")
    game_teams = identify_teams_per_game(df)
    logger.info(f"  Identified teams for {len(game_teams)} games")

    game_ids = sorted(game_teams.keys())

    if sample:
        rng = np.random.default_rng(42)
        game_ids = sorted(rng.choice(game_ids, size=min(sample, len(game_ids)), replace=False))
        logger.info(f"  Sampled {len(game_ids)} games for testing")

    # Process each game
    all_states = []
    all_outcomes = []
    errors = 0

    for idx, game_id in enumerate(game_ids):
        home_team, away_team = game_teams[game_id]
        game_df = df[df["GAME_ID"] == game_id]

        try:
            states, outcome = extract_game_states(game_df, game_id, home_team, away_team)
            all_states.extend(states)
            all_outcomes.append(outcome)
        except Exception as e:
            errors += 1
            logger.warning(f"  Error on game {game_id}: {e}")
            if errors > 50:
                logger.error("  Too many errors, stopping early")
                break
            continue

        if (idx + 1) % 200 == 0:
            logger.info(f"  Processed {idx + 1}/{len(game_ids)} games ({len(all_states):,} states)...")

    # Save outputs
    states_df = pd.DataFrame(all_states)
    outcomes_df = pd.DataFrame(all_outcomes)

    states_path = DATA_DIR / f"game_states_{season_year}.parquet"
    outcomes_path = DATA_DIR / f"game_outcomes_{season_year}.parquet"

    states_df.to_parquet(states_path, index=False)
    outcomes_df.to_parquet(outcomes_path, index=False)

    logger.info(f"  Saved {len(states_df):,} game-state snapshots → {states_path}")
    logger.info(f"  Saved {len(outcomes_df):,} game outcomes → {outcomes_path}")
    logger.info(f"  Errors: {errors}")

    # Sanity checks
    home_win_pct = outcomes_df["home_win"].mean()
    avg_margin = outcomes_df["home_margin_final"].mean()
    logger.info(f"  Home win%: {home_win_pct:.1%}, Avg home margin: {avg_margin:+.1f}")

    return states_df


def extract_game_states(
    game_df: pd.DataFrame,
    game_id: int,
    home_team: str,
    away_team: str,
) -> tuple[list[dict], dict]:
    """
    From a single game's play-by-play DataFrame, extract game-state snapshots.

    Score format in the CSV: "VISITOR_PTS - HOME_PTS"
    SCOREMARGIN: HOME_PTS - VISITOR_PTS (positive = home ahead, "TIE" = tied)

    Returns:
        states: list of game-state snapshot dicts
        outcome: game-level outcome dict
    """
    scored = game_df[game_df["SCORE"].notna()].copy()
    if len(scored) == 0:
        raise ValueError("No scored plays found")

    states = []
    for _, play in scored.iterrows():
        score_str = str(play["SCORE"]).strip()
        try:
            parts = score_str.split(" - ")
            away_score = int(parts[0].strip())
            home_score = int(parts[1].strip())
        except (ValueError, IndexError):
            continue

        period = int(play["PERIOD"]) if pd.notna(play["PERIOD"]) else 0
        clock = str(play["PCTIMESTRING"]) if pd.notna(play["PCTIMESTRING"]) else "0:00"
        secs_remaining = seconds_remaining_in_game(period, clock)

        states.append({
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "period": period,
            "clock": clock,
            "seconds_remaining": secs_remaining,
            "home_score": home_score,
            "away_score": away_score,
            "home_margin": home_score - away_score,
        })

    # Determine winner from final score
    last = states[-1]
    home_win = int(last["home_score"] > last["away_score"])

    # Tag all states with outcome
    for s in states:
        s["home_win"] = home_win

    outcome = {
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "home_pts": last["home_score"],
        "away_pts": last["away_score"],
        "home_margin_final": last["home_margin"],
        "home_win": home_win,
    }

    return states, outcome


def main():
    parser = argparse.ArgumentParser(
        description="Process NBA play-by-play data for win probability modeling"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
        help="Season years to process (e.g., 2024 = 2024-25 season)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N games per season (for testing)",
    )
    args = parser.parse_args()

    for season in args.seasons:
        logger.info(f"{'='*60}")
        logger.info(f"Processing season {season}-{season+1}")
        logger.info(f"{'='*60}")
        process_season(season, sample=args.sample)

    logger.success("Done! Data saved to data/ directory.")


if __name__ == "__main__":
    main()
