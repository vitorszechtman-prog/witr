"""
01_fetch_pbp_data.py
====================
Pull historical NBA play-by-play data and extract game-state snapshots
(score margin, seconds remaining) with known outcomes for building a
win probability model.

Usage:
    python scripts/01_fetch_pbp_data.py --seasons 2022 2023 2024
    python scripts/01_fetch_pbp_data.py --seasons 2024 --sample 50  # quick test

Output:
    data/game_states_{season}.parquet  — one row per play with game state
    data/game_outcomes_{season}.parquet — game-level outcomes
"""

import argparse
import time
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# nba_api imports
from nba_api.stats.endpoints import (
    LeagueGameLog,
    PlayByPlayV2,
)
from nba_api.stats.static import teams

sys.path.append(str(Path(__file__).parent))
from utils import DATA_DIR, seconds_remaining_in_game


def get_season_games(season_year: int) -> pd.DataFrame:
    """
    Fetch all regular season + playoff games for a given season.
    season_year=2024 means the 2024-25 season.
    """
    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    logger.info(f"Fetching game log for season {season_str}...")

    # Home team game log
    gl = LeagueGameLog(
        season=season_str,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
    )
    time.sleep(0.6)  # respect rate limits

    df = gl.get_data_frames()[0]

    # Each game appears twice (home + away). Deduplicate by GAME_ID.
    # We need to identify home vs away and final scores.
    games = []
    for game_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            continue

        # The matchup column contains "vs." for home team, "@" for away
        home_row = group[group["MATCHUP"].str.contains("vs.", na=False)]
        away_row = group[group["MATCHUP"].str.contains("@", na=False)]

        if len(home_row) != 1 or len(away_row) != 1:
            continue

        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]

        games.append({
            "game_id": game_id,
            "date": home_row["GAME_DATE"],
            "home_team_id": home_row["TEAM_ID"],
            "home_team": home_row["TEAM_ABBREVIATION"],
            "away_team_id": away_row["TEAM_ID"],
            "away_team": away_row["TEAM_ABBREVIATION"],
            "home_pts": home_row["PTS"],
            "away_pts": away_row["PTS"],
            "home_win": int(home_row["WL"] == "W"),
        })

    games_df = pd.DataFrame(games)
    logger.info(f"  Found {len(games_df)} games for {season_str}")
    return games_df


def get_game_pbp(game_id: str) -> pd.DataFrame:
    """Fetch play-by-play for a single game."""
    pbp = PlayByPlayV2(game_id=game_id)
    time.sleep(0.6)  # rate limit
    df = pbp.get_data_frames()[0]
    return df


def extract_game_states(pbp_df: pd.DataFrame, home_team_id: int, home_win: int) -> list[dict]:
    """
    From a play-by-play DataFrame, extract game-state snapshots.
    
    Each snapshot contains:
    - seconds_remaining (in regulation)
    - home_margin (home score - away score)
    - period
    - home_win (outcome label)
    """
    states = []

    for _, play in pbp_df.iterrows():
        # Skip rows without score info
        score_str = play.get("SCORE")
        if pd.isna(score_str) or not isinstance(score_str, str):
            continue

        # Score format: "AWAY_PTS - HOME_PTS" (visitor listed first)
        try:
            parts = score_str.strip().split(" - ")
            away_score = int(parts[0].strip())
            home_score = int(parts[1].strip())
        except (ValueError, IndexError):
            continue

        period = play.get("PERIOD", 0)
        clock = play.get("PCTIMESTRING", "0:00")

        secs_remaining = seconds_remaining_in_game(period, clock)

        states.append({
            "period": period,
            "clock": clock,
            "seconds_remaining": secs_remaining,
            "home_score": home_score,
            "away_score": away_score,
            "home_margin": home_score - away_score,
            "home_win": home_win,
        })

    return states


def fetch_season(season_year: int, sample: int | None = None):
    """
    Fetch and process an entire season's play-by-play data.
    """
    games_df = get_season_games(season_year)

    if sample:
        games_df = games_df.sample(n=min(sample, len(games_df)), random_state=42)
        logger.info(f"  Sampled {len(games_df)} games for testing")

    # Save game outcomes
    outcomes_path = DATA_DIR / f"game_outcomes_{season_year}.parquet"
    games_df.to_parquet(outcomes_path, index=False)
    logger.info(f"  Saved game outcomes → {outcomes_path}")

    # Fetch play-by-play for each game
    all_states = []
    errors = 0

    for i, game in games_df.iterrows():
        game_id = game["game_id"]
        try:
            pbp_df = get_game_pbp(game_id)
            states = extract_game_states(
                pbp_df,
                home_team_id=game["home_team_id"],
                home_win=game["home_win"],
            )
            for s in states:
                s["game_id"] = game_id
                s["home_team"] = game["home_team"]
                s["away_team"] = game["away_team"]

            all_states.extend(states)

            if (len(all_states) // 10000) > ((len(all_states) - len(states)) // 10000):
                logger.info(f"  Processed {len(all_states):,} game states so far...")

        except Exception as e:
            errors += 1
            logger.warning(f"  Error on game {game_id}: {e}")
            if errors > 20:
                logger.error("  Too many errors, stopping early")
                break
            continue

        # Progress log every 50 games
        idx = games_df.index.get_loc(i) if i in games_df.index else 0
        if isinstance(idx, int) and idx % 50 == 0 and idx > 0:
            logger.info(f"  Fetched {idx}/{len(games_df)} games...")

    # Save game states
    states_df = pd.DataFrame(all_states)
    states_path = DATA_DIR / f"game_states_{season_year}.parquet"
    states_df.to_parquet(states_path, index=False)
    logger.info(
        f"  Saved {len(states_df):,} game-state snapshots → {states_path}"
    )
    logger.info(f"  Errors: {errors}")

    return states_df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA play-by-play data for win probability modeling"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2023, 2024],
        help="Season years to fetch (e.g., 2024 = 2024-25 season)",
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
        fetch_season(season, sample=args.sample)

    logger.success("Done! Data saved to data/ directory.")


if __name__ == "__main__":
    main()
