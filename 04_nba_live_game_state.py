"""
04_nba_live_game_state.py
=========================
Track live NBA game state (score, clock, period) and compute real-time
win probabilities using the trained model.

Can run standalone or be imported by the main logger.

Usage:
    python 04_nba_live_game_state.py                  # log live game states
    python 04_nba_live_game_state.py --interval 15    # every 15 seconds
    python 04_nba_live_game_state.py --with-model     # include model predictions

Output:
    logs/nba_game_state_YYYYMMDD.jsonl
    data/nba_game_state.jsonl  (cumulative)
"""

import argparse
import pickle
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import requests
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from utils import DATA_DIR, LOGS_DIR, append_jsonl, utcnow_iso, seconds_remaining_in_game


class TeamStatsCache:
    """
    Cache current-season team stats for win probability model input.
    Fetches from NBA's stats endpoints and refreshes periodically.
    """

    STANDINGS_URL = "https://cdn.nba.com/static/json/liveData/standings/standings_00.json"

    def __init__(self, refresh_interval: int = 3600):
        self.refresh_interval = refresh_interval
        self._cache = {}
        self._last_refresh = 0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
        })

    def _refresh(self):
        """Fetch latest standings from NBA CDN."""
        now = time.time()
        if now - self._last_refresh < self.refresh_interval and self._cache:
            return

        try:
            resp = self.session.get(self.STANDINGS_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            standings = data.get("standings", {}).get("entries", [])

            for team in standings:
                stats = team.get("stats", {})
                tricode = team.get("team", {}).get("abbreviation", "")
                if not tricode:
                    continue

                wins = int(stats.get("wins", {}).get("value", 0))
                losses = int(stats.get("losses", {}).get("value", 0))
                gp = wins + losses

                # Try to get home record
                home_wins = int(stats.get("homeWins", {}).get("value", 0))
                home_losses = int(stats.get("homeLosses", {}).get("value", 0))
                home_gp = home_wins + home_losses

                # Try to get L10
                last10_wins = int(stats.get("last10Wins", {}).get("value", 0))
                last10_losses = int(stats.get("last10Losses", {}).get("value", 0))
                last10_gp = last10_wins + last10_losses

                # Point differential (net rating proxy)
                ppg = float(stats.get("pointsPerGame", {}).get("value", 0))
                opp_ppg = float(stats.get("oppPointsPerGame", {}).get("value", 0))

                self._cache[tricode] = {
                    "win_pct": wins / max(1, gp),
                    "home_win_pct": home_wins / max(1, home_gp),
                    "last10": last10_wins / max(1, last10_gp),
                    "net_rating": ppg - opp_ppg,
                    "pace": ppg + opp_ppg,  # total points proxy
                    "games_played": gp,
                }

            self._last_refresh = now
            logger.info(f"Refreshed team stats for {len(self._cache)} teams")

        except Exception as e:
            logger.warning(f"Failed to refresh team stats: {e}")
            # If we have stale data, keep using it
            if not self._cache:
                # Fallback: league-average defaults
                logger.warning("Using league-average defaults for all teams")

    def get(self, tricode: str) -> dict:
        """Get team stats for a tricode. Returns defaults if not found."""
        self._refresh()
        return self._cache.get(tricode, {
            "win_pct": 0.5,
            "home_win_pct": 0.55,
            "last10": 0.5,
            "net_rating": 0.0,
            "pace": 210.0,
            "games_played": 0,
        })


def load_model():
    """Load the fitted win probability model from disk."""
    model_path = DATA_DIR / "win_prob_model.pkl"
    if not model_path.exists():
        logger.warning("No model found. Run 02_build_win_prob_model.py first.")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_live_win_prob(
    models: dict,
    margin: int,
    seconds_remaining: int,
    home_stats: dict,
    away_stats: dict,
) -> dict:
    """
    Compute win probability using all available model tiers.
    Returns dict with baseline, enhanced, and full probabilities.
    """
    from utils import DATA_DIR  # ensure import
    sys.path.append(str(Path(__file__).parent))

    time_frac = seconds_remaining / 2880

    result = {}

    # Baseline
    if models.get("baseline"):
        X = np.array([[margin, time_frac, margin * time_frac]])
        result["wp_baseline"] = float(models["baseline"].predict_proba(X)[0, 1])

    # Enhanced
    if models.get("enhanced"):
        strength_diff = home_stats["win_pct"] - away_stats["win_pct"]
        X = np.array([[
            margin, time_frac, margin * time_frac,
            home_stats["win_pct"], away_stats["win_pct"], home_stats["home_win_pct"],
            home_stats["last10"], away_stats["last10"], strength_diff,
            strength_diff * time_frac,
            margin * strength_diff,
        ]])
        result["wp_enhanced"] = float(models["enhanced"].predict_proba(X)[0, 1])

    # Full (with net rating + pace)
    if models.get("full"):
        strength_diff = home_stats["win_pct"] - away_stats["win_pct"]
        net_rating_diff = home_stats["net_rating"] - away_stats["net_rating"]
        expected_pace = (home_stats["pace"] + away_stats["pace"]) / 2

        X = np.array([[
            margin, time_frac, margin * time_frac,
            home_stats["win_pct"], away_stats["win_pct"], home_stats["home_win_pct"],
            home_stats["last10"], away_stats["last10"], strength_diff,
            home_stats["net_rating"], away_stats["net_rating"], net_rating_diff,
            home_stats["pace"], away_stats["pace"], expected_pace,
            strength_diff * time_frac,
            margin * strength_diff,
            net_rating_diff * time_frac,
            margin * net_rating_diff,
            expected_pace / 210.0 - 1.0,
        ]])
        result["wp_full"] = float(models["full"].predict_proba(X)[0, 1])

    # Best available
    result["wp_best"] = result.get("wp_full", result.get("wp_enhanced", result.get("wp_baseline", 0.5)))

    return result


class NBALiveClient:
    """
    Fetch live NBA game data from the NBA's CDN endpoints.
    
    These endpoints don't require authentication and are updated in near-real-time.
    More reliable than nba_api for live data since they're simple JSON fetches.
    """

    # NBA CDN endpoints for live data
    SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    GAME_DETAIL_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
        })

    def get_todays_scoreboard(self) -> dict:
        """Fetch today's scoreboard with all games."""
        resp = self.session.get(self.SCOREBOARD_URL)
        resp.raise_for_status()
        return resp.json()

    def get_live_games(self) -> list[dict]:
        """
        Get all of today's games with current state.
        Returns a simplified list of game states.
        """
        data = self.get_todays_scoreboard()
        scoreboard = data.get("scoreboard", {})
        games = scoreboard.get("games", [])

        results = []
        for game in games:
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})

            game_status = game.get("gameStatus", 0)
            # gameStatus: 1 = not started, 2 = in progress, 3 = final

            period = game.get("period", 0)
            clock = game.get("gameClock", "PT00M00.00S")

            home_score = home.get("score", 0)
            away_score = away.get("score", 0)

            # Calculate seconds remaining
            if game_status == 2:  # live
                secs_remaining = seconds_remaining_in_game(period, clock)
            elif game_status == 3:  # final
                secs_remaining = 0
            else:
                secs_remaining = 2880  # not started

            results.append({
                "game_id": game.get("gameId", ""),
                "game_status": game_status,
                "game_status_text": game.get("gameStatusText", ""),
                "home_team": home.get("teamTricode", ""),
                "home_team_id": home.get("teamId", 0),
                "away_team": away.get("teamTricode", ""),
                "away_team_id": away.get("teamId", 0),
                "home_score": home_score,
                "away_score": away_score,
                "home_margin": home_score - away_score,
                "period": period,
                "game_clock": clock,
                "seconds_remaining": secs_remaining,
                "home_timeouts": home.get("timeoutsRemaining", 0),
                "away_timeouts": away.get("timeoutsRemaining", 0),
            })

        return results


def run_game_state_logger(
    interval: int = 10,
    dry_run: bool = False,
    with_model: bool = False,
):
    """
    Main loop: log live NBA game states at regular intervals.
    Optionally computes live win probabilities using the trained model.
    """
    client = NBALiveClient()

    today_str = date.today().strftime("%Y%m%d")
    log_path = LOGS_DIR / f"nba_game_state_{today_str}.jsonl"
    cumulative_path = DATA_DIR / "nba_game_state.jsonl"

    logger.info("Starting NBA live game state logger")
    logger.info(f"  Poll interval: {interval}s")
    logger.info(f"  Log file: {log_path}")
    logger.info(f"  Model predictions: {with_model}")

    # Load model and team stats if requested
    models = None
    team_stats_cache = None
    if with_model:
        models = load_model()
        if models:
            tiers = [k for k in ["baseline", "enhanced", "full"] if models.get(k)]
            logger.info(f"  Model tiers loaded: {', '.join(tiers)}")
            team_stats_cache = TeamStatsCache(refresh_interval=1800)
        else:
            logger.warning("  Model not available — logging without predictions")

    snapshot_count = 0

    while True:
        try:
            games = client.get_live_games()
            live_games = [g for g in games if g["game_status"] == 2]

            if not live_games:
                upcoming = [g for g in games if g["game_status"] == 1]
                if upcoming:
                    logger.debug(
                        f"No live games. {len(upcoming)} upcoming. Waiting..."
                    )
                else:
                    logger.debug("No games today or all finished. Waiting...")
                time.sleep(interval * 3)
                continue

            for game in live_games:
                record = {
                    "timestamp": utcnow_iso(),
                    **game,
                }

                # Add win probability predictions
                if models and team_stats_cache:
                    home_stats = team_stats_cache.get(game["home_team"])
                    away_stats = team_stats_cache.get(game["away_team"])

                    wp = compute_live_win_prob(
                        models,
                        game["home_margin"],
                        game["seconds_remaining"],
                        home_stats,
                        away_stats,
                    )
                    record.update(wp)
                    record["home_win_pct"] = home_stats["win_pct"]
                    record["away_win_pct"] = away_stats["win_pct"]
                    record["home_net_rating"] = home_stats["net_rating"]
                    record["away_net_rating"] = away_stats["net_rating"]

                snapshot_count += 1

                if dry_run:
                    wp_str = ""
                    if "wp_best" in record:
                        wp_str = f" | WP: {record['wp_best']:.1%}"
                    logger.info(
                        f"  {game['away_team']}@{game['home_team']} "
                        f"{game['away_score']}-{game['home_score']} "
                        f"Q{game['period']} {game['game_clock']} "
                        f"(margin: {game['home_margin']:+d}, "
                        f"secs_rem: {game['seconds_remaining']}){wp_str}"
                    )
                else:
                    append_jsonl(log_path, record)
                    append_jsonl(cumulative_path, record)

            if snapshot_count % 50 == 0 and snapshot_count > 0:
                logger.info(f"  Total game-state snapshots: {snapshot_count:,}")

        except KeyboardInterrupt:
            logger.info(f"\nStopped. Total snapshots: {snapshot_count:,}")
            break
        except requests.RequestException as e:
            logger.warning(f"NBA API error: {e}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(5)

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Log live NBA game states")
    parser.add_argument(
        "--interval", type=int, default=10, help="Poll interval in seconds"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print without saving"
    )
    parser.add_argument(
        "--with-model", action="store_true",
        help="Include live win probability predictions from trained model",
    )
    args = parser.parse_args()

    run_game_state_logger(
        interval=args.interval,
        dry_run=args.dry_run,
        with_model=args.with_model,
    )


if __name__ == "__main__":
    main()
