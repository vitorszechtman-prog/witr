"""
04_nba_live_game_state.py
=========================
Track live NBA game state (score, clock, period) and log alongside
Kalshi prices for synchronized analysis.

Can run standalone or be imported by the main logger.

Usage:
    python scripts/04_nba_live_game_state.py                  # log live game states
    python scripts/04_nba_live_game_state.py --interval 15    # every 15 seconds

Output:
    logs/nba_game_state_YYYYMMDD.jsonl
    data/nba_game_state.jsonl  (cumulative)
"""

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from utils import DATA_DIR, LOGS_DIR, append_jsonl, utcnow_iso, seconds_remaining_in_game


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


def run_game_state_logger(interval: int = 10, dry_run: bool = False):
    """
    Main loop: log live NBA game states at regular intervals.
    """
    client = NBALiveClient()

    today_str = date.today().strftime("%Y%m%d")
    log_path = LOGS_DIR / f"nba_game_state_{today_str}.jsonl"
    cumulative_path = DATA_DIR / "nba_game_state.jsonl"

    logger.info("Starting NBA live game state logger")
    logger.info(f"  Poll interval: {interval}s")
    logger.info(f"  Log file: {log_path}")

    snapshot_count = 0

    while True:
        try:
            games = client.get_live_games()
            live_games = [g for g in games if g["game_status"] == 2]

            if not live_games:
                # Check if there are upcoming games
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
                snapshot_count += 1

                if dry_run:
                    logger.info(
                        f"  {game['away_team']}@{game['home_team']} "
                        f"{game['away_score']}-{game['home_score']} "
                        f"Q{game['period']} {game['game_clock']} "
                        f"(margin: {game['home_margin']:+d}, "
                        f"secs_rem: {game['seconds_remaining']})"
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
    args = parser.parse_args()

    run_game_state_logger(interval=args.interval, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
