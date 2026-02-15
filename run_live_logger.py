"""
run_live_logger.py
==================
Combined launcher: runs both the Kalshi price logger and NBA game state
logger in parallel threads, so data is synchronized by timestamp.

Usage:
    python scripts/run_live_logger.py
    python scripts/run_live_logger.py --interval 10 --dry-run

This is the main script to run during live NBA games to build your database.
"""

import argparse
import sys
import threading
from pathlib import Path

from loguru import logger

sys.path.append(str(Path(__file__).parent))


def run_nba_logger(interval: int, dry_run: bool):
    """Thread target for NBA game state logging."""
    from nba_live_game_state_04 import run_game_state_logger
    # Rename to avoid import issues — or just import directly
    import importlib
    mod = importlib.import_module("04_nba_live_game_state")
    mod.run_game_state_logger(interval=interval, dry_run=dry_run)


def run_kalshi_logger(interval: int, dry_run: bool):
    """Thread target for Kalshi price logging."""
    import importlib
    mod = importlib.import_module("03_kalshi_live_logger")
    
    client = mod.KalshiClient()
    try:
        client.login_from_env()
    except ValueError as e:
        logger.error(f"Kalshi login failed: {e}")
        return
    
    mod.run_logger(client, interval=interval, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Run combined Kalshi + NBA live loggers"
    )
    parser.add_argument("--interval", type=int, default=10, help="Poll interval (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Print without saving")
    parser.add_argument(
        "--nba-only", action="store_true",
        help="Only run NBA game state logger (no Kalshi credentials needed)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KALSHI NBA LIVE LOGGER")
    logger.info("=" * 60)
    logger.info(f"Poll interval: {args.interval}s")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("Press Ctrl+C to stop\n")

    threads = []

    # Always run NBA logger
    nba_thread = threading.Thread(
        target=run_nba_logger,
        args=(args.interval, args.dry_run),
        daemon=True,
        name="nba-logger",
    )
    threads.append(nba_thread)

    if not args.nba_only:
        kalshi_thread = threading.Thread(
            target=run_kalshi_logger,
            args=(args.interval, args.dry_run),
            daemon=True,
            name="kalshi-logger",
        )
        threads.append(kalshi_thread)

    # Start all threads
    for t in threads:
        logger.info(f"Starting {t.name}...")
        t.start()

    # Wait for threads (Ctrl+C will exit)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")


if __name__ == "__main__":
    main()
