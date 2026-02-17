"""
run_live_logger.py
==================
Combined launcher: runs both the Kalshi price logger and NBA game state
logger in parallel threads, so data is synchronized by timestamp.

Usage:
    python run_live_logger.py                        # full pipeline
    python run_live_logger.py --interval 10 --dry-run
    python run_live_logger.py --nba-only --with-model  # NBA + win prob, no Kalshi
    python run_live_logger.py --kalshi-ws             # use WebSocket for Kalshi
    python run_live_logger.py --with-sim              # also run paper trading sim

This is the main script to run during live NBA games to build your database.
"""

import argparse
import sys
import threading
from pathlib import Path

from loguru import logger

sys.path.append(str(Path(__file__).parent))


def run_nba_logger(interval: int, dry_run: bool, with_model: bool):
    """Thread target for NBA game state logging."""
    import importlib
    mod = importlib.import_module("04_nba_live_game_state")
    mod.run_game_state_logger(interval=interval, dry_run=dry_run, with_model=with_model)


def run_paper_trader(interval: int, bankroll: float, dry_run: bool):
    """Thread target for paper trading simulator."""
    import importlib
    mod = importlib.import_module("08_kalshi_sim")
    mod.run_live_sim(interval=interval, bankroll=bankroll, dry_run=dry_run)


def run_kalshi_logger(interval: int, dry_run: bool, use_ws: bool):
    """Thread target for Kalshi price logging."""
    import importlib
    mod = importlib.import_module("03_kalshi_live_logger")

    client = mod.KalshiClient()
    try:
        client.login_from_env()
    except ValueError as e:
        logger.error(f"Kalshi login failed: {e}")
        return

    if use_ws:
        mod.run_ws_logger(client, dry_run=dry_run)
    else:
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
    parser.add_argument(
        "--with-model", action="store_true",
        help="Include live win probability predictions in NBA logger"
    )
    parser.add_argument(
        "--kalshi-ws", action="store_true",
        help="Use WebSocket streaming for Kalshi (lower latency)"
    )
    parser.add_argument(
        "--with-sim", action="store_true",
        help="Also run paper trading simulator (requires --with-model)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Starting bankroll for paper trading simulator",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("WITR — LIVE GAME LOGGER")
    logger.info("=" * 60)
    logger.info(f"Poll interval: {args.interval}s")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Model predictions: {args.with_model}")
    logger.info(f"Kalshi mode: {'WebSocket' if args.kalshi_ws else 'REST polling'}")
    logger.info(f"Paper trading sim: {args.with_sim}")
    logger.info("Press Ctrl+C to stop\n")

    threads = []

    # Always run NBA logger
    nba_thread = threading.Thread(
        target=run_nba_logger,
        args=(args.interval, args.dry_run, args.with_model),
        daemon=True,
        name="nba-logger",
    )
    threads.append(nba_thread)

    if not args.nba_only:
        kalshi_thread = threading.Thread(
            target=run_kalshi_logger,
            args=(args.interval, args.dry_run, args.kalshi_ws),
            daemon=True,
            name="kalshi-logger",
        )
        threads.append(kalshi_thread)

    if args.with_sim:
        sim_thread = threading.Thread(
            target=run_paper_trader,
            args=(args.interval, args.bankroll, args.dry_run),
            daemon=True,
            name="paper-trader",
        )
        threads.append(sim_thread)

    for t in threads:
        logger.info(f"Starting {t.name}...")
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")


if __name__ == "__main__":
    main()
