"""
03_kalshi_live_logger.py
========================
Continuously poll Kalshi's API for live NBA game markets and log prices
tick-by-tick to build a historical database for backtesting.

Logs to: logs/kalshi_nba_YYYYMMDD.jsonl

Each record contains:
    - timestamp (UTC)
    - market ticker/slug
    - yes_bid, yes_ask, yes_mid (cents, 1-99)
    - volume, open_interest
    - game metadata (teams, etc.)

Usage:
    python scripts/03_kalshi_live_logger.py
    python scripts/03_kalshi_live_logger.py --interval 5   # poll every 5 seconds
    python scripts/03_kalshi_live_logger.py --dry-run      # print but don't save

Notes:
    - Kalshi rate limit: ~10 requests/second for most endpoints
    - NBA markets are typically listed as "NBA" or team-specific tickers
    - Run this during game hours (typically 7 PM - 12 AM ET)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path

import requests
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from utils import (
    KALSHI_BASE_URL,
    KALSHI_ENV,
    LOGS_DIR,
    DATA_DIR,
    append_jsonl,
    utcnow_iso,
)


class KalshiClient:
    """Simple Kalshi REST API client."""

    def __init__(self):
        self.session = requests.Session()
        self.base_url = KALSHI_BASE_URL
        self.token = None

    def login(self, email: str, password: str):
        """Authenticate and store session token."""
        resp = self.session.post(
            f"{self.base_url}/login",
            json={"email": email, "password": password},
        )
        resp.raise_for_status()
        data = resp.json()
        self.token = data.get("token")
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })
        logger.info(f"Logged in to Kalshi ({KALSHI_ENV})")
        return self.token

    def login_from_env(self):
        """Login using .env credentials."""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        email = os.getenv("KALSHI_API_KEY", "")
        password = os.getenv("KALSHI_API_SECRET", "")
        if not email or email == "your_api_key_here":
            raise ValueError("Set KALSHI_API_KEY and KALSHI_API_SECRET in .env")
        return self.login(email, password)

    def get_events(self, series_ticker: str = None, status: str = "open", 
                   cursor: str = None, limit: int = 100) -> dict:
        """Fetch events (series of related markets)."""
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        resp = self.session.get(f"{self.base_url}/events", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_markets(self, event_ticker: str = None, series_ticker: str = None,
                    status: str = "open", cursor: str = None, limit: int = 100) -> dict:
        """Fetch markets, optionally filtered by event or series."""
        params = {"status": status, "limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        resp = self.session.get(f"{self.base_url}/markets", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_market(self, ticker: str) -> dict:
        """Fetch a single market by ticker."""
        resp = self.session.get(f"{self.base_url}/markets/{ticker}")
        resp.raise_for_status()
        return resp.json()

    def get_orderbook(self, ticker: str) -> dict:
        """Fetch orderbook for a market."""
        resp = self.session.get(f"{self.base_url}/orderbook/{ticker}")
        resp.raise_for_status()
        return resp.json()

    def get_market_history(self, ticker: str, limit: int = 100) -> dict:
        """Fetch trade history for a market."""
        params = {"limit": limit}
        resp = self.session.get(
            f"{self.base_url}/markets/{ticker}/trades", params=params
        )
        resp.raise_for_status()
        return resp.json()


def find_nba_markets(client: KalshiClient) -> list[dict]:
    """
    Find all currently active NBA game markets on Kalshi.
    
    Kalshi NBA markets typically have tickers/series containing "NBA" 
    or team abbreviations. The exact naming convention may change, 
    so we search broadly.
    """
    nba_markets = []
    
    # Search strategies — Kalshi's ticker naming evolves
    search_terms = ["NBA", "nba", "BASKETBALL"]
    
    for term in search_terms:
        try:
            # Search events first
            events_data = client.get_events(series_ticker=term)
            events = events_data.get("events", [])
            
            for event in events:
                event_ticker = event.get("event_ticker", "")
                # Get markets within this event
                markets_data = client.get_markets(event_ticker=event_ticker)
                markets = markets_data.get("markets", [])
                
                for market in markets:
                    nba_markets.append(market)
                    
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                continue
            raise

    # Also try a broad market search
    try:
        all_markets = client.get_markets(status="open", limit=200)
        for market in all_markets.get("markets", []):
            title = (market.get("title", "") + market.get("ticker", "")).upper()
            # Look for NBA-related keywords
            nba_keywords = [
                "NBA", "LAKERS", "CELTICS", "WARRIORS", "NETS", "KNICKS",
                "76ERS", "BUCKS", "SUNS", "NUGGETS", "HEAT", "MAVERICKS",
                "CLIPPERS", "BULLS", "ROCKETS", "SPURS", "RAPTORS", "HAWKS",
                "CAVALIERS", "THUNDER", "TIMBERWOLVES", "PELICANS", "KINGS",
                "PACERS", "MAGIC", "HORNETS", "WIZARDS", "PISTONS", "BLAZERS",
                "JAZZ", "GRIZZLIES",
            ]
            if any(kw in title for kw in nba_keywords):
                if market["ticker"] not in [m["ticker"] for m in nba_markets]:
                    nba_markets.append(market)
    except Exception as e:
        logger.warning(f"Broad market search failed: {e}")

    logger.info(f"Found {len(nba_markets)} NBA-related markets")
    return nba_markets


def snapshot_market(client: KalshiClient, market: dict) -> dict:
    """
    Take a snapshot of a market's current state including orderbook.
    """
    ticker = market["ticker"]
    
    # Base market info
    record = {
        "timestamp": utcnow_iso(),
        "ticker": ticker,
        "title": market.get("title", ""),
        "event_ticker": market.get("event_ticker", ""),
        "status": market.get("status", ""),
        "yes_bid": market.get("yes_bid", None),
        "yes_ask": market.get("yes_ask", None),
        "last_price": market.get("last_price", None),
        "volume": market.get("volume", 0),
        "open_interest": market.get("open_interest", 0),
        "close_time": market.get("close_time", ""),
    }
    
    # Compute mid price
    if record["yes_bid"] is not None and record["yes_ask"] is not None:
        record["yes_mid"] = (record["yes_bid"] + record["yes_ask"]) / 2
    else:
        record["yes_mid"] = record.get("last_price")
    
    # Try to get orderbook depth
    try:
        ob = client.get_orderbook(ticker)
        orderbook = ob.get("orderbook", {})
        record["yes_bids"] = orderbook.get("yes", [])[:5]  # top 5 levels
        record["no_bids"] = orderbook.get("no", [])[:5]
    except Exception:
        record["yes_bids"] = []
        record["no_bids"] = []

    return record


def run_logger(
    client: KalshiClient,
    interval: int = 10,
    dry_run: bool = False,
):
    """
    Main logging loop. Polls Kalshi for NBA markets at regular intervals.
    """
    today_str = date.today().strftime("%Y%m%d")
    log_path = LOGS_DIR / f"kalshi_nba_{today_str}.jsonl"
    
    # Also maintain a cumulative file for easy loading
    cumulative_path = DATA_DIR / "kalshi_nba_prices.jsonl"
    
    logger.info(f"Starting Kalshi NBA logger")
    logger.info(f"  Environment: {KALSHI_ENV}")
    logger.info(f"  Poll interval: {interval}s")
    logger.info(f"  Log file: {log_path}")
    logger.info(f"  Cumulative: {cumulative_path}")
    if dry_run:
        logger.info("  DRY RUN — not saving to disk")

    snapshot_count = 0
    
    while True:
        try:
            # Find current NBA markets
            markets = find_nba_markets(client)
            
            if not markets:
                logger.debug("No active NBA markets found, waiting...")
                time.sleep(interval * 3)  # wait longer if no markets
                continue

            # Snapshot each market
            for market in markets:
                try:
                    record = snapshot_market(client, market)
                    snapshot_count += 1

                    if dry_run:
                        logger.info(
                            f"  {record['ticker']}: "
                            f"bid={record['yes_bid']} "
                            f"ask={record['yes_ask']} "
                            f"mid={record['yes_mid']} "
                            f"vol={record['volume']}"
                        )
                    else:
                        append_jsonl(log_path, record)
                        append_jsonl(cumulative_path, record)

                    time.sleep(0.15)  # brief pause between market queries

                except Exception as e:
                    logger.warning(f"  Error snapshotting {market.get('ticker')}: {e}")

            if snapshot_count % 100 == 0 and snapshot_count > 0:
                logger.info(f"  Total snapshots: {snapshot_count:,}")

        except KeyboardInterrupt:
            logger.info(f"\nStopped. Total snapshots: {snapshot_count:,}")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)
            continue

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Log live Kalshi NBA market prices")
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Poll interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print snapshots without saving to disk",
    )
    args = parser.parse_args()

    client = KalshiClient()
    
    try:
        client.login_from_env()
    except ValueError as e:
        logger.error(str(e))
        logger.info(
            "\nTo set up Kalshi credentials:\n"
            "  1. Copy .env.example to .env\n"
            "  2. Fill in your Kalshi email and password\n"
            "  3. Set KALSHI_ENV to 'demo' or 'prod'\n"
        )
        sys.exit(1)

    run_logger(client, interval=args.interval, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
