"""
03_kalshi_live_logger.py
========================
Continuously poll Kalshi's API for live NBA game markets and log prices
tick-by-tick to build a historical database for backtesting.

Supports two modes:
    1. REST polling (default) — simpler, works everywhere
    2. WebSocket streaming — lower latency, real-time orderbook updates

Logs to: logs/kalshi_nba_YYYYMMDD.jsonl

Each record contains:
    - timestamp (UTC)
    - market ticker/slug
    - yes_bid, yes_ask, yes_mid (cents, 1-99)
    - volume, open_interest
    - game metadata (teams, etc.)

Usage:
    python 03_kalshi_live_logger.py                    # REST polling
    python 03_kalshi_live_logger.py --interval 5       # poll every 5 seconds
    python 03_kalshi_live_logger.py --mode ws           # WebSocket streaming
    python 03_kalshi_live_logger.py --dry-run           # print but don't save

Notes:
    - Kalshi rate limit: ~10 requests/second for most endpoints
    - NBA markets are typically listed as "NBA" or team-specific tickers
    - Run this during game hours (typically 7 PM - 12 AM ET)
"""

import argparse
import json
import sys
import time
import threading
from datetime import datetime, timezone, date
from pathlib import Path

import requests
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from utils import (
    KALSHI_BASE_URL,
    KALSHI_WS_URL,
    KALSHI_ENV,
    LOGS_DIR,
    DATA_DIR,
    append_jsonl,
    utcnow_iso,
    NBA_TEAM_NAMES,
)


class KalshiClient:
    """Kalshi REST + WebSocket API client."""

    def __init__(self):
        self.session = requests.Session()
        self.base_url = KALSHI_BASE_URL
        self.ws_url = KALSHI_WS_URL
        self.token = None
        self._ws = None
        self._ws_callbacks = []

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

    # ── WebSocket Methods ──────────────────────────────────────────────────

    def connect_ws(self, tickers: list[str], on_message=None):
        """
        Connect to Kalshi's WebSocket for real-time orderbook updates.

        Args:
            tickers: list of market tickers to subscribe to
            on_message: callback(ticker, data_dict) for each update
        """
        import websocket

        if on_message:
            self._ws_callbacks.append(on_message)

        def _on_open(ws):
            logger.info(f"WebSocket connected to {KALSHI_ENV}")
            # Authenticate
            ws.send(json.dumps({
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta", "ticker"],
                    "market_tickers": tickers,
                },
            }))
            logger.info(f"Subscribed to {len(tickers)} market(s)")

        def _on_message(ws, message):
            try:
                data = json.loads(message)
                msg_type = data.get("type", "")
                msg = data.get("msg", {})

                if msg_type in ("orderbook_snapshot", "orderbook_delta", "ticker"):
                    ticker = msg.get("market_ticker", "")
                    for cb in self._ws_callbacks:
                        cb(ticker, msg)
            except Exception as e:
                logger.warning(f"WS message parse error: {e}")

        def _on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def _on_close(ws, close_status_code, close_msg):
            logger.warning(f"WebSocket closed: {close_status_code} {close_msg}")

        ws_url = f"{self.ws_url}?token={self.token}" if self.token else self.ws_url
        self._ws = websocket.WebSocketApp(
            ws_url,
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )
        return self._ws

    def run_ws(self):
        """Run WebSocket in current thread (blocking)."""
        if self._ws:
            self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def run_ws_background(self):
        """Run WebSocket in a background thread."""
        if self._ws:
            t = threading.Thread(target=self.run_ws, daemon=True, name="kalshi-ws")
            t.start()
            return t


def extract_teams_from_market(market: dict) -> tuple[str, str] | None:
    """
    Try to extract (home_team_tricode, away_team_tricode) from a Kalshi market.
    Returns None if teams can't be identified.
    """
    title = market.get("title", "")
    ticker = market.get("ticker", "")
    search_text = f"{title} {ticker}".upper()

    matched_teams = []
    for tricode, names in NBA_TEAM_NAMES.items():
        for name in names:
            if name.upper() in search_text:
                matched_teams.append(tricode)
                break

    if len(matched_teams) == 2:
        return (matched_teams[0], matched_teams[1])
    return None


def find_nba_markets(client: KalshiClient) -> list[dict]:
    """
    Find all currently active NBA game markets on Kalshi.

    Searches by series ticker, event ticker, and broad keyword scan.
    Annotates each market with extracted team codes where possible.
    """
    nba_markets = []
    seen_tickers = set()

    def _add_market(m):
        t = m["ticker"]
        if t not in seen_tickers:
            seen_tickers.add(t)
            # Try to extract teams
            teams = extract_teams_from_market(m)
            if teams:
                m["_home_team"], m["_away_team"] = teams
            nba_markets.append(m)

    # Search strategies — Kalshi's ticker naming evolves
    search_terms = ["NBA", "nba", "BASKETBALL"]

    for term in search_terms:
        try:
            events_data = client.get_events(series_ticker=term)
            events = events_data.get("events", [])

            for event in events:
                event_ticker = event.get("event_ticker", "")
                markets_data = client.get_markets(event_ticker=event_ticker)
                for market in markets_data.get("markets", []):
                    _add_market(market)

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                continue
            raise

    # Also try a broad market search
    try:
        all_markets = client.get_markets(status="open", limit=200)
        for market in all_markets.get("markets", []):
            title = (market.get("title", "") + market.get("ticker", "")).upper()
            nba_keywords = list(NBA_TEAM_NAMES.keys()) + ["NBA", "BASKETBALL"]
            # Also match full team names
            for _tricode, names in NBA_TEAM_NAMES.items():
                nba_keywords.extend([n.upper() for n in names])
            if any(kw in title for kw in nba_keywords):
                _add_market(market)
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


def run_ws_logger(
    client: KalshiClient,
    dry_run: bool = False,
):
    """
    WebSocket-based logging. Lower latency than REST polling.
    Receives real-time orderbook updates and logs them.
    """
    today_str = date.today().strftime("%Y%m%d")
    log_path = LOGS_DIR / f"kalshi_nba_{today_str}.jsonl"
    cumulative_path = DATA_DIR / "kalshi_nba_prices.jsonl"

    logger.info("Starting Kalshi NBA WebSocket logger")
    logger.info(f"  Environment: {KALSHI_ENV}")
    logger.info(f"  Log file: {log_path}")

    # First find NBA markets via REST
    markets = find_nba_markets(client)
    if not markets:
        logger.warning("No active NBA markets found. Falling back to REST polling.")
        run_logger(client, interval=10, dry_run=dry_run)
        return

    tickers = [m["ticker"] for m in markets]
    ticker_info = {m["ticker"]: m for m in markets}
    snapshot_count = 0

    def on_ws_update(ticker: str, data: dict):
        nonlocal snapshot_count

        record = {
            "timestamp": utcnow_iso(),
            "ticker": ticker,
            "title": ticker_info.get(ticker, {}).get("title", ""),
            "event_ticker": ticker_info.get(ticker, {}).get("event_ticker", ""),
            "yes_bid": data.get("yes_bid"),
            "yes_ask": data.get("yes_ask"),
            "last_price": data.get("last_price"),
            "volume": data.get("volume", 0),
        }

        if record["yes_bid"] is not None and record["yes_ask"] is not None:
            record["yes_mid"] = (record["yes_bid"] + record["yes_ask"]) / 2
        else:
            record["yes_mid"] = record.get("last_price")

        # Include team info if extracted
        market_info = ticker_info.get(ticker, {})
        if "_home_team" in market_info:
            record["home_team"] = market_info["_home_team"]
            record["away_team"] = market_info["_away_team"]

        snapshot_count += 1

        if dry_run:
            logger.info(
                f"  WS {ticker}: bid={record['yes_bid']} "
                f"ask={record['yes_ask']} mid={record['yes_mid']}"
            )
        else:
            append_jsonl(log_path, record)
            append_jsonl(cumulative_path, record)

        if snapshot_count % 100 == 0:
            logger.info(f"  WS snapshots: {snapshot_count:,}")

    ws = client.connect_ws(tickers, on_message=on_ws_update)
    try:
        client.run_ws()
    except KeyboardInterrupt:
        logger.info(f"\nStopped. Total WS snapshots: {snapshot_count:,}")


def main():
    parser = argparse.ArgumentParser(description="Log live Kalshi NBA market prices")
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Poll interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--mode",
        choices=["rest", "ws"],
        default="rest",
        help="Logging mode: 'rest' (polling) or 'ws' (WebSocket streaming)",
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

    if args.mode == "ws":
        run_ws_logger(client, dry_run=args.dry_run)
    else:
        run_logger(client, interval=args.interval, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
