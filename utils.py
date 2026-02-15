"""
Shared utilities for the Kalshi NBA trading bot.
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Kalshi Config ──────────────────────────────────────────────────────────
KALSHI_ENV = os.getenv("KALSHI_ENV", "demo")

KALSHI_BASE_URLS = {
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
}

KALSHI_WS_URLS = {
    "prod": "wss://api.elections.kalshi.com/trade-api/ws/v2",
    "demo": "wss://demo-api.kalshi.co/trade-api/ws/v2",
}

KALSHI_BASE_URL = KALSHI_BASE_URLS[KALSHI_ENV]
KALSHI_WS_URL = KALSHI_WS_URLS[KALSHI_ENV]


def get_kalshi_headers() -> dict:
    """Build auth headers for Kalshi REST API."""
    api_key = os.getenv("KALSHI_API_KEY", "")
    api_secret = os.getenv("KALSHI_API_SECRET", "")
    
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Set KALSHI_API_KEY and KALSHI_API_SECRET in your .env file. "
            "Get credentials from your Kalshi account settings."
        )
    
    # Kalshi uses a login endpoint to get a session token
    # For simplicity, we use API key auth directly in headers
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def kalshi_login(session) -> str:
    """
    Login to Kalshi and return a session token.
    Kalshi's v2 API uses email/password or API key login.
    """
    import requests
    
    api_key = os.getenv("KALSHI_API_KEY", "")
    api_secret = os.getenv("KALSHI_API_SECRET", "")
    
    resp = session.post(
        f"{KALSHI_BASE_URL}/login",
        json={"email": api_key, "password": api_secret},
    )
    resp.raise_for_status()
    token = resp.json()["token"]
    session.headers.update({"Authorization": f"Bearer {token}"})
    return token


# ── NBA Utilities ──────────────────────────────────────────────────────────

def seconds_remaining_in_game(period: int, period_clock: str) -> int:
    """
    Convert NBA game clock to total seconds remaining.
    
    Args:
        period: Current period (1-4, 5+ for OT)
        period_clock: Clock string like "5:32" or "PT05M32.00S" (NBA API format)
    
    Returns:
        Total seconds remaining in regulation (negative means OT territory)
    """
    # Parse clock string
    if "PT" in str(period_clock):
        # ISO duration format from NBA API: "PT05M32.00S"
        import re
        match = re.match(r"PT(\d+)M([\d.]+)S", str(period_clock))
        if match:
            minutes = int(match.group(1))
            seconds = int(float(match.group(2)))
        else:
            minutes, seconds = 0, 0
    elif ":" in str(period_clock):
        parts = str(period_clock).split(":")
        minutes = int(parts[0])
        seconds = int(parts[1]) if len(parts) > 1 else 0
    else:
        minutes, seconds = 0, 0
    
    clock_seconds = minutes * 60 + seconds
    
    if period <= 4:
        # Regulation: 4 periods of 12 minutes each
        remaining_full_periods = max(0, 4 - period)
        return remaining_full_periods * 12 * 60 + clock_seconds
    else:
        # OT: 5-minute periods, return as negative to flag OT
        return -(clock_seconds)


def margin_bucket(margin: int, bucket_size: int = 1) -> int:
    """Round margin to nearest bucket."""
    return round(margin / bucket_size) * bucket_size


def time_bucket(seconds_remaining: int, bucket_size: int = 30) -> int:
    """Round seconds remaining to nearest bucket."""
    return max(0, round(seconds_remaining / bucket_size) * bucket_size)


# ── File I/O ───────────────────────────────────────────────────────────────

def append_jsonl(filepath: Path, record: dict):
    """Append a JSON record to a JSONL file."""
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(filepath: Path) -> list[dict]:
    """Read all records from a JSONL file."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def utcnow_iso() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()
