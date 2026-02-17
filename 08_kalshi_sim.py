"""
08_kalshi_sim.py
================
Kalshi paper trading simulator and live performance tracker.

Collects live Kalshi prices alongside NBA game states, computes model fair value,
detects edge opportunities, and tracks hypothetical P&L in real time.

Key features:
    - Score-delay awareness: if Kalshi price moves but our score feed hasn't updated,
      we flag a "stale score" and reduce confidence (someone scored, we just don't know yet)
    - Timeout detection: during timeouts the score is stable and known. Trades during
      timeouts have the highest signal confidence because our reference score matches
      what the market sees. We upweight these.
    - Momentum-aware edge: uses the momentum model tier when scoring runs are detected
    - Paper P&L tracking with Kelly sizing

Usage:
    python 08_kalshi_sim.py --live                    # live paper trading
    python 08_kalshi_sim.py --replay                  # replay from logged data
    python 08_kalshi_sim.py --live --dry-run           # print signals without logging
    python 08_kalshi_sim.py --report                  # summarize past performance

Output:
    data/kalshi_sim_trades.jsonl     — individual trade signals
    data/kalshi_sim_summary.jsonl    — session summaries
    analysis/kalshi_sim_report.csv   — aggregated performance
"""

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).parent))
from utils import DATA_DIR, LOGS_DIR, PROJECT_ROOT, append_jsonl, read_jsonl, utcnow_iso


# ── Score Delay & Timeout Detection ──────────────────────────────────────────


class ScoreDelayDetector:
    """
    Detect when the market has information we don't have yet.

    The NBA CDN feed can lag 5-30 seconds behind what courtside observers
    (and market makers) see. Key signals:

    1. STALE SCORE: Kalshi price moves significantly (>3 cents) but our score
       hasn't changed → someone likely scored, our feed is behind.
       Action: DO NOT trade (our model inputs are wrong).

    2. TIMEOUT: game clock not ticking, score stable across multiple polls.
       During timeouts our score is definitively correct.
       Action: HIGH CONFIDENCE trades — our reference score matches market's.

    3. FRESH SCORE: our score just changed (within last 2 polls).
       Action: NORMAL confidence — score is current but may not have propagated
       to all market participants yet.
    """

    def __init__(self, stale_price_threshold: float = 0.03):
        self.stale_price_threshold = stale_price_threshold
        # game_id → tracking state
        self._game_state: dict[str, dict] = {}

    def update(
        self,
        game_id: str,
        home_score: int,
        away_score: int,
        seconds_remaining: int,
        kalshi_mid: float = None,
    ) -> dict:
        """
        Update tracker and return signal confidence assessment.

        Returns:
            dict with:
                confidence: "timeout" | "fresh" | "stale" | "normal"
                confidence_weight: 1.5 (timeout), 1.0 (fresh/normal), 0.0 (stale)
                stale_flag: True if we suspect our score is behind
                timeout_flag: True if timeout detected
                price_change: absolute change in Kalshi mid since last poll
                score_changed: True if score changed this poll
        """
        state = self._game_state.setdefault(game_id, {
            "last_home_score": None,
            "last_away_score": None,
            "last_secs": None,
            "last_kalshi_mid": None,
            "score_stable_count": 0,
            "clock_stable_count": 0,
        })

        score_changed = False
        if state["last_home_score"] is not None:
            score_changed = (
                home_score != state["last_home_score"] or
                away_score != state["last_away_score"]
            )

        clock_frozen = False
        if state["last_secs"] is not None:
            clock_frozen = (seconds_remaining == state["last_secs"])

        price_change = 0.0
        if kalshi_mid is not None and state["last_kalshi_mid"] is not None:
            price_change = abs(kalshi_mid - state["last_kalshi_mid"])

        # Update stable counters
        if score_changed:
            state["score_stable_count"] = 0
        else:
            state["score_stable_count"] += 1

        if clock_frozen:
            state["clock_stable_count"] += 1
        else:
            state["clock_stable_count"] = 0

        # Detect timeout: clock frozen AND score stable for 2+ polls
        timeout_flag = (state["clock_stable_count"] >= 2 and
                        state["score_stable_count"] >= 2)

        # Detect stale score: price moved significantly but score didn't change
        stale_flag = (
            price_change >= self.stale_price_threshold and
            not score_changed and
            not timeout_flag  # during timeouts, price moves are re-pricing, not new scores
        )

        # Determine confidence
        if stale_flag:
            confidence = "stale"
            confidence_weight = 0.0  # DO NOT TRADE
        elif timeout_flag:
            confidence = "timeout"
            confidence_weight = 1.5  # PREMIUM signal — score is known
        elif score_changed:
            confidence = "fresh"
            confidence_weight = 1.0
        else:
            confidence = "normal"
            confidence_weight = 1.0

        # Update state
        state["last_home_score"] = home_score
        state["last_away_score"] = away_score
        state["last_secs"] = seconds_remaining
        state["last_kalshi_mid"] = kalshi_mid

        return {
            "confidence": confidence,
            "confidence_weight": confidence_weight,
            "stale_flag": stale_flag,
            "timeout_flag": timeout_flag,
            "price_change": round(price_change, 4),
            "score_changed": score_changed,
        }

    def clear_game(self, game_id: str):
        self._game_state.pop(game_id, None)


# ── Paper Trading Engine ─────────────────────────────────────────────────────


class PaperTrader:
    """
    Paper trading engine that tracks hypothetical positions and P&L.

    Trades are triggered when:
    1. |edge| exceeds threshold
    2. Signal confidence is not "stale"
    3. Kelly fraction is meaningful (> 0.5%)
    """

    def __init__(
        self,
        bankroll: float = 1000.0,
        edge_threshold_cents: float = 5.0,
        max_bet_pct: float = 0.05,
        max_bet_dollars: float = 500.0,
    ):
        self.starting_bankroll = bankroll
        self.bankroll = bankroll
        self.edge_threshold = edge_threshold_cents / 100.0
        self.max_bet_pct = max_bet_pct
        self.max_bet_dollars = max_bet_dollars
        self.trades: list[dict] = []
        self.peak_bankroll = bankroll
        self.max_drawdown = 0.0
        # Track open positions per game
        self.open_positions: dict[str, list[dict]] = defaultdict(list)

    def evaluate_trade(
        self,
        game_id: str,
        model_prob: float,
        kalshi_mid: float,
        confidence: dict,
        margin: int,
        seconds_remaining: int,
        momentum_features: dict = None,
        home_team: str = "",
        away_team: str = "",
    ) -> dict | None:
        """
        Evaluate whether to take a trade and record it.

        Returns trade dict if triggered, None otherwise.
        """
        kalshi_prob = kalshi_mid / 100.0
        edge = model_prob - kalshi_prob
        abs_edge = abs(edge)

        # Skip if stale score
        if confidence["confidence"] == "stale":
            return None

        # Check threshold
        if abs_edge < self.edge_threshold:
            return None

        # Kelly fraction
        if edge > 0:
            if kalshi_prob >= 1.0 or kalshi_prob <= 0.0:
                return None
            kelly_f = (edge) / (1 - kalshi_prob)
        else:
            if kalshi_prob >= 1.0 or kalshi_prob <= 0.0:
                return None
            kelly_f = (-edge) / kalshi_prob

        kelly_f = max(0.0, kelly_f) * 0.5  # half-Kelly
        kelly_f = min(kelly_f, 0.25)

        if kelly_f < 0.005:
            return None

        # Apply confidence weighting
        adjusted_kelly = kelly_f * confidence["confidence_weight"]
        bet_size = min(
            adjusted_kelly * self.bankroll,
            self.bankroll * self.max_bet_pct,
            self.max_bet_dollars,
        )

        direction = "YES" if edge > 0 else "NO"

        trade = {
            "timestamp": utcnow_iso(),
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "margin": margin,
            "seconds_remaining": seconds_remaining,
            "model_prob": round(model_prob, 4),
            "kalshi_mid": kalshi_mid,
            "kalshi_prob": round(kalshi_prob, 4),
            "edge": round(edge, 4),
            "abs_edge": round(abs_edge, 4),
            "direction": direction,
            "kelly_f": round(kelly_f, 4),
            "confidence": confidence["confidence"],
            "confidence_weight": confidence["confidence_weight"],
            "bet_size": round(bet_size, 2),
            "bankroll_before": round(self.bankroll, 2),
            "status": "open",
            "realized_pnl": None,
        }

        # Add momentum info if available
        if momentum_features:
            trade["run_team"] = momentum_features.get("run_team", 0)
            trade["run_points"] = momentum_features.get("run_points", 0)
            trade["momentum"] = momentum_features.get("momentum", 0.0)

        self.trades.append(trade)
        self.open_positions[game_id].append(trade)

        return trade

    def settle_game(self, game_id: str, home_won: bool):
        """Settle all open positions for a finished game."""
        positions = self.open_positions.pop(game_id, [])
        for trade in positions:
            if trade["direction"] == "YES":
                if home_won:
                    pnl = (1.0 - trade["kalshi_prob"]) * trade["bet_size"]
                else:
                    pnl = -trade["kalshi_prob"] * trade["bet_size"]
            else:  # NO
                if not home_won:
                    pnl = trade["kalshi_prob"] * trade["bet_size"]
                else:
                    pnl = -(1.0 - trade["kalshi_prob"]) * trade["bet_size"]

            trade["realized_pnl"] = round(pnl, 2)
            trade["status"] = "settled"
            trade["home_won"] = home_won

            self.bankroll += pnl
            self.bankroll = max(self.bankroll, 1.0)
            self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
            dd = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
            self.max_drawdown = max(self.max_drawdown, dd)

        return positions

    def get_summary(self) -> dict:
        """Get current paper trading summary."""
        settled = [t for t in self.trades if t["status"] == "settled"]
        open_trades = [t for t in self.trades if t["status"] == "open"]

        winners = [t for t in settled if t["realized_pnl"] and t["realized_pnl"] > 0]
        total_pnl = sum(t["realized_pnl"] for t in settled if t["realized_pnl"])

        # By confidence level
        by_conf = defaultdict(list)
        for t in settled:
            by_conf[t["confidence"]].append(t)

        conf_summary = {}
        for conf, trades in by_conf.items():
            w = len([t for t in trades if t["realized_pnl"] and t["realized_pnl"] > 0])
            p = sum(t["realized_pnl"] for t in trades if t["realized_pnl"])
            conf_summary[conf] = {"trades": len(trades), "wins": w, "pnl": round(p, 2)}

        return {
            "total_trades": len(self.trades),
            "settled_trades": len(settled),
            "open_trades": len(open_trades),
            "winners": len(winners),
            "win_rate": len(winners) / max(1, len(settled)),
            "total_pnl": round(total_pnl, 2),
            "bankroll": round(self.bankroll, 2),
            "return_pct": round((self.bankroll / self.starting_bankroll - 1) * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "by_confidence": conf_summary,
        }


# ── Live Paper Trading Loop ─────────────────────────────────────────────────


def run_live_sim(
    interval: int = 10,
    bankroll: float = 1000.0,
    edge_threshold: float = 5.0,
    dry_run: bool = False,
):
    """
    Run live paper trading simulation.

    Polls both NBA game state and Kalshi prices, computes model fair value,
    detects edge, and tracks paper P&L.
    """
    import importlib
    nba_mod = importlib.import_module("04_nba_live_game_state")
    kalshi_mod = importlib.import_module("03_kalshi_live_logger")

    # Load model
    model_path = DATA_DIR / "win_prob_model.pkl"
    if not model_path.exists():
        logger.error("No model found. Run 02_build_win_prob_model.py first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        models = pickle.load(f)

    tiers = [k for k in ["baseline", "enhanced", "full", "momentum"] if models.get(k)]
    logger.info(f"Model tiers: {', '.join(tiers)}")

    # Initialize components
    nba_client = nba_mod.NBALiveClient()
    team_stats = nba_mod.TeamStatsCache(refresh_interval=1800)
    momentum_tracker = nba_mod.GameMomentumTracker()
    delay_detector = ScoreDelayDetector()
    paper_trader = PaperTrader(
        bankroll=bankroll,
        edge_threshold_cents=edge_threshold,
    )

    # Try Kalshi client
    kalshi_client = None
    try:
        kalshi_client = kalshi_mod.KalshiClient()
        kalshi_client.login_from_env()
        logger.info("Kalshi connected")
    except Exception as e:
        logger.warning(f"Kalshi not available: {e}")
        logger.info("Running in NBA-only mode (no live Kalshi prices)")

    today_str = date.today().strftime("%Y%m%d")
    trades_path = DATA_DIR / "kalshi_sim_trades.jsonl"
    session_log = LOGS_DIR / f"kalshi_sim_{today_str}.jsonl"

    logger.info(f"\n{'='*60}")
    logger.info("KALSHI PAPER TRADING SIMULATOR")
    logger.info(f"{'='*60}")
    logger.info(f"Bankroll: ${bankroll:.0f}")
    logger.info(f"Edge threshold: {edge_threshold:.0f} cents")
    logger.info(f"Poll interval: {interval}s")
    logger.info(f"Trades log: {trades_path}")
    logger.info("Press Ctrl+C to stop\n")

    # Cache Kalshi market → game mapping
    kalshi_markets = {}  # ticker → market dict
    last_market_refresh = 0
    market_refresh_interval = 120  # refresh every 2 minutes

    poll_count = 0

    while True:
        try:
            # Get NBA games
            games = nba_client.get_live_games()
            live_games = [g for g in games if g["game_status"] == 2]
            finished = [g for g in games if g["game_status"] == 3]

            # Settle finished games
            for g in finished:
                gid = g["game_id"]
                if paper_trader.open_positions.get(gid):
                    home_won = g["home_score"] > g["away_score"]
                    settled = paper_trader.settle_game(gid, home_won)
                    for t in settled:
                        logger.info(
                            f"  SETTLED: {t['home_team']}v{t['away_team']} "
                            f"{t['direction']} @ {t['kalshi_mid']}¢ → "
                            f"PnL: ${t['realized_pnl']:+.2f}"
                        )
                        if not dry_run:
                            append_jsonl(trades_path, t)
                momentum_tracker.clear_game(g["game_id"])
                delay_detector.clear_game(g["game_id"])

            if not live_games:
                upcoming = [g for g in games if g["game_status"] == 1]
                if upcoming:
                    logger.debug(f"No live games. {len(upcoming)} upcoming.")
                time.sleep(interval * 3)
                continue

            # Refresh Kalshi markets periodically
            if kalshi_client and time.time() - last_market_refresh > market_refresh_interval:
                try:
                    markets = kalshi_mod.find_nba_markets(kalshi_client)
                    kalshi_markets = {}
                    for m in markets:
                        teams = kalshi_mod.extract_teams_from_market(m)
                        if teams:
                            # Key by sorted team pair for easy lookup
                            key = tuple(sorted(teams))
                            kalshi_markets[key] = m
                    last_market_refresh = time.time()
                    logger.debug(f"Refreshed Kalshi markets: {len(kalshi_markets)} NBA markets")
                except Exception as e:
                    logger.warning(f"Kalshi market refresh failed: {e}")

            # Process each live game
            for game in live_games:
                gid = game["game_id"]
                ht, at = game["home_team"], game["away_team"]

                # Get team stats
                home_stats = team_stats.get(ht)
                away_stats = team_stats.get(at)

                # Update momentum
                mom_features = momentum_tracker.update(
                    gid, game["home_score"], game["away_score"]
                )

                # Compute model win probability
                wp = nba_mod.compute_live_win_prob(
                    models,
                    game["home_margin"],
                    game["seconds_remaining"],
                    home_stats,
                    away_stats,
                    momentum_features=mom_features,
                )
                model_prob = wp.get("wp_best", 0.5)

                # Try to find matching Kalshi market
                team_key = tuple(sorted([ht, at]))
                kalshi_market = kalshi_markets.get(team_key)
                kalshi_mid = None

                if kalshi_market and kalshi_client:
                    try:
                        snapshot = kalshi_mod.snapshot_market(kalshi_client, kalshi_market)
                        kalshi_mid = snapshot.get("yes_mid")
                    except Exception:
                        pass

                # Detect score delay
                confidence = delay_detector.update(
                    gid,
                    game["home_score"],
                    game["away_score"],
                    game["seconds_remaining"],
                    kalshi_mid=(kalshi_mid / 100.0) if kalshi_mid else None,
                )

                # Build log record
                record = {
                    "timestamp": utcnow_iso(),
                    "game_id": gid,
                    "home_team": ht,
                    "away_team": at,
                    "home_score": game["home_score"],
                    "away_score": game["away_score"],
                    "margin": game["home_margin"],
                    "seconds_remaining": game["seconds_remaining"],
                    "model_prob": round(model_prob, 4),
                    "kalshi_mid": kalshi_mid,
                    **mom_features,
                    **confidence,
                    **{k: round(v, 4) for k, v in wp.items()},
                }

                if not dry_run:
                    append_jsonl(session_log, record)

                # Evaluate trade
                if kalshi_mid is not None:
                    trade = paper_trader.evaluate_trade(
                        game_id=gid,
                        model_prob=model_prob,
                        kalshi_mid=kalshi_mid,
                        confidence=confidence,
                        margin=game["home_margin"],
                        seconds_remaining=game["seconds_remaining"],
                        momentum_features=mom_features,
                        home_team=ht,
                        away_team=at,
                    )
                    if trade:
                        emoji = {"timeout": "[TIMEOUT]", "fresh": "[FRESH]", "normal": ""}
                        conf_label = emoji.get(confidence["confidence"], "")
                        logger.info(
                            f"  TRADE {conf_label}: {at}@{ht} "
                            f"{trade['direction']} @ {kalshi_mid}¢ "
                            f"(model={model_prob:.1%}, edge={trade['edge']*100:+.1f}¢, "
                            f"kelly={trade['kelly_f']:.1%}, bet=${trade['bet_size']:.0f})"
                        )
                        if mom_features.get("run_points", 0) >= 6:
                            logger.info(
                                f"    Momentum: {'home' if mom_features['run_team'] > 0 else 'away'} "
                                f"on {mom_features['run_points']}-0 run"
                            )
                        if not dry_run:
                            append_jsonl(trades_path, trade)

            poll_count += 1
            if poll_count % 30 == 0:
                summary = paper_trader.get_summary()
                logger.info(
                    f"\n  Session: {summary['total_trades']} trades, "
                    f"{summary['settled_trades']} settled, "
                    f"PnL: ${summary['total_pnl']:+.2f}, "
                    f"Bankroll: ${summary['bankroll']:.0f} "
                    f"({summary['return_pct']:+.1f}%)\n"
                )

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(5)
            continue

        time.sleep(interval)

    # Final summary
    summary = paper_trader.get_summary()
    logger.info(f"\n{'='*60}")
    logger.info("SESSION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total trades: {summary['total_trades']}")
    logger.info(f"Settled: {summary['settled_trades']}, Open: {summary['open_trades']}")
    logger.info(f"Win rate: {summary['win_rate']:.1%}")
    logger.info(f"Total PnL: ${summary['total_pnl']:+.2f}")
    logger.info(f"Bankroll: ${summary['bankroll']:.2f} ({summary['return_pct']:+.1f}%)")
    logger.info(f"Max drawdown: {summary['max_drawdown_pct']:.1f}%")

    if summary.get("by_confidence"):
        logger.info("\nBy Signal Confidence:")
        for conf, stats in summary["by_confidence"].items():
            wr = stats["wins"] / max(1, stats["trades"])
            logger.info(
                f"  {conf:10s}: {stats['trades']} trades, "
                f"win rate {wr:.1%}, PnL ${stats['pnl']:+.2f}"
            )

    # Save session summary
    if not dry_run:
        summary["session_date"] = today_str
        summary["timestamp"] = utcnow_iso()
        append_jsonl(DATA_DIR / "kalshi_sim_summary.jsonl", summary)


# ── Replay Mode ──────────────────────────────────────────────────────────────


def run_replay(
    bankroll: float = 1000.0,
    edge_threshold: float = 5.0,
):
    """
    Replay paper trading from logged data (Kalshi prices + NBA game state).
    Uses the same logic as live mode but replays from disk.
    """
    logger.info("Replaying from logged data...")

    # Load logged data
    nba_path = DATA_DIR / "nba_game_state.jsonl"
    kalshi_path = DATA_DIR / "kalshi_nba_prices.jsonl"

    if not nba_path.exists():
        logger.error(f"No NBA game state data at {nba_path}")
        sys.exit(1)

    nba_records = read_jsonl(nba_path)
    nba_df = pd.DataFrame(nba_records)
    nba_df["timestamp"] = pd.to_datetime(nba_df["timestamp"])
    logger.info(f"Loaded {len(nba_df):,} NBA snapshots")

    kalshi_df = None
    if kalshi_path.exists():
        kalshi_records = read_jsonl(kalshi_path)
        if kalshi_records:
            kalshi_df = pd.DataFrame(kalshi_records)
            kalshi_df["timestamp"] = pd.to_datetime(kalshi_df["timestamp"])
            logger.info(f"Loaded {len(kalshi_df):,} Kalshi snapshots")

    if kalshi_df is None or kalshi_df.empty:
        logger.warning("No Kalshi data — simulating market prices from empirical table")
        # Load empirical table for market price simulation
        table_path = DATA_DIR / "win_prob_empirical.parquet"
        if table_path.exists():
            table = pd.read_parquet(table_path)
        else:
            logger.error("No empirical table either. Run 02_build_win_prob_model.py first.")
            sys.exit(1)

    # Load model
    model_path = DATA_DIR / "win_prob_model.pkl"
    with open(model_path, "rb") as f:
        models = pickle.load(f)

    delay_detector = ScoreDelayDetector()
    paper_trader = PaperTrader(bankroll=bankroll, edge_threshold_cents=edge_threshold)

    # Group NBA data by game
    games = nba_df.groupby("game_id")

    for game_id, game_df in games:
        game_df = game_df.sort_values("timestamp")

        for _, row in game_df.iterrows():
            margin = row.get("home_margin", 0)
            secs = row.get("seconds_remaining", 0)
            model_prob = row.get("wp_best", row.get("wp_full", row.get("wp_baseline")))

            if model_prob is None or pd.isna(model_prob):
                continue

            # Find Kalshi price if available
            kalshi_mid = None
            if kalshi_df is not None:
                # Find nearest Kalshi snapshot
                ts = row["timestamp"]
                time_mask = (kalshi_df["timestamp"] - ts).abs() < pd.Timedelta(seconds=15)
                nearby = kalshi_df[time_mask]
                if len(nearby) > 0:
                    kalshi_mid = nearby.iloc[0].get("yes_mid")

            if kalshi_mid is None:
                continue

            confidence = delay_detector.update(
                str(game_id),
                int(row.get("home_score", 0)),
                int(row.get("away_score", 0)),
                int(secs),
                kalshi_mid=(kalshi_mid / 100.0) if kalshi_mid else None,
            )

            mom_features = {
                "run_team": row.get("run_team", 0),
                "run_length": row.get("run_length", 0),
                "run_points": row.get("run_points", 0),
                "momentum": row.get("momentum", 0.0),
                "scoring_burst": row.get("scoring_burst", 0),
                "margin_change_last5": row.get("margin_change_last5", 0),
            }

            paper_trader.evaluate_trade(
                game_id=str(game_id),
                model_prob=float(model_prob),
                kalshi_mid=float(kalshi_mid),
                confidence=confidence,
                margin=int(margin),
                seconds_remaining=int(secs),
                momentum_features=mom_features,
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
            )

        # Settle game (use last row's score)
        last = game_df.iloc[-1]
        if last.get("game_status", 0) == 3:
            home_won = last.get("home_score", 0) > last.get("away_score", 0)
            paper_trader.settle_game(str(game_id), home_won)

    summary = paper_trader.get_summary()
    logger.info(f"\n{'='*60}")
    logger.info("REPLAY SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total trades: {summary['total_trades']}")
    logger.info(f"Settled: {summary['settled_trades']}")
    logger.info(f"Win rate: {summary['win_rate']:.1%}")
    logger.info(f"PnL: ${summary['total_pnl']:+.2f}")
    logger.info(f"Bankroll: ${summary['bankroll']:.2f} ({summary['return_pct']:+.1f}%)")
    logger.info(f"Max drawdown: {summary['max_drawdown_pct']:.1f}%")

    if summary.get("by_confidence"):
        logger.info("\nBy Signal Confidence:")
        for conf, stats in summary["by_confidence"].items():
            wr = stats["wins"] / max(1, stats["trades"])
            logger.info(f"  {conf:10s}: {stats['trades']} trades, win rate {wr:.1%}, PnL ${stats['pnl']:+.2f}")

    return paper_trader


# ── Report Mode ──────────────────────────────────────────────────────────────


def generate_report():
    """Generate performance report from past paper trading sessions."""
    trades_path = DATA_DIR / "kalshi_sim_trades.jsonl"
    summary_path = DATA_DIR / "kalshi_sim_summary.jsonl"

    analysis_dir = PROJECT_ROOT / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    if not trades_path.exists():
        logger.error("No trade data found. Run --live or --replay first.")
        sys.exit(1)

    trades = pd.DataFrame(read_jsonl(trades_path))
    logger.info(f"Loaded {len(trades):,} trades")

    settled = trades[trades["status"] == "settled"].copy()
    if settled.empty:
        logger.warning("No settled trades to analyze.")
        return

    settled["timestamp"] = pd.to_datetime(settled["timestamp"])
    settled["date"] = settled["timestamp"].dt.date

    # Overall metrics
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE REPORT")
    logger.info(f"{'='*60}")

    winners = settled[settled["realized_pnl"] > 0]
    total_pnl = settled["realized_pnl"].sum()

    logger.info(f"Period: {settled['date'].min()} to {settled['date'].max()}")
    logger.info(f"Total settled trades: {len(settled)}")
    logger.info(f"Win rate: {len(winners)/len(settled):.1%}")
    logger.info(f"Total PnL: ${total_pnl:+.2f}")
    logger.info(f"Avg PnL per trade: ${settled['realized_pnl'].mean():+.2f}")

    # By confidence
    logger.info("\nBy Signal Confidence:")
    for conf, group in settled.groupby("confidence"):
        w = (group["realized_pnl"] > 0).sum()
        p = group["realized_pnl"].sum()
        logger.info(f"  {conf:10s}: {len(group)} trades, win rate {w/len(group):.1%}, PnL ${p:+.2f}")

    # By momentum
    if "run_points" in settled.columns:
        logger.info("\nBy Momentum (run points):")
        settled["run_bucket"] = pd.cut(
            settled["run_points"].fillna(0),
            bins=[-1, 0, 5, 10, 100],
            labels=["No run", "Small (1-5)", "Medium (6-10)", "Large (10+)"],
        )
        for bucket, group in settled.groupby("run_bucket", observed=True):
            if len(group) > 0:
                w = (group["realized_pnl"] > 0).sum()
                p = group["realized_pnl"].sum()
                logger.info(f"  {bucket:15s}: {len(group)} trades, win rate {w/len(group):.1%}, PnL ${p:+.2f}")

    # Save report
    settled.to_csv(analysis_dir / "kalshi_sim_report.csv", index=False)
    logger.info(f"\nSaved report → {analysis_dir / 'kalshi_sim_report.csv'}")

    # Daily P&L
    daily = settled.groupby("date").agg(
        trades=("realized_pnl", "count"),
        wins=("realized_pnl", lambda x: (x > 0).sum()),
        pnl=("realized_pnl", "sum"),
        avg_edge=("abs_edge", "mean"),
    ).reset_index()
    daily["win_rate"] = daily["wins"] / daily["trades"]

    logger.info("\nDaily Summary:")
    for _, row in daily.iterrows():
        logger.info(
            f"  {row['date']}: {row['trades']} trades, "
            f"win rate {row['win_rate']:.0%}, "
            f"PnL ${row['pnl']:+.2f}, "
            f"avg |edge| {row['avg_edge']*100:.1f}¢"
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Kalshi paper trading simulator"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run live paper trading")
    mode.add_argument("--replay", action="store_true", help="Replay from logged data")
    mode.add_argument("--report", action="store_true", help="Generate performance report")

    parser.add_argument("--interval", type=int, default=10, help="Poll interval (seconds)")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll")
    parser.add_argument("--threshold", type=float, default=5.0, help="Edge threshold (cents)")
    parser.add_argument("--dry-run", action="store_true", help="Print without saving")

    args = parser.parse_args()

    if args.live:
        run_live_sim(
            interval=args.interval,
            bankroll=args.bankroll,
            edge_threshold=args.threshold,
            dry_run=args.dry_run,
        )
    elif args.replay:
        run_replay(bankroll=args.bankroll, edge_threshold=args.threshold)
    elif args.report:
        generate_report()


if __name__ == "__main__":
    main()
