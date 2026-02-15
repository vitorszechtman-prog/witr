# Kalshi NBA Live Odds Trading Bot

## Strategy
Exploit mispricings in Kalshi's live NBA game markets by comparing market-implied 
win probabilities against a fair-value model built from historical play-by-play data.

### Edge Sources
1. **Tail overreaction** — prices near 0/100 that overweight recent momentum
2. **Margin/time mispricing** — e.g., team priced at 80% while only up 2 with 8 min left
3. **Mean reversion after runs** — market overreacts to scoring runs

## Project Structure

```
witr/
├── 01_fetch_pbp_data.py        # Process NBA play-by-play CSV data into game states
├── 02_build_win_prob_model.py   # Build empirical win probability surface
├── 03_kalshi_live_logger.py     # Log live Kalshi NBA prices tick-by-tick
├── 04_nba_live_game_state.py    # Track live NBA game state (score, time, etc.)
├── 05_edge_analysis.py          # Compare model vs Kalshi prices (backtest)
├── utils.py                     # Shared utilities
├── data/
│   ├── raw/                     # Source CSV files from shufinskiy/nba_data
│   ├── game_states_*.parquet    # Per-play game state snapshots
│   ├── game_outcomes_*.parquet  # Game-level outcomes
│   └── win_prob_*.parquet       # Win probability lookup table
├── logs/                        # Live logger output
└── requirements.txt
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Process historical NBA play-by-play data (auto-downloads from GitHub if needed)
python 01_fetch_pbp_data.py --seasons 2022 2023 2024

# 4. Build win probability model
python scripts/02_build_win_prob_model.py

# 5. Start logging Kalshi prices (run during live games)
python scripts/03_kalshi_live_logger.py

# 6. After collecting data, analyze edge
python scripts/05_edge_analysis.py
```

## Kalshi API Setup
1. Create account at kalshi.com
2. Generate API keys in settings
3. Copy `.env.example` to `.env` and fill in your credentials

## Key Concepts

### Data Source
- Play-by-play data from [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) (sourced from stats.nba.com)
- 3 seasons (2022-2024): ~3,690 games, ~472K game-state snapshots

### Win Probability Model
- Empirical model: P(home_win | score_margin, seconds_remaining)
- Built from ~1,200+ NBA games per season of real play-by-play data
- Granularity: 1-point margin buckets × 30-second time buckets
- Smoothed via kernel regression to handle sparse cells

### Edge Detection
- Edge = Model_Prob - Kalshi_Price
- Trade when |Edge| > threshold (calibrated from historical analysis)
- Size based on Kelly criterion with fractional sizing
