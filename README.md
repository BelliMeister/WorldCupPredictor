# WC 2026 Match Predictor

A machine learning system that predicts World Cup 2026 match outcomes, estimates total goals, and identifies value bets by comparing model probabilities against bookmaker odds.

---

## Quick Start

```bash
cd WorldCupPredictor

# 1. Pull latest match results from ESPN (run this daily)
python src/live_ingest.py

# 2. Retrain model on updated data
python src/train.py --quick

# 3. Predict a match
python src/predict.py --home "France" --away "England"
```

---

## Setup

**Requirements**
```bash
pip install pandas scikit-learn xgboost lightgbm optuna openpyxl
```

**API keys** (kept out of git — see [Secrets](#secrets--api-keys))
```bash
cp .secrets.env.example .secrets.env   # then paste your real keys
```

**First-time training** (builds models from 30,000+ historical matches)
```bash
python src/train.py          # full Optuna tuning — ~8 min
python src/train.py --quick  # default params — ~3 min
```

Models are saved to `models/` and reused by `predict.py`.

**One-time: build the player-props dataset** (per-player stats from API-Football)
```bash
python src/fetch_player_stats.py     # writes player_stats.csv + team_stats.csv
python src/fetch_match_xg.py         # writes data/raw/wc2026_xg.csv (live WC xG)
```
`fetch_player_stats.py` pulls each WC nation's **10 most recent internationals**
weighted by recency and competition importance (deeper history = steadier rates).
It also writes `team_stats.csv` — per-game **corners, fouls, yellow/red cards** from
API-Football (primary source for the discipline table; falls back to `master.csv`).
`fetch_match_xg.py` pulls per-match **expected goals (xG)** for finished WC 2026
fixtures — a cleaner attacking signal that sharpens the expected-goals split and
props as the tournament unfolds. Re-run both every matchday.

---

## Daily Workflow During the Tournament

### Step 1 — Pull overnight results
```bash
python src/live_ingest.py
```
Fetches the last 4 days of WC 2026 results from ESPN and patches them into the training dataset automatically. Shows a live scoreboard:

```
======================================================
  WC 2026 Scoreboard  [03:20 UTC]
======================================================

  FINISHED:
    Brazil 1 - 1 Morocco
    Haiti 0 - 1 Scotland

  UPCOMING:
    2026-06-14  Australia vs Turkey
    2026-06-14  Germany vs Curacao
```

### Step 2 — Retrain on fresh data
```bash
python src/train.py --quick
```
Only needed after new results are ingested. Takes ~3 minutes.

Or do both in one command:
```bash
python src/live_ingest.py --retrain
```

### Step 3 — Predict upcoming matches
```bash
python src/predict.py
```
It will prompt you interactively — no flags needed.

---

## Predict Command

Just run it and follow the prompts:

```
python src/predict.py
```

```
  WC 2026 Match Predictor
  ──────────────────────────────
  Home team: Turkiye
    → recognised as 'Turkey'
  Away team: Holland
    → recognised as 'Netherlands'

  Home odds  (press Enter to skip): 2.80
  Away odds  (press Enter to skip): 2.60
  Draw odds  (press Enter to skip): 3.20
```

**Odds are optional** — press Enter to skip any of them if you don't have them.

**Example output:**
```
==========================================================
  Turkey  vs  Netherlands
  Elo: 1859 vs 1878  |  FIFA Rank: #29 vs #14
  Model confidence: 39.9%
==========================================================

  OUTCOME PROBABILITIES
  Home Win      32.6%  █████████
  Draw          27.5%  ███████
  Away Win      39.9%  ███████████ ◄

  GOALS  (predicted total: 2.52)
  Over 0.5        92.0%  █████████████████████████
  Over 1.5        71.3%  ████████████████████
  Over 2.5        44.8%  ████████████  ← prediction sits here
  Over 3.5        23.5%  ██████
  Over 4.5        10.5%  ██
  BTTS (yes)      50.1%  █████████████

  TURKEY — Last 5 Games
    ...

  VALUE BET ANALYSIS
  ► Away Win
    Model prob   39.9%  vs  implied 38.5%
    Edge         +1.4%  at odds 2.60
    Stake        0.5% of bankroll  (¼ Kelly)
```

---

## Player Props

After the match analysis, `predict.py` prints a **menu of player-prop markets** for each team — the top 3 candidates in every market, so you can shop across them. A ★ flags any pick above 60%.

```
  PLAYER PROPS  (top picks per market — recent-form weighted; ★ = above 60%)
  expected goals: Argentina 3.25 / Haiti 1.81

  ARGENTINA
    To score         Martínez 31%   Messi 26%   Almada 20%
    To assist        Martínez 14%   González 14%   Paul 13%
    Goal or assist   Martínez 41%   Messi 35%   Almada 26%
    Shot on target   Paz 76%★   Martínez 75%★   Messi 74%★
    Fouls 1+         Barco 86%★   Capaldo 86%★   Giay 86%★
    To be booked     Barco 63%★   Capaldo 63%★   Martínez 37%
```

Markets evaluated: **to score**, **to assist**, **goal or assist**, **1+ / 2+ shots on target**, **1+ / 2+ fouls committed**, **to be booked**. The premium goal markets rarely clear 60% (even elite strikers sit ~30–50% anytime-scorer), so they're shown anyway with their % — the ★ just marks the near-locks. Tune the star line via `PROP_THRESHOLD` in `src/player_props.py`.

**Likely starters first (▶ / ▷):** likely starters are listed first in every market, regardless of probability, and get full-match weighting (their shot/goal/foul rates aren't discounted for rotation) — a confirmed starter is a far safer prop than a higher-rated bench player who might not play. Two confidence levels:

- **▶ WC matchweek-1 starter** — started their team's opening WC 2026 fixture (highest confidence).
- **▷ recent starter (fallback)** — started ≥2 of their last 3 internationals. Used when the team hasn't played its WC opener yet, so there's still a starter signal before matchweek 1.

Both are read from API-Football lineups by `fetch_player_stats.py`; re-run it after each matchday to keep the flags current (▷ players become ▶ once they start a WC game).

**How it works:** the match model predicts each team's expected goals, then that total is distributed across the squad by each player's recency-weighted goals-per-90 (plus a position baseline so attackers still get a share with little goal history). A Poisson distribution turns each player's rate into the probabilities above. Anytime-scorer rarely clears 60% even for elite strikers, so the cut mostly surfaces shots-on-target and card props — that is by design.

**Data source:** per-player stats come from **API-Football** — each nation's **5 most recent international games**, weighted so recent and higher-stakes matches count more:

```
weight = 0.85 ^ (games_ago)  ×  importance(competition)
```
where importance is World Cup 1.6, continental cups ~1.2, qualifiers ~1.1, friendlies 0.6. Captured per player: goals, shots, shots-on-target, assists, fouls, fouled, yellow/red cards.

**Note:** per-player xG is not available, so shot volume + conversion (shots/90, SoT/90, goals/shot) are used as the shooting-quality proxy. Debutant teams with no recent data show *"no props above 60%"*.

**Live xG sharpening:** the expected-goals split between the two teams blends each side's squad strength with its **actual WC 2026 xG** (`fetch_match_xg.py`). xG gains weight as more matches are played — a team's blend is one-third xG after one game and fully xG-driven after three. This makes both the goal-line and player-prop probabilities more accurate the deeper the tournament goes.

---

## Half Scoring (1st / 2nd half)

`predict.py` also shows each team's chance of scoring **in each half**:

```
  HALF SCORING  (chance the team scores in each half)
                      1st half  2nd half
  Brazil                  50%       56%
  Morocco                 43%       48%
```

Built from the minute of all 47,000+ historical international goals
(`goalscorers.csv`): each team's first-half vs second-half scoring split (shrunk
toward the league baseline of ~44% / ~56% for small samples) is applied to its
match expected-goals, then Poisson gives the per-half probabilities. Teams
reliably score more after the break — useful for "team to score in 2nd half" and
half-with-most-goals markets.

---

## Secrets / API keys

> **You must supply your own API keys.** This repo ships **no** keys — they are
> never committed. The two live-data tools (`live_ingest.py`, `fetch_player_stats.py`)
> will not work until you add yours.

### 1. Get your keys (both have free tiers)

| Key | Used by | Where to sign up | Free tier |
|-----|---------|------------------|-----------|
| `FOOTBALL_DATA_API_KEY` | `live_ingest.py` (live WC results) | https://www.football-data.org/client/register | 10 calls/min |
| `API_FOOTBALL_KEY` | `fetch_player_stats.py` (player stats) | https://www.api-football.com (or RapidAPI) | 100 calls/day* |

*Player props pull each team's last 5 games (~290 calls for all 48 teams), so the
free 100/day tier only covers a handful of teams per day — a paid plan fetches them
all at once. Results are cached in `.cache/` so re-runs don't re-spend calls.

### 2. Put them where they belong

```bash
cp .secrets.env.example .secrets.env     # create your private key file
```
Then open **`.secrets.env`** and paste your keys:
```ini
FOOTBALL_DATA_API_KEY=your_real_football_data_key
API_FOOTBALL_KEY=your_real_api_football_key
```

`.secrets.env` is in `.gitignore`, so it is **never pushed to GitHub**. The committed
`.secrets.env.example` is only a blank template. `src/config.py` loads the file at
runtime; you can instead `export` the same variables as real environment variables.
If a key is missing, the tool that needs it prints a clear setup message rather than
failing cryptically.

---

## Live Match Monitoring

To watch for a result as a match is happening and auto-ingest the final score:
```bash
python src/live_ingest.py --watch 90
```
Polls ESPN every 90 seconds. Once the match ends the result is patched into the dataset. Press `Ctrl+C` to stop.

To also retrain the moment a result comes in:
```bash
python src/live_ingest.py --watch 90 --retrain
```

---

## Team Names

You don't need to type exact names. The predictor handles common alternates and typos automatically:

| What you type | Resolves to |
|---------------|-------------|
| Turkiye / Türkiye | Turkey |
| USA / America | United States |
| Holland | Netherlands |
| Korea / Korea Republic | South Korea |
| Czechia | Czech Republic |
| Bosnia | Bosnia and Herzegovina |
| Congo DR / DRC | DR Congo |
| UAE / Emirates | United Arab Emirates |
| Ireland | Republic of Ireland |
| Curacao | Curaçao |
| Cote d'Ivoire | Ivory Coast |

**Typos** are also handled — if you type "Spaen" it will suggest Spain and let you confirm before running.

If a team genuinely can't be found, the prediction still runs but uses default Elo (1500) and no FIFA ranking, so accuracy will be lower.

---

## How the Betting Side Works

### What the model outputs
- **3 probabilities** — home win / draw / away win (always sum to 100%)
- **Predicted total goals** — used for over/under bets
- **Confidence** — the highest of the 3 probabilities

### What "value" means
A bet has value when the model's probability is **higher than what the odds imply**.

```
Bookmaker odds of 3.00  →  implied probability = 1 / 3.00 = 33.3%
Model says home win = 42%
Edge = 42% - 33.3% = +8.7%  ← value bet
```

Bookmakers also build in a 5–8% margin, so you need to overcome that — which is why the minimum edge threshold is set at **3%**.

### Stake sizing — quarter-Kelly
The output shows a recommended stake as a percentage of your total bankroll. This uses the Kelly Criterion divided by 4:

```
Full Kelly  = (prob × odds - 1) / (odds - 1)
Quarter Kelly = Full Kelly × 0.25
```

Quarter-Kelly is used because the model is not perfectly calibrated — full Kelly would overbet and risk large drawdown if the probabilities are even slightly off.

### Decision rules — only bet when ALL of these are true
1. **Edge > 3%** (model prob minus implied prob)
2. **Confidence >= 50%** (highest of the 3 outcome probs)
3. **Avoid standalone draw bets** — draws are the hardest class to predict
4. **Use the best available odds** — always shop between bookmakers to maximise edge

---

## Model Accuracy

| Metric | Value |
|--------|-------|
| Accuracy — all predictions | 60.7% |
| Accuracy — confidence >= 50% | **67.7%** (covers 70% of matches) |
| AUC (One-vs-Rest) | 0.734 |
| Goals MAE | 1.40 goals |
| Training data | 29,911 matches (1994–2026) |

### Why not higher?

3-class football prediction has a real ceiling regardless of model complexity:

- **Draws are nearly unpredictable** — the draw class achieves ~24% accuracy on its own and drags the overall figure down. No model predicts draws consistently because they're often decided by noise (a deflection, a missed penalty) rather than team quality.
- **Bookmakers employ large analyst teams** and still only get ~55–58% of outcomes right on public data.
- **Optuna confirmed this** — tuning 80 combinations of XGBoost + LightGBM hyperparameters produced the same result as the defaults. The bottleneck is features, not the model.

**67.7% on confident predictions is the number that matters for betting** — that's matches where the model has ≥50% confidence, covering 70% of all fixtures. On those, the model's probabilities are well-calibrated enough to find genuine value against bookmaker lines.

---

## Data Sources

| Data | Source | Coverage |
|------|--------|----------|
| Match results (49k+ matches) | martj42/international_results on GitHub | 1872–present, updated live |
| WC 2014 / 2018 / 2022 odds + xG | football-data.co.uk | Historical only |
| FIFA world rankings | samuraitruong/fifa-ranking-data on GitHub | 2003–present |
| WC 2026 live scores | ESPN API — no API key needed | Live |

---

## File Structure

```
WorldCupPredictor/
├── src/
│   ├── features.py      — Elo ratings, rolling form, H2H, FIFA rank features
│   ├── train.py         — Model training with Optuna hyperparameter tuning
│   ├── predict.py       — Match predictor + value bet calculator
│   └── live_ingest.py   — Live ESPN result fetcher + dataset patcher
├── data/
│   └── raw/
│       └── international_results.csv   — master match dataset (auto-updated)
└── models/
    ├── outcome_model.pkl   — XGBoost + LightGBM stacking ensemble
    ├── goals_model.pkl     — gradient boosting goals regressor
    └── elo_ratings.pkl     — per-team Elo snapshot for fast inference
```
