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

**First-time training** (builds models from 30,000+ historical matches)
```bash
python src/train.py          # full Optuna tuning — ~8 min
python src/train.py --quick  # default params — ~3 min
```

Models are saved to `models/` and reused by `predict.py`.

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
