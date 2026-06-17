"""
Value-betting backtest — does the model actually beat the bookmaker?

Trains on all but the most recent time fold, predicts that held-out fold
(strictly out of sample), joins the predictions to real bookmaker odds in
master.csv, and settles value bets to report ROI. This is the money metric —
accuracy is secondary to whether the edges are real.

Run: python src/backtest.py [--edge 0.05] [--threshold 0.0]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

from features import FEATURE_COLS, engineer_features
from train import build_default_model, DATA_PATH, CUTOFF_YEAR

MASTER_CSV = Path(__file__).parent.parent.parent / "world cup match odds" / "data" / "processed" / "master.csv"

# master.csv team spelling → international_results spelling
NAME_FIX = {
    "USA": "United States", "Korea Republic": "South Korea", "China PR": "China",
    "Czechia": "Czech Republic", "IR Iran": "Iran",
}


def _norm(s: pd.Series) -> pd.Series:
    return s.replace(NAME_FIX).str.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=0.05, help="Min model edge (prob*odds-1) to bet")
    ap.add_argument("--threshold", type=float, default=0.0, help="Min model prob to consider")
    ap.add_argument("--kelly-frac", type=float, default=0.25, help="Kelly fraction for staking")
    args = ap.parse_args()

    print("Loading + engineering features...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df[df["date"].dt.year >= CUTOFF_YEAR].dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    df = engineer_features(df, attach_rankings=True)

    X = df[FEATURE_COLS].values
    y = df["outcome"].values
    folds = list(TimeSeriesSplit(n_splits=5).split(X))
    tr, te = folds[-1]

    print(f"Training on {len(tr):,} matches, predicting {len(te):,} held-out...")
    model = build_default_model()
    model.fit(X[tr], y[tr])
    probs = model.predict_proba(X[te])

    test = df.iloc[te][["date", "home_team", "away_team", "outcome"]].copy()
    test["pH"], test["pD"], test["pA"] = probs[:, 0], probs[:, 1], probs[:, 2]
    test["home_team"] = _norm(test["home_team"]); test["away_team"] = _norm(test["away_team"])

    # Odds
    m = pd.read_csv(MASTER_CSV, parse_dates=["date"])
    for c in ["odds_home_avg", "odds_draw_avg", "odds_away_avg"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["odds_home_avg", "odds_draw_avg", "odds_away_avg"])
    m["home_team"] = _norm(m["home_team"]); m["away_team"] = _norm(m["away_team"])
    m["date"] = m["date"].dt.normalize()
    test["date"] = test["date"].dt.normalize()

    j = test.merge(m[["date", "home_team", "away_team", "odds_home_avg", "odds_draw_avg", "odds_away_avg"]],
                   on=["date", "home_team", "away_team"], how="inner")
    print(f"Held-out matches with bookmaker odds: {len(j)}")
    if len(j) < 30:
        print("Too few overlapping matches with odds to backtest reliably.")
        return

    probs_arr = j[["pH", "pD", "pA"]].values
    odds_arr  = j[["odds_home_avg", "odds_draw_avg", "odds_away_avg"]].values
    outcomes  = j["outcome"].values

    flat_pl = kelly_pl = flat_staked = kelly_staked = 0.0
    n_bets = wins = 0
    for p, o, res in zip(probs_arr, odds_arr, outcomes):
        for k in range(3):
            if p[k] < args.threshold:
                continue
            edge = p[k] * o[k] - 1.0
            if edge <= args.edge:
                continue
            n_bets += 1
            won = (res == k)
            wins += won
            flat_staked += 1.0
            flat_pl += (o[k] - 1.0) if won else -1.0
            kf = max(0.0, (p[k] * o[k] - 1) / (o[k] - 1)) * args.kelly_frac
            kelly_staked += kf
            kelly_pl += kf * (o[k] - 1.0) if won else -kf

    # Baseline: back the bookmaker favourite flat on every match
    fav = odds_arr.argmin(axis=1)
    fav_win = (fav == outcomes)
    fav_pl = np.where(fav_win, odds_arr[np.arange(len(j)), fav] - 1, -1).sum()

    print("\n=== VALUE BACKTEST (out-of-sample) ===")
    print(f"  edge>{args.edge:.0%}  prob>{args.threshold:.0%}")
    print(f"  Value bets placed   : {n_bets}")
    if n_bets:
        print(f"  Hit rate            : {wins/n_bets:.1%}")
        print(f"  Flat-stake ROI      : {flat_pl/flat_staked:+.1%}  (P/L {flat_pl:+.1f}u on {flat_staked:.0f}u)")
        if kelly_staked > 0:
            print(f"  ¼-Kelly ROI         : {kelly_pl/kelly_staked:+.1%}  (P/L {kelly_pl:+.2f}u on {kelly_staked:.1f}u)")
    print(f"\n  Baseline (back favourite every game):")
    print(f"  Favourite hit rate  : {fav_win.mean():.1%}")
    print(f"  Favourite flat ROI  : {fav_pl/len(j):+.1%}")


if __name__ == "__main__":
    main()
