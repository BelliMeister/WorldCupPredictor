"""
Feature engineering for match prediction — v2.
Adds FIFA ranking features, xG rolling form, and match importance weighting.
All functions are pure — they return new DataFrames, never mutate inputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path

FIFA_RANKINGS_PATH = Path(__file__).parent.parent.parent / "world cup match odds" / "data" / "processed" / "fifa_rankings.csv"

ELO_K_BASE = 40
ELO_DEFAULT = 1500
ELO_HOME_ADVANTAGE = 60  # applied when not neutral

TOURNAMENT_K = {
    "FIFA World Cup": 1.5,
    "UEFA Euro": 1.3,
    "Copa America": 1.3,
    "AFC Asian Cup": 1.2,
    "Africa Cup of Nations": 1.2,
    "FIFA World Cup qualification": 1.1,
    "UEFA Euro qualification": 1.0,
    "Friendly": 0.6,
}

WORLD_CUP_TOURNAMENTS = {"FIFA World Cup", "FIFA World Cup qualification", "Confederations Cup"}


# ── FIFA rankings lookup ───────────────────────────────────────────────────────

def load_fifa_rankings() -> pd.DataFrame | None:
    if not FIFA_RANKINGS_PATH.exists():
        return None
    df = pd.read_csv(FIFA_RANKINGS_PATH, parse_dates=["ranking_date"])
    return df.sort_values(["team", "ranking_date"]).reset_index(drop=True)


def build_rank_lookup(rankings: pd.DataFrame) -> dict[str, list]:
    """Pre-index rankings by team for fast date lookups."""
    lookup = {}
    for team, grp in rankings.groupby("team"):
        lookup[team] = {
            "dates": grp["ranking_date"].values,
            "ranks": grp["fifa_rank"].values,
            "pts": grp["fifa_points"].values,
        }
    return lookup


def get_rank_at(lookup: dict, team: str, match_date: pd.Timestamp) -> tuple[float, float]:
    if team not in lookup:
        return np.nan, np.nan
    entry = lookup[team]
    mask = entry["dates"] <= match_date.to_datetime64()
    if not mask.any():
        return np.nan, np.nan
    idx = mask.sum() - 1
    return float(entry["ranks"][idx]), float(entry["pts"][idx])


# ── Elo ratings ────────────────────────────────────────────────────────────────

def _elo_expected(r_a: float, r_b: float) -> float:
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def _elo_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0


def build_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    home_elo_before, away_elo_before = [], []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        r_home = ratings.get(home, ELO_DEFAULT)
        r_away = ratings.get(away, ELO_DEFAULT)

        home_elo_before.append(r_home)
        away_elo_before.append(r_away)

        t = row["tournament"]
        k = ELO_K_BASE * TOURNAMENT_K.get(t, 0.8)

        # Home advantage in Elo space for non-neutral venues
        r_home_adj = r_home + (0 if row["neutral"] else ELO_HOME_ADVANTAGE)

        score_a = _elo_score(row["home_score"], row["away_score"])
        exp_a = _elo_expected(r_home_adj, r_away)

        delta = k * (score_a - exp_a)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    return df.assign(
        home_elo_before=home_elo_before,
        away_elo_before=away_elo_before,
        elo_diff=lambda d: d["home_elo_before"] - d["away_elo_before"],
    )


# ── Rolling form ───────────────────────────────────────────────────────────────

def rolling_form(df: pd.DataFrame, n_games: int = 7) -> pd.DataFrame:
    """Last n games rolling averages per team (no leakage)."""
    records = df.to_dict("records")
    team_hist: dict[str, list] = {}

    home_cols: dict[str, list] = {"home_avg_scored": [], "home_avg_conceded": [], "home_win_rate": [], "home_points_rate": []}
    away_cols: dict[str, list] = {"away_avg_scored": [], "away_avg_conceded": [], "away_win_rate": [], "away_points_rate": []}

    for row in records:
        for team, cols, prefix in [(row["home_team"], home_cols, "home_"), (row["away_team"], away_cols, "away_")]:
            hist = team_hist.get(team, [])[-n_games:]
            if hist:
                cols[f"{prefix}avg_scored"].append(np.mean([h["scored"] for h in hist]))
                cols[f"{prefix}avg_conceded"].append(np.mean([h["conceded"] for h in hist]))
                cols[f"{prefix}win_rate"].append(np.mean([h["win"] for h in hist]))
                cols[f"{prefix}points_rate"].append(np.mean([h["pts"] for h in hist]))
            else:
                cols[f"{prefix}avg_scored"].append(np.nan)
                cols[f"{prefix}avg_conceded"].append(np.nan)
                cols[f"{prefix}win_rate"].append(np.nan)
                cols[f"{prefix}points_rate"].append(np.nan)

        def record(scored, conceded):
            if scored > conceded:
                return {"scored": scored, "conceded": conceded, "win": 1, "pts": 3}
            if scored == conceded:
                return {"scored": scored, "conceded": conceded, "win": 0, "pts": 1}
            return {"scored": scored, "conceded": conceded, "win": 0, "pts": 0}

        team_hist.setdefault(row["home_team"], []).append(record(row["home_score"], row["away_score"]))
        team_hist.setdefault(row["away_team"], []).append(record(row["away_score"], row["home_score"]))

    return df.assign(**home_cols, **away_cols)


# ── Head-to-head ───────────────────────────────────────────────────────────────

def head_to_head(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    records = df.to_dict("records")
    pair_hist: dict[tuple, list] = {}
    h2h_home_win_rate, h2h_avg_goals = [], []

    for row in records:
        home, away = row["home_team"], row["away_team"]
        key = tuple(sorted([home, away]))
        hist = pair_hist.get(key, [])[-n:]

        if hist:
            hw = sum(
                1 for h in hist
                if (h["home"] == home and h["result"] == "home_win")
                or (h["away"] == home and h["result"] == "away_win")
            )
            h2h_home_win_rate.append(hw / len(hist))
            h2h_avg_goals.append(np.mean([h["home_score"] + h["away_score"] for h in hist]))
        else:
            h2h_home_win_rate.append(0.5)
            h2h_avg_goals.append(2.5)

        if row["home_score"] > row["away_score"]:
            result = "home_win"
        elif row["home_score"] == row["away_score"]:
            result = "draw"
        else:
            result = "away_win"

        pair_hist.setdefault(key, []).append({
            "home": home, "away": away,
            "home_score": row["home_score"], "away_score": row["away_score"],
            "result": result,
        })

    return df.assign(h2h_home_win_rate=h2h_home_win_rate, h2h_avg_goals=h2h_avg_goals)


# ── FIFA ranking features ──────────────────────────────────────────────────────

def attach_fifa_rankings(df: pd.DataFrame) -> pd.DataFrame:
    rankings = load_fifa_rankings()
    if rankings is None:
        return df.assign(
            home_fifa_rank=np.nan, away_fifa_rank=np.nan,
            home_fifa_pts=np.nan, away_fifa_pts=np.nan,
            rank_diff=np.nan, pts_diff=np.nan,
        )

    lookup = build_rank_lookup(rankings)
    home_ranks, home_pts, away_ranks, away_pts = [], [], [], []

    for _, row in df.iterrows():
        hr, hp = get_rank_at(lookup, row["home_team"], row["date"])
        ar, ap = get_rank_at(lookup, row["away_team"], row["date"])
        home_ranks.append(hr)
        home_pts.append(hp)
        away_ranks.append(ar)
        away_pts.append(ap)

    return df.assign(
        home_fifa_rank=home_ranks,
        home_fifa_pts=home_pts,
        away_fifa_rank=away_ranks,
        away_fifa_pts=away_pts,
        rank_diff=lambda d: d["away_fifa_rank"] - d["home_fifa_rank"],
        pts_diff=lambda d: d["home_fifa_pts"] - d["away_fifa_pts"],
    )


# ── Label helpers ──────────────────────────────────────────────────────────────

def label_outcome(home: int, away: int) -> int:
    """0=home win, 1=draw, 2=away win."""
    if home > away:
        return 0
    if home == away:
        return 1
    return 2


# ── Tournament importance weight ───────────────────────────────────────────────

def match_weight(tournament: str) -> float:
    return TOURNAMENT_K.get(tournament, 0.8)


# ── Master pipeline ────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, attach_rankings: bool = True) -> pd.DataFrame:
    df = build_elo_ratings(df)
    df = rolling_form(df)
    df = head_to_head(df)
    if attach_rankings:
        print("  Attaching FIFA rankings (slow — only runs on first call)...")
        df = attach_fifa_rankings(df)
    return df.assign(
        outcome=df.apply(lambda r: label_outcome(r["home_score"], r["away_score"]), axis=1),
        total_goals=df["home_score"] + df["away_score"],
        is_neutral=df["neutral"].astype(int),
        is_world_cup=df["tournament"].isin(WORLD_CUP_TOURNAMENTS).astype(int),
        match_weight=df["tournament"].map(match_weight).fillna(0.8),
    )


FEATURE_COLS = [
    # Elo
    "elo_diff",
    "home_elo_before",
    "away_elo_before",
    # Form
    "home_avg_scored",
    "home_avg_conceded",
    "home_win_rate",
    "home_points_rate",
    "away_avg_scored",
    "away_avg_conceded",
    "away_win_rate",
    "away_points_rate",
    # H2H
    "h2h_home_win_rate",
    "h2h_avg_goals",
    # FIFA rank
    "rank_diff",
    "pts_diff",
    "home_fifa_rank",
    "away_fifa_rank",
    # Context
    "is_neutral",
    "is_world_cup",
]
