"""
Feature engineering for match prediction — v2.
Adds FIFA ranking features, xG rolling form, and match importance weighting.
All functions are pure — they return new DataFrames, never mutate inputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path

FIFA_RANKINGS_PATH = Path(__file__).parent.parent.parent / "world cup match odds" / "data" / "processed" / "fifa_rankings.csv"
SQUAD_VALUES_PATH  = Path(__file__).parent.parent / "data" / "raw" / "squad_values.csv"
MASTER_CSV_PATH    = Path(__file__).parent.parent.parent / "world cup match odds" / "data" / "processed" / "master.csv"

RECENCY_DECAY = 0.85  # exponential decay weight per game back in rolling form

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


def _goal_diff_multiplier(home_goals: int, away_goals: int) -> float:
    """
    World Football Elo goal-difference weight: a bigger winning margin moves the
    rating more. |diff|<=1 → 1.0, =2 → 1.5, >=3 → (11+diff)/8.
    """
    diff = abs(home_goals - away_goals)
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11 + diff) / 8.0


ELO_MOMENTUM_WINDOW = 5  # matches over which to measure Elo trend


def build_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    history: dict[str, list] = {}  # team -> list of elo_before snapshots
    home_elo_before, away_elo_before = [], []
    home_momentum, away_momentum = [], []

    def momentum(team: str, current: float) -> float:
        hist = history.get(team, [])
        if not hist:
            return 0.0
        past = hist[-ELO_MOMENTUM_WINDOW] if len(hist) >= ELO_MOMENTUM_WINDOW else hist[0]
        return current - past

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        r_home = ratings.get(home, ELO_DEFAULT)
        r_away = ratings.get(away, ELO_DEFAULT)

        home_elo_before.append(r_home)
        away_elo_before.append(r_away)
        home_momentum.append(momentum(home, r_home))
        away_momentum.append(momentum(away, r_away))
        history.setdefault(home, []).append(r_home)
        history.setdefault(away, []).append(r_away)

        t = row["tournament"]
        k = ELO_K_BASE * TOURNAMENT_K.get(t, 0.8)
        # Scale the update by winning margin (draws/1-goal games unchanged)
        k *= _goal_diff_multiplier(row["home_score"], row["away_score"])

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
        home_elo_momentum=home_momentum,
        away_elo_momentum=away_momentum,
        elo_diff=lambda d: d["home_elo_before"] - d["away_elo_before"],
        elo_momentum_diff=lambda d: d["home_elo_momentum"] - d["away_elo_momentum"],
    )


# ── Rolling form ───────────────────────────────────────────────────────────────

def _decay_mean(values: list[float], decay: float = RECENCY_DECAY) -> float:
    """Exponentially weighted mean — most recent game has weight 1, each prior game * decay."""
    if not values:
        return np.nan
    weights = np.array([decay ** i for i in range(len(values) - 1, -1, -1)])
    return float(np.dot(weights, values) / weights.sum())


def rolling_form(df: pd.DataFrame, n_games: int = 7) -> pd.DataFrame:
    """Last n games recency-weighted averages per team (no leakage)."""
    records = df.to_dict("records")
    team_hist: dict[str, list] = {}

    home_cols: dict[str, list] = {"home_avg_scored": [], "home_avg_conceded": [], "home_win_rate": [], "home_points_rate": []}
    away_cols: dict[str, list] = {"away_avg_scored": [], "away_avg_conceded": [], "away_win_rate": [], "away_points_rate": []}

    for row in records:
        for team, cols, prefix in [(row["home_team"], home_cols, "home_"), (row["away_team"], away_cols, "away_")]:
            hist = team_hist.get(team, [])[-n_games:]
            if hist:
                cols[f"{prefix}avg_scored"].append(_decay_mean([h["scored"] for h in hist]))
                cols[f"{prefix}avg_conceded"].append(_decay_mean([h["conceded"] for h in hist]))
                cols[f"{prefix}win_rate"].append(_decay_mean([h["win"] for h in hist]))
                cols[f"{prefix}points_rate"].append(_decay_mean([h["pts"] for h in hist]))
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


# ── xG rolling form ───────────────────────────────────────────────────────────

def attach_xg_form(df: pd.DataFrame, n_games: int = 7) -> pd.DataFrame:
    """
    Attach per-team recency-weighted rolling xG averages from master.csv.
    Only WC/qualifier matches have xG data, so most training rows will be NaN —
    the model learns to treat NaN as 'no signal' via median imputation in the pipeline.
    """
    if not MASTER_CSV_PATH.exists():
        return df.assign(home_avg_xg=np.nan, away_avg_xg=np.nan)

    xg_df = pd.read_csv(MASTER_CSV_PATH, parse_dates=["date"],
                         usecols=["date", "home_team", "away_team", "home_xg", "away_xg"])
    xg_df = xg_df.dropna(subset=["home_xg", "away_xg"]).sort_values("date").reset_index(drop=True)

    # Build per-team xG history keyed by (team, date)
    team_xg_hist: dict[str, list[tuple]] = {}  # team -> [(date, xg_for)]
    for _, row in xg_df.iterrows():
        team_xg_hist.setdefault(row["home_team"], []).append((row["date"], row["home_xg"]))
        team_xg_hist.setdefault(row["away_team"], []).append((row["date"], row["away_xg"]))

    def lookup_xg(team: str, before_date: pd.Timestamp) -> float:
        hist = team_xg_hist.get(team, [])
        past = [xg for d, xg in hist if d < before_date][-n_games:]
        return _decay_mean(past) if past else np.nan

    home_xg, away_xg = [], []
    for _, row in df.iterrows():
        home_xg.append(lookup_xg(row["home_team"], row["date"]))
        away_xg.append(lookup_xg(row["away_team"], row["date"]))

    return df.assign(home_avg_xg=home_xg, away_avg_xg=away_xg)


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


# ── Rest & congestion ──────────────────────────────────────────────────────────

REST_CAP_DAYS = 180   # cap long gaps so a 2-year absence isn't an outlier
CONGESTION_WINDOW_DAYS = 30


def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Days of rest since each team's last match and matches played in the last 30 days."""
    last_date: dict[str, pd.Timestamp] = {}
    recent: dict[str, list] = {}
    home_rest, away_rest, home_cong, away_cong = [], [], [], []

    for row in df.itertuples():
        d = row.date
        for team, rest_list, cong_list in [
            (row.home_team, home_rest, home_cong),
            (row.away_team, away_rest, away_cong),
        ]:
            ld = last_date.get(team)
            rest_list.append(min((d - ld).days, REST_CAP_DAYS) if ld is not None else np.nan)
            window = [x for x in recent.get(team, []) if (d - x).days <= CONGESTION_WINDOW_DAYS]
            cong_list.append(len(window))
            recent[team] = window + [d]
            last_date[team] = d

    return df.assign(
        home_rest_days=home_rest,
        away_rest_days=away_rest,
        rest_diff=lambda x: x["home_rest_days"] - x["away_rest_days"],
        home_matches_30d=home_cong,
        away_matches_30d=away_cong,
    )


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


# ── Squad market values ────────────────────────────────────────────────────────

def attach_squad_values(df: pd.DataFrame) -> pd.DataFrame:
    if not SQUAD_VALUES_PATH.exists():
        return df.assign(home_squad_value=np.nan, away_squad_value=np.nan, squad_value_ratio=np.nan)

    sv = pd.read_csv(SQUAD_VALUES_PATH).set_index("team")["squad_value_eur_m"].to_dict()
    DEFAULT_VALUE = float(np.nanmedian(list(sv.values())))

    home_vals = df["home_team"].map(sv).fillna(DEFAULT_VALUE)
    away_vals = df["away_team"].map(sv).fillna(DEFAULT_VALUE)
    ratio = (home_vals / away_vals.replace(0, np.nan)).fillna(1.0)

    return df.assign(
        home_squad_value=home_vals,
        away_squad_value=away_vals,
        squad_value_ratio=ratio,
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
    df = attach_xg_form(df)
    df = head_to_head(df)
    df = temporal_features(df)
    if attach_rankings:
        print("  Attaching FIFA rankings (slow — only runs on first call)...")
        df = attach_fifa_rankings(df)
    df = attach_squad_values(df)
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
    # Elo momentum (form trend)
    "elo_momentum_diff",
    "home_elo_momentum",
    "away_elo_momentum",
    # Rest & congestion
    "rest_diff",
    "home_matches_30d",
    "away_matches_30d",
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
    # Squad market value (Transfermarkt)
    "home_squad_value",
    "away_squad_value",
    "squad_value_ratio",
    # xG rolling form (WC/qualifier matches only — NaN elsewhere)
    "home_avg_xg",
    "away_avg_xg",
]
