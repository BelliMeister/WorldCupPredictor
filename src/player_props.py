"""
Player prop model — turns a team's expected goals into per-player betting props.

Approach (standard bookmaker decomposition):
  1. The match model predicts each team's expected goals (team_mu).
  2. Distribute team_mu across the squad weighted by each player's
     recency-weighted goals-per-90 × expected share of minutes.
  3. Poisson on the resulting per-player rate gives:
       - anytime goalscorer probability
       - assist probability (team assists ≈ 0.75 × team goals)
       - shots / shots-on-target over-under lines
       - to-be-booked probability

Reads:  data/raw/player_stats.csv   (built by fetch_player_stats.py)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

PLAYER_STATS_CSV = Path("data/raw/player_stats.csv")
TEAM_STATS_CSV   = Path("data/raw/team_stats.csv")
MATCH_XG_CSV     = Path("data/raw/wc2026_xg.csv")

# Opponent-defence adjustment is clamped so a small/noisy sample can't swing a
# prop too far (a leaky/stingy opponent moves the rate at most ±40%).
DEF_FACTOR_MIN = 0.65
DEF_FACTOR_MAX = 1.40

# Recency decay for the live WC xG signal (most recent match weighted highest)
XG_DECAY = 0.85

# Fraction of a goal that is assisted (rest are solo / rebound / own goals)
ASSIST_RATE = 0.75

# Empirical-Bayes shrinkage: pseudo-90s added to the denominator so a player
# with one hot game doesn't get a runaway per-90 rate. ~3 full matches of prior.
PRIOR_90S = 3.0

# Position-based baseline scoring/assist propensity. Spreads the team's expected
# goals across attackers even when a squad has little/no goal history yet (early
# in the tournament), instead of dumping it all on the one player who has scored.
POS_GOAL_PRIOR   = {"FW": 0.35, "MF": 0.12, "DF": 0.04, "GK": 0.0}
POS_ASSIST_PRIOR = {"FW": 0.12, "MF": 0.18, "DF": 0.08, "GK": 0.0}


def _pos_prior(position, table: dict) -> float:
    """First listed position drives the baseline (e.g. 'FW,MF' → FW)."""
    if not isinstance(position, str) or not position:
        return table["MF"]
    first = position.replace(",", " ").split()[0].upper()
    return table.get(first, table["MF"])

# API-Football team spelling → international_results.csv spelling
TEAM_NAME_MAP = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands":   "Cape Verde",
    "Congo DR":             "DR Congo",
    "Czechia":              "Czech Republic",
    "Türkiye":              "Turkey",
    "USA":                  "United States",
    # FBref fallbacks (in case an older dataset is present)
    "Bosnia-Herzegovina":   "Bosnia and Herzegovina",
    "Côte d'Ivoire":        "Ivory Coast",
    "IR Iran":              "Iran",
    "Korea Republic":       "South Korea",
}


def _poisson_ge1(lam: float) -> float:
    """P(X >= 1) for X ~ Poisson(lam)."""
    return 0.0 if lam <= 0 else 1.0 - math.exp(-lam)


def _poisson_ge2(lam: float) -> float:
    """P(X >= 2) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 0.0
    return 1.0 - math.exp(-lam) - lam * math.exp(-lam)


def load_player_stats() -> pd.DataFrame | None:
    if not PLAYER_STATS_CSV.exists():
        return None
    df = pd.read_csv(PLAYER_STATS_CSV)
    df["team"] = df["team"].replace(TEAM_NAME_MAP)
    return df


def load_team_stats() -> pd.DataFrame | None:
    if not TEAM_STATS_CSV.exists():
        return None
    df = pd.read_csv(TEAM_STATS_CSV)
    df["team"] = df["team"].replace(TEAM_NAME_MAP)
    return df


def defensive_factors(team_stats: pd.DataFrame | None, opponent: str) -> dict[str, float]:
    """
    How leaky/stingy `opponent` is vs league average, split into:
      - shots   : shot VOLUME conceded (a compact side still allows many shots)
      - quality : SoT% the opponent allows vs league — a packed box forces
                  low-quality attempts, so this dampens on-target/goal props even
                  when shot volume is high (the 'ten men behind the ball' effect)
      - fouls   : fouls the opponent draws → fouls our players commit
    Each is a multiplier around 1.0, clamped ±40%. Neutral (1.0) when unknown.
    """
    neutral = {"shots": 1.0, "quality": 1.0, "fouls": 1.0}
    if team_stats is None or "shots_against" not in team_stats.columns:
        return neutral
    row = team_stats[team_stats["team"] == opponent]
    if row.empty:
        return neutral
    r = row.iloc[0]

    def factor(against_col: str) -> float:
        league = team_stats[against_col].mean()
        if not league or pd.isna(r.get(against_col)):
            return 1.0
        return float(np.clip(r[against_col] / league, DEF_FACTOR_MIN, DEF_FACTOR_MAX))

    # Shot quality conceded: opponent's SoT% allowed vs the league's SoT%.
    shots_a = r.get("shots_against")
    sot_a   = r.get("sot_against")
    lg_q = team_stats["sot_against"].mean() / team_stats["shots_against"].mean()
    if shots_a and not pd.isna(sot_a) and lg_q:
        quality = float(np.clip((sot_a / shots_a) / lg_q, DEF_FACTOR_MIN, DEF_FACTOR_MAX))
    else:
        quality = 1.0

    return {"shots": factor("shots_against"), "quality": quality,
            "fouls": factor("fouls_against")}


def _start_fraction(row) -> float:
    """Estimate the share of a full match a player typically plays (caps sub noise)."""
    matches = row.get("matches", 0) or 0
    minutes = row.get("minutes", 0) or 0
    if matches > 0 and minutes > 0:
        return float(min(1.0, minutes / (matches * 90.0)))
    return 0.6  # unknown — assume rotation player


def team_props(stats: pd.DataFrame, team: str, team_mu: float, n_top: int = 6,
               opp_factors: dict[str, float] | None = None) -> list[dict]:
    """
    Compute per-player props for one team given its expected goals (team_mu).
    `opp_factors` scales the shot/SoT/foul props by the opponent's defensive
    leakiness (see defensive_factors). Returns the n_top players by score prob.
    """
    f = opp_factors or {"shots": 1.0, "quality": 1.0, "fouls": 1.0}
    squad = stats[stats["team"] == team].copy()
    if squad.empty:
        return []

    # Prefer the current (2026) squad; fall back to historical only if the team
    # has not appeared in the 2026 edition yet.
    if "active_2026" in squad.columns and squad["active_2026"].any():
        squad = squad[squad["active_2026"]].copy()

    for col in ("started_first_wc", "recent_starter"):
        if col not in squad.columns:
            squad[col] = False
        squad[col] = squad[col].fillna(False).astype(bool)

    # Likely starter = started the WC opener (primary) OR started ≥2 of last 3
    # games (fallback, for teams yet to play their WC opener).
    squad["likely_starter"] = squad["started_first_wc"] | squad["recent_starter"]

    # Known/likely starters are given full-match weighting instead of the
    # historical minutes estimate (which discounts for rotation).
    squad["start_frac"] = squad.apply(
        lambda r: max(_start_fraction(r), 0.95) if r["likely_starter"] else _start_fraction(r),
        axis=1,
    )

    # Shrunk per-match rate = pooled goals / (90s played + prior). Pulls small
    # samples (1–2 games) toward zero so they can't dominate the goal share.
    # A position baseline is added so attackers still get a share with no goals yet.
    nineties = (squad["minutes"].clip(lower=0) / 90.0) + PRIOR_90S
    goal_pos   = squad["position"].apply(lambda p: _pos_prior(p, POS_GOAL_PRIOR))
    assist_pos = squad["position"].apply(lambda p: _pos_prior(p, POS_ASSIST_PRIOR))
    squad["goal_weight"]   = (squad["goals"].clip(lower=0)   / nineties + goal_pos)   * squad["start_frac"]
    squad["assist_weight"] = (squad["assists"].clip(lower=0) / nineties + assist_pos) * squad["start_frac"]

    g_total = squad["goal_weight"].sum()
    a_total = squad["assist_weight"].sum()
    team_assist_mu = team_mu * ASSIST_RATE

    props = []
    for _, r in squad.iterrows():
        goal_lambda   = team_mu * (r["goal_weight"] / g_total) if g_total > 0 else 0.0
        assist_lambda = team_assist_mu * (r["assist_weight"] / a_total) if a_total > 0 else 0.0

        # Shots: scaled by opponent shot VOLUME conceded.
        exp_shots = r["shots_per90"] * r["start_frac"] * f["shots"]
        # SoT: derive from expected shots × the player's own on-target accuracy ×
        # the opponent's shot-quality factor (compact defences force worse shots),
        # so a leaky-but-compact side boosts shots far more than shots-on-target.
        accuracy = r["sot_per90"] / r["shots_per90"] if r["shots_per90"] > 0 else 0.34
        exp_sot   = exp_shots * min(max(accuracy, 0.05), 0.6) * f["quality"]
        exp_fouls = r["fouls_per90"] * r["start_frac"] * f["fouls"]
        card_lam  = r["yellow_per90"] * r["start_frac"]

        props.append({
            "player":        r["player"],
            "position":      r.get("position", ""),
            "started_first_wc": bool(r["started_first_wc"]),
            "recent_starter":   bool(r["recent_starter"]),
            "likely_starter":   bool(r["likely_starter"]),
            "p_score":       _poisson_ge1(goal_lambda),
            "p_assist":      _poisson_ge1(assist_lambda),
            "p_goal_or_ast": _poisson_ge1(goal_lambda + assist_lambda),
            "exp_shots":     exp_shots,
            "p_shots_1plus": _poisson_ge1(exp_shots),
            "p_shots_2plus": _poisson_ge2(exp_shots),
            "exp_sot":       exp_sot,
            "p_sot_1plus":   _poisson_ge1(exp_sot),
            "p_sot_2plus":   _poisson_ge2(exp_sot),
            "exp_fouls":     exp_fouls,
            "p_foul_1plus":  _poisson_ge1(exp_fouls),
            "p_foul_2plus":  _poisson_ge2(exp_fouls),
            "p_card":        _poisson_ge1(card_lam),
        })

    props.sort(key=lambda p: p["p_score"], reverse=True)
    return props[:n_top]


# Only surface a player prop when its probability clears this bar
PROP_THRESHOLD = 0.60

# Human labels for each prop key
PROP_LABELS = {
    "p_score":       "to score",
    "p_assist":      "to assist",
    "p_goal_or_ast": "goal or assist",
    "p_shots_1plus": "1+ shots taken",
    "p_shots_2plus": "2+ shots taken",
    "p_sot_1plus":   "1+ shot on target",
    "p_sot_2plus":   "2+ shots on target",
    "p_foul_1plus":  "1+ fouls committed",
    "p_foul_2plus":  "2+ fouls committed",
    "p_card":        "to be booked",
}


# Markets shown as grouped sections (label, prop key), in display order
PROP_MARKETS = [
    ("To score",        "p_score"),
    ("To assist",       "p_assist"),
    ("Goal or assist",  "p_goal_or_ast"),
    ("Shots taken 1+",  "p_shots_1plus"),
    ("Shots taken 2+",  "p_shots_2plus"),
    ("Shot on target",  "p_sot_1plus"),
    ("2+ shots on tgt", "p_sot_2plus"),
    ("Fouls 1+",        "p_foul_1plus"),
    ("Fouls 2+",        "p_foul_2plus"),
    ("To be booked",    "p_card"),
]


def props_by_market(stats: pd.DataFrame, team: str, team_mu: float,
                    top_n: int = 3,
                    opp_factors: dict[str, float] | None = None) -> list[tuple[str, list[dict]]]:
    """
    Group props by betting market. Returns [(market_label, [top players]), ...]
    where each player is {player, position, prob}. Shows the best candidates in
    every market (so scorer/assist always appear, even below the 60% line).
    `opp_factors` adjusts shot/SoT/foul props for the opponent's defence.
    """
    everyone = team_props(stats, team, team_mu, n_top=99, opp_factors=opp_factors)
    out = []
    for label, key in PROP_MARKETS:
        # Likely starters rank first, then by probability — a known/likely
        # starter is a safer prop than a higher-rated bench player.
        ranked = sorted(
            everyone,
            key=lambda p: (p.get("likely_starter", False), p[key]),
            reverse=True,
        )[:top_n]
        rows = [{"player": p["player"], "position": p["position"], "prob": p[key],
                 "started_first_wc": p.get("started_first_wc", False),
                 "recent_starter":   p.get("recent_starter", False),
                 "likely_starter":   p.get("likely_starter", False)}
                for p in ranked if p[key] > 0]
        out.append((label, rows))
    return out


def props_above_threshold(stats: pd.DataFrame, team: str, team_mu: float,
                          threshold: float = PROP_THRESHOLD) -> list[dict]:
    """
    Flatten every player×action whose probability clears `threshold`.
    Returns dicts {player, position, action, prob} sorted by prob desc.
    """
    everyone = team_props(stats, team, team_mu, n_top=99)
    hits = []
    for p in everyone:
        for key, label in PROP_LABELS.items():
            prob = p[key]
            if prob >= threshold:
                hits.append({
                    "player":   p["player"],
                    "position": p["position"],
                    "action":   label,
                    "prob":     prob,
                })
    hits.sort(key=lambda h: h["prob"], reverse=True)
    return hits


def load_match_xg() -> pd.DataFrame | None:
    if not MATCH_XG_CSV.exists():
        return None
    return pd.read_csv(MATCH_XG_CSV, parse_dates=["date"])


def team_recent_xg(xg_df: pd.DataFrame, team: str, n: int = 5) -> tuple[float, int]:
    """
    Recency-weighted average xG-for over a team's last n WC 2026 matches.
    Returns (weighted_xg, n_matches). (0.0, 0) when the team has no xG yet.
    """
    games = xg_df[xg_df["team"] == team].sort_values("date").tail(n)
    if games.empty:
        return 0.0, 0
    vals = games["xg_for"].to_numpy()              # oldest → newest
    weights = XG_DECAY ** np.arange(len(vals) - 1, -1, -1)
    return float(np.dot(weights, vals) / weights.sum()), len(vals)


def _squad_attack(stats: pd.DataFrame, team: str) -> float:
    """Squad-based goals/match estimate: Σ player goals-per-90 × minutes share."""
    squad = stats[stats["team"] == team]
    if squad.empty:
        return 0.0
    if "active_2026" in squad.columns and squad["active_2026"].any():
        squad = squad[squad["active_2026"]]
    start_frac = squad.apply(_start_fraction, axis=1)
    return float((squad["goals_per90"].clip(lower=0) * start_frac).sum())


def team_attack_strength(stats: pd.DataFrame, team: str,
                         xg_df: pd.DataFrame | None = None) -> float:
    """
    Estimate a team's goals-per-match. Base estimate comes from the squad's
    goals-per-90. When live WC 2026 xG is available it is blended in, gaining
    weight as more matches are played (full weight at 3+ games) because xG is a
    cleaner forward-looking signal than squad goal history.
    """
    squad_est = _squad_attack(stats, team)
    if xg_df is None:
        return squad_est

    xg_est, n_matches = team_recent_xg(xg_df, team)
    if n_matches == 0:
        return squad_est

    w = min(n_matches, 3) / 3.0          # 1 game→0.33, 2→0.67, 3+→1.0
    return w * xg_est + (1 - w) * squad_est


def split_expected_goals(total_goals: float, home_elo: float, away_elo: float,
                         home_attack: float | None = None,
                         away_attack: float | None = None) -> tuple[float, float]:
    """
    Split a predicted match total into home/away expected goals.

    Base split uses Elo strength. When squad attacking strengths are supplied
    (from team_attack_strength), the Elo share is blended 50/50 with the share
    implied by the two squads' goal output — a sharper, data-driven split.
    Neutral venue assumed; share squeezed into [0.30, 0.70] so the weaker side
    never collapses to zero expected goals.
    """
    elo_share = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))

    if home_attack and away_attack and (home_attack + away_attack) > 0:
        attack_share = home_attack / (home_attack + away_attack)
        share_home = 0.5 * elo_share + 0.5 * attack_share
    else:
        share_home = elo_share

    share_home = 0.30 + 0.40 * share_home
    return total_goals * share_home, total_goals * (1 - share_home)
