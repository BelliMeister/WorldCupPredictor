"""
First-half / second-half scoring model.

Uses the minute of every historical international goal (goalscorers.csv) to learn
each team's scoring timing — what share of their goals come before vs after the
break — then splits the match expected-goals into each half and applies Poisson
to give the chance a team scores in the 1st and 2nd half.

Read-only; no API calls.
"""

import math
from pathlib import Path

import pandas as pd

GOALSCORERS_CSV = (Path(__file__).parent.parent.parent
                   / "world cup match odds" / "data" / "processed" / "goalscorers.csv")

# Shrinkage toward the league baseline for teams with few recorded goals
SHRINK_K = 20


def load_goal_timing() -> dict | None:
    """Return league 1H baseline plus per-team first-half/total goal counts."""
    if not GOALSCORERS_CSV.exists():
        return None
    g = pd.read_csv(GOALSCORERS_CSV, usecols=["team", "minute"])
    g["minute"] = pd.to_numeric(g["minute"], errors="coerce")
    g = g.dropna(subset=["minute"])
    if g.empty:
        return None

    is_first = g["minute"] <= 45
    league_1h = float(is_first.mean())
    first = g[is_first].groupby("team").size()
    total = g.groupby("team").size()
    return {"league_1h": league_1h, "first": first.to_dict(), "total": total.to_dict()}


def team_first_half_fraction(timing: dict, team: str) -> float:
    """Shrunk fraction of a team's goals scored in the first half."""
    league = timing["league_1h"]
    f = timing["first"].get(team, 0)
    n = timing["total"].get(team, 0)
    return (f + SHRINK_K * league) / (n + SHRINK_K)


def half_scoring_probs(timing: dict, team: str, team_mu: float) -> tuple[float, float]:
    """
    (P(scores in 1st half), P(scores in 2nd half)) given the team's expected
    goals for the match, split by its historical timing and run through Poisson.
    """
    frac1 = team_first_half_fraction(timing, team)
    mu1 = team_mu * frac1
    mu2 = team_mu * (1 - frac1)
    p1 = 1.0 - math.exp(-mu1) if mu1 > 0 else 0.0
    p2 = 1.0 - math.exp(-mu2) if mu2 > 0 else 0.0
    return p1, p2
