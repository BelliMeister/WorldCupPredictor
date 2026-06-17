"""
Fetch per-match expected goals (xG) for WC 2026 from API-Football and store a
running per-team table. xG predicts future scoring better than actual goals
(it strips out finishing luck), so this feeds a live attacking-quality signal
that sharpens the expected-goals split and player props as the tournament runs.

Output: data/raw/wc2026_xg.csv   (one row per team per finished match)

Usage:
  python src/fetch_match_xg.py
"""

from pathlib import Path

import pandas as pd

from fetch_player_stats import _api_get, WC_LEAGUE, WC_SEASON
from player_props import TEAM_NAME_MAP

OUT_CSV = Path("data/raw/wc2026_xg.csv")


def _team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def _xg_value(team_block: dict) -> float | None:
    for s in team_block.get("statistics", []):
        if s.get("type") == "expected_goals":
            v = s.get("value")
            if v in (None, ""):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def fetch_match_xg() -> pd.DataFrame:
    fixtures = _api_get("fixtures", {"league": WC_LEAGUE, "season": WC_SEASON}).get("response", [])
    finished = [f for f in fixtures if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN")]
    print(f"Finished WC 2026 fixtures: {len(finished)}")

    rows = []
    for f in finished:
        fid  = f["fixture"]["id"]
        date = f["fixture"]["date"][:10]
        stat = _api_get("fixtures/statistics", {"fixture": fid}).get("response", [])
        if len(stat) != 2:
            continue
        xg    = {tm["team"]["id"]: _xg_value(tm) for tm in stat}
        names = {tm["team"]["id"]: _team(tm["team"]["name"]) for tm in stat}
        ids = list(xg.keys())
        if any(xg[i] is None for i in ids):
            continue  # xG not published for this fixture
        a, b = ids
        rows.append({"date": date, "team": names[a], "opponent": names[b],
                     "xg_for": xg[a], "xg_against": xg[b]})
        rows.append({"date": date, "team": names[b], "opponent": names[a],
                     "xg_for": xg[b], "xg_against": xg[a]})

    return pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)


def main():
    df = fetch_match_xg()
    if df.empty:
        print("No xG rows available yet.")
        return
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} team-match xG rows ({df['team'].nunique()} teams) → {OUT_CSV}")
    print("\nMost recent:")
    print(df.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
