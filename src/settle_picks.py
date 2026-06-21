"""
Settle logged player-prop picks against actual results (API-Football) and report
calibration — does a 70%-rated pick actually land ~70% of the time?

Reads/updates data/predictions/picks.csv. Pending picks for finished fixtures are
graded; a calibration table (predicted prob vs actual hit-rate) is printed over all
settled picks.

Run: python src/settle_picks.py
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd

from config import require_key, API_FOOTBALL_KEY

API_BASE  = "https://v3.football.api-sports.io"
WC_LEAGUE = 1
WC_SEASON = 2026
PICKS_CSV = Path("data/predictions/picks.csv")

# API-Football team spelling → our spelling (for fixture matching)
NAME_FIX = {"USA": "United States", "Korea Republic": "South Korea", "Czechia": "Czech Republic",
            "IR Iran": "Iran", "Türkiye": "Turkey", "Bosnia & Herzegovina": "Bosnia and Herzegovina"}


def _get(path: str, params: dict) -> dict:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"{API_BASE}/{path}?{qs}", headers={"x-apisports-key": require_key(API_FOOTBALL_KEY)})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _norm(name: str) -> str:
    return NAME_FIX.get(name, name)


def _stat(p_stats: dict, market: str) -> float | None:
    """Actual value for a market from a fixtures/players statistics block."""
    g, sh, go, fl, cd = (p_stats.get(k, {}) or {} for k in ("games", "shots", "goals", "fouls", "cards"))
    if (g.get("minutes") or 0) == 0:
        return None  # did not play → void
    if market == "sot":       return float(sh.get("on") or 0)
    if market == "shots":     return float(sh.get("total") or 0)
    if market == "to_score":  return float(go.get("total") or 0)
    if market == "to_assist": return float(go.get("assists") or 0)
    if market == "fouls":     return float(fl.get("committed") or 0)
    if market == "booked":    return float((cd.get("yellow") or 0) + (cd.get("red") or 0))
    return None


def find_fixture_id(home: str, away: str, date: str) -> int | None:
    data = _get("fixtures", {"league": WC_LEAGUE, "season": WC_SEASON})
    for f in data.get("response", []):
        h, a = _norm(f["teams"]["home"]["name"]), _norm(f["teams"]["away"]["name"])
        if h == home and a == away and f["fixture"]["date"][:10] == date:
            if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN"):
                return f["fixture"]["id"]
    return None


def player_stats_for_fixture(fixture_id: int) -> dict[str, dict]:
    """Map lowercased player name → statistics block for a finished fixture."""
    data = _get("fixtures/players", {"fixture": fixture_id})
    out = {}
    for tm in data.get("response", []):
        for p in tm["players"]:
            out[p["player"]["name"].lower()] = p["statistics"][0]
    return out


def _match_player(name: str, lineup: dict[str, dict]) -> dict | None:
    key = name.lower()
    if key in lineup:
        return lineup[key]
    last = name.split()[-1].lower()
    hits = [v for k, v in lineup.items() if last in k.split()]
    return hits[0] if len(hits) == 1 else None


def settle():
    if not PICKS_CSV.exists():
        print("No picks log found.")
        return
    df = pd.read_csv(PICKS_CSV)

    fixtures = {}  # (home,away,date) -> {id, lineup}
    settled_now = 0
    for i, p in df.iterrows():
        if p["status"] != "pending":
            continue
        key = (p["home_team"], p["away_team"], p["fixture_date"])
        if key not in fixtures:
            fid = find_fixture_id(*key)
            fixtures[key] = {"id": fid, "lineup": player_stats_for_fixture(fid) if fid else None}
        fx = fixtures[key]
        if not fx["id"] or fx["lineup"] is None:
            continue  # not finished yet

        ps = _match_player(p["player"], fx["lineup"])
        if ps is None:
            df.at[i, "status"] = "void"; df.at[i, "result"] = "player_not_found"
            settled_now += 1; continue
        actual = _stat(ps, p["market"])
        if actual is None:
            df.at[i, "status"] = "void"; df.at[i, "result"] = "dnp"
            settled_now += 1; continue

        won = actual >= p["line"]
        df.at[i, "actual"] = actual
        df.at[i, "status"] = "won" if won else "lost"
        df.at[i, "result"] = "win" if won else "loss"
        settled_now += 1

    df.to_csv(PICKS_CSV, index=False)
    print(f"Settled {settled_now} pick(s) this run.\n")
    report(df)


def report(df: pd.DataFrame):
    graded = df[df["status"].isin(["won", "lost"])].copy()
    if graded.empty:
        print("No graded picks yet — run again after the fixtures finish.")
        return
    graded["win"] = (graded["status"] == "won").astype(int)
    n, hits = len(graded), int(graded["win"].sum())
    print(f"=== CALIBRATION ({n} settled picks) ===")
    print(f"  Overall hit rate : {hits/n:.0%}   ({hits}/{n})")
    print(f"  Mean model prob  : {graded['model_prob'].mean():.0%}  (should track hit rate if calibrated)")
    roi = ((graded["win"] * graded["fair_odds"]) - 1).mean()
    print(f"  ROI @ fair odds  : {roi:+.1%}  (≈0 means well-calibrated to its own prices)")
    print("\n  By probability bucket:")
    print(f"  {'bucket':>10} {'picks':>6} {'pred':>6} {'actual':>7}")
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)]:
        b = graded[(graded["model_prob"] >= lo) & (graded["model_prob"] < hi)]
        if len(b):
            print(f"  {f'{lo:.0%}-{hi:.0%}':>10} {len(b):>6} {b['model_prob'].mean():>6.0%} {b['win'].mean():>7.0%}")


if __name__ == "__main__":
    settle()
