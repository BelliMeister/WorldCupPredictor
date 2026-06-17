"""
Build the per-player prop dataset from API-Football (v3.football.api-sports.io).

For every WC-2026 nation we take that team's 5 MOST RECENT international games and
aggregate each player's stats with a combined weight:

    weight = recency_decay ** (games_ago)  ×  importance_K(competition)

so a goal in last week's World Cup game counts far more than one in an old
friendly. Per-90 rates are computed from the weighted totals.

Output: data/raw/player_stats.csv  (schema consumed by player_props.py)

Usage:
  python src/fetch_player_stats.py              # all 48 WC teams, last 5 games each
  python src/fetch_player_stats.py --last 5
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from config import require_key, API_FOOTBALL_KEY

API_BASE   = "https://v3.football.api-sports.io"
WC_LEAGUE  = 1
WC_SEASON  = 2026
OUT_CSV       = Path("data/raw/player_stats.csv")
TEAM_STATS_CSV = Path("data/raw/team_stats.csv")
CACHE_DIR     = Path(".cache/apifootball")

RECENCY_DECAY = 0.85      # weight of each game one step further back
REQUEST_PAUSE = 0.12      # seconds between calls (stay under rate limit)

# Competition importance multiplier (matched by substring on the league name)
IMPORTANCE_K = [
    ("World Cup - Qualification", 1.10),
    ("World Cup",                 1.60),
    ("Euro",                      1.20),
    ("Copa America",              1.20),
    ("Africa Cup",                1.15),
    ("Asian Cup",                 1.15),
    ("Gold Cup",                  1.10),
    ("Nations League",            1.05),
    ("Qualification",             1.05),
    ("Friendl",                   0.60),
]
DEFAULT_K = 0.90

# API-Football position code → player_props position bucket
POS_MAP = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}


def importance_k(league_name: str) -> float:
    for needle, k in IMPORTANCE_K:
        if needle.lower() in (league_name or "").lower():
            return k
    return DEFAULT_K


def _api_get(path: str, params: dict) -> dict:
    """GET with a simple on-disk cache (keyed by path+params)."""
    key = require_key(API_FOOTBALL_KEY)
    qs  = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API_BASE}/{path}?{qs}"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / (path.replace("/", "_") + "_" + qs.replace("&", "_").replace("=", "-") + ".json")
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    req = urllib.request.Request(url, headers={"x-apisports-key": key})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    time.sleep(REQUEST_PAUSE)
    cache_file.write_text(json.dumps(data))
    return data


def get_wc_teams() -> list[dict]:
    data = _api_get("teams", {"league": WC_LEAGUE, "season": WC_SEASON})
    return [{"id": t["team"]["id"], "name": t["team"]["name"]} for t in data.get("response", [])]


def get_last_fixtures(team_id: int, last: int) -> list[dict]:
    data = _api_get("fixtures", {"team": team_id, "last": last})
    out = []
    for f in data.get("response", []):
        if f["fixture"]["status"]["short"] not in ("FT", "AET", "PEN"):
            continue
        out.append({
            "fixture_id": f["fixture"]["id"],
            "date":       f["fixture"]["date"][:10],
            "league":     f["league"]["name"],
        })
    return out


def get_fixture_players(fixture_id: int, team_id: int) -> list[dict]:
    """Return the given team's player stat rows for one fixture (cached per fixture)."""
    data = _api_get("fixtures/players", {"fixture": fixture_id})
    for tm in data.get("response", []):
        if tm["team"]["id"] == team_id:
            return tm["players"]
    return []


def _num(v) -> float:
    return float(v) if v is not None else 0.0


def first_wc_starters(team_id: int) -> set[str]:
    """
    Names of players who started this team's FIRST WC 2026 match (matchweek 1).
    A confirmed starter is a much safer prop than a bench/squad player.
    Empty set if the team hasn't played its opener yet.
    """
    data = _api_get("fixtures", {"team": team_id, "league": WC_LEAGUE, "season": WC_SEASON})
    played = [f for f in data.get("response", [])
              if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN")]
    if not played:
        return set()
    played.sort(key=lambda f: f["fixture"]["date"])  # earliest = matchweek 1
    first_id = played[0]["fixture"]["id"]

    starters = set()
    for p in get_fixture_players(first_id, team_id):
        g = p["statistics"][0]["games"]
        if g.get("substitute") is False and _num(g.get("minutes")) > 0:
            starters.add(p["player"]["name"])
    return starters


def _stat_value(stats: list[dict], type_name: str):
    """Pull one numeric stat (e.g. 'Corner Kicks') from a fixtures/statistics block."""
    for s in stats:
        if s.get("type") == type_name:
            v = s.get("value")
            if v in (None, ""):
                return None
            if isinstance(v, str) and v.endswith("%"):
                v = v[:-1]
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def team_match_stats(team: dict, last: int) -> dict | None:
    """
    Recency- and importance-weighted per-game team discipline / set-piece averages
    from API-Football fixtures/statistics over the team's last `last` games.
    """
    fixtures = get_last_fixtures(team["id"], last)
    if not fixtures:
        return None
    fixtures.sort(key=lambda f: f["date"], reverse=True)

    FIELDS = {"corners": "Corner Kicks", "yellow": "Yellow Cards", "red": "Red Cards",
              "fouls": "Fouls", "shots": "Total Shots", "sot": "Shots on Goal"}
    acc = {k: 0.0 for k in FIELDS}
    wsum = 0.0
    matches = 0

    for games_ago, fx in enumerate(fixtures):
        data = _api_get("fixtures/statistics", {"fixture": fx["fixture_id"]})
        block = next((tm for tm in data.get("response", []) if tm["team"]["id"] == team["id"]), None)
        if not block:
            continue
        vals = {k: _stat_value(block["statistics"], name) for k, name in FIELDS.items()}
        if all(v is None for v in vals.values()):
            continue
        w = (RECENCY_DECAY ** games_ago) * importance_k(fx["league"])
        wsum += w
        matches += 1
        for k, v in vals.items():
            acc[k] += w * (v or 0.0)

    if matches == 0 or wsum == 0:
        return None
    row = {"team": team["name"], "matches": matches}
    row.update({k: round(acc[k] / wsum, 2) for k in FIELDS})
    return row


def build_team_rows(team: dict, last: int) -> list[dict]:
    fixtures = get_last_fixtures(team["id"], last)
    if not fixtures:
        return []

    starters = first_wc_starters(team["id"])

    # Most recent first → games_ago index drives the recency decay
    fixtures.sort(key=lambda f: f["date"], reverse=True)

    acc: dict[str, dict] = {}  # player name -> weighted accumulators
    for games_ago, fx in enumerate(fixtures):
        w = (RECENCY_DECAY ** games_ago) * importance_k(fx["league"])
        for p in get_fixture_players(fx["fixture_id"], team["id"]):
            st  = p["statistics"][0]
            mins = _num(st["games"].get("minutes"))
            if mins <= 0:
                continue
            name = p["player"]["name"]
            a = acc.setdefault(name, {
                "raw_minutes": 0.0, "matches": 0, "position": "", "starts_last3": 0,
                "wmin": 0.0, "goals": 0.0, "assists": 0.0, "shots": 0.0,
                "sot": 0.0, "fouls": 0.0, "fouled": 0.0, "yellow": 0.0, "red": 0.0,
            })
            a["raw_minutes"] += mins
            a["matches"]     += 1
            # Started (not a sub) in one of the 3 most recent games → likely starter
            if games_ago < 3 and st["games"].get("substitute") is False:
                a["starts_last3"] += 1
            pos = st["games"].get("position")
            a["position"]    = POS_MAP.get(pos, a["position"] or "MF")
            a["wmin"]    += w * mins
            a["goals"]   += w * _num(st["goals"].get("total"))
            a["assists"] += w * _num(st["goals"].get("assists"))
            a["shots"]   += w * _num(st["shots"].get("total"))
            a["sot"]     += w * _num(st["shots"].get("on"))
            a["fouls"]   += w * _num(st["fouls"].get("committed"))
            a["fouled"]  += w * _num(st["fouls"].get("drawn"))
            a["yellow"]  += w * _num(st["cards"].get("yellow"))
            a["red"]     += w * _num(st["cards"].get("red"))

    rows = []
    for name, a in acc.items():
        w90 = a["wmin"] / 90.0
        per90 = (lambda x: round(x / w90, 3) if w90 > 0 else 0.0)
        rows.append({
            "player":   name,
            "team":     team["name"],
            "position": a["position"],
            "active_2026": True,
            "started_first_wc": name in starters,
            "starts_last3": a["starts_last3"],
            "recent_starter": a["starts_last3"] >= 2,
            "minutes":  round(a["raw_minutes"], 0),
            "matches":  a["matches"],
            "goals":    round(a["goals"], 3),
            "assists":  round(a["assists"], 3),
            "shots":    round(a["shots"], 3),
            "sot":      round(a["sot"], 3),
            "fouls":    round(a["fouls"], 3),
            "fouled":   round(a["fouled"], 3),
            "yellow":   round(a["yellow"], 3),
            "red":      round(a["red"], 3),
            "goals_per90":   per90(a["goals"]),
            "assists_per90": per90(a["assists"]),
            "shots_per90":   per90(a["shots"]),
            "sot_per90":     per90(a["sot"]),
            "fouls_per90":   per90(a["fouls"]),
            "yellow_per90":  per90(a["yellow"]),
            "g_per_sh":      round(a["goals"] / a["shots"], 3) if a["shots"] > 0 else 0.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=10,
                        help="Most recent N internationals per team (deeper = more stable rates)")
    args = parser.parse_args()

    teams = get_wc_teams()
    print(f"WC 2026 teams: {len(teams)}  |  using last {args.last} games each")

    all_rows = []
    team_rows = []
    for i, team in enumerate(teams, 1):
        try:
            rows = build_team_rows(team, args.last)
            all_rows.extend(rows)
            ts = team_match_stats(team, args.last)
            if ts:
                team_rows.append(ts)
            print(f"  [{i:>2}/{len(teams)}] {team['name']:<24} {len(rows):>3} players")
        except Exception as e:
            print(f"  [{i:>2}/{len(teams)}] {team['name']:<24} FAILED: {type(e).__name__}: {str(e)[:80]}")

    if not all_rows:
        print("No player data fetched — aborting.")
        return

    df = pd.DataFrame(all_rows).sort_values(["team", "goals_per90"], ascending=[True, False])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} players → {OUT_CSV}")

    if team_rows:
        pd.DataFrame(team_rows).to_csv(TEAM_STATS_CSV, index=False)
        print(f"Saved {len(team_rows)} team discipline rows → {TEAM_STATS_CSV}")
    top = df[df["minutes"] >= 90].head(10)
    print("\nTop weighted scorers (goals/90):")
    print(top[["player", "team", "goals_per90", "shots_per90", "sot_per90"]].to_string(index=False))


if __name__ == "__main__":
    main()
