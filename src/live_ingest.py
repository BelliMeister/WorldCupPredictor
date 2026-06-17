"""
Live match ingestion — fetches WC 2026 results from football-data.org (primary)
with ESPN as fallback, then patches international_results.csv.

Usage:
  python src/live_ingest.py               # fetch + show live scoreboard
  python src/live_ingest.py --retrain     # fetch + patch + retrain
  python src/live_ingest.py --watch 90    # poll every 90 seconds (runs during match)
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import get_key, FOOTBALL_DATA_API_KEY

RESULTS_CSV = Path("data/raw/international_results.csv")

FD_API_KEY  = get_key(FOOTBALL_DATA_API_KEY)
FD_BASE     = "https://api.football-data.org/v4/competitions/WC/matches"
ESPN_BASE   = "http://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

# football-data.org match statuses
FD_FINISHED  = {"FINISHED"}
FD_LIVE      = {"IN_PLAY", "PAUSED", "HALFTIME"}
FD_SCHEDULED = {"TIMED", "SCHEDULED"}

# ESPN statuses
ESPN_COMPLETED = {
    "STATUS_FINAL", "STATUS_FULL_TIME", "STATUS_FULL_PEN",
    "STATUS_EXTRA_TIME", "STATUS_PENALTY", "STATUS_FT",
}

# football-data.org team names → international_results.csv names
FD_TEAM_MAP = {
    "USA":               "United States",
    "Korea Republic":    "South Korea",
    "IR Iran":           "Iran",
    "Türkiye":           "Turkey",
    "Czechia":           "Czech Republic",
    "Bosnia-Herzegovina":"Bosnia and Herzegovina",
    "Congo DR":          "DR Congo",
    "Cape Verde Islands":"Cape Verde",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Côte d'Ivoire":     "Ivory Coast",
    "Curaçao":           "Curaçao",
}

# ESPN team names → international_results.csv names
ESPN_TEAM_MAP = {
    "USA":                      "United States",
    "United States of America": "United States",
    "Korea Republic":           "South Korea",
    "IR Iran":                  "Iran",
    "Türkiye":                  "Turkey",
    "Czechia":                  "Czech Republic",
    "Bosnia-Herzegovina":       "Bosnia and Herzegovina",
    "Congo DR":                 "DR Congo",
    "Cape Verde Islands":       "Cape Verde",
    "Trinidad & Tobago":        "Trinidad and Tobago",
}


def _fd_name(name: str) -> str:
    return FD_TEAM_MAP.get(name, name)


def _espn_name(name: str) -> str:
    return ESPN_TEAM_MAP.get(name, name)


# ── football-data.org fetcher ──────────────────────────────────────────────────

def fetch_fd_matches() -> list[dict] | None:
    """Fetch all WC 2026 matches from football-data.org. Returns None on error."""
    if not FD_API_KEY:
        print("[football-data.org] no API key set — skipping (using ESPN).")
        return None
    req = urllib.request.Request(
        FD_BASE,
        headers={"X-Auth-Token": FD_API_KEY, "User-Agent": "WCPredictor/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except Exception as e:
        print(f"[football-data.org] Error: {e}")
        return None

    matches = []
    for m in data.get("matches", []):
        home   = _fd_name(m["homeTeam"]["name"])
        away   = _fd_name(m["awayTeam"]["name"])
        ft     = m.get("score", {}).get("fullTime", {})
        status = m.get("status", "")

        hs = ft.get("home")
        as_ = ft.get("away")
        matches.append({
            "date":       m["utcDate"][:10],
            "home_team":  home,
            "away_team":  away,
            "home_score": float(hs)  if hs  is not None else None,
            "away_score": float(as_) if as_ is not None else None,
            "status":     status,
            "clock":      "",
            "tournament": "FIFA World Cup",
            "country":    "United States",
            "neutral":    True,
            "source":     "fd",
        })
    return matches


# ── ESPN fallback fetcher ──────────────────────────────────────────────────────

def _espn_fetch_date(date_str: str) -> list[dict]:
    url = f"{ESPN_BASE}?dates={date_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        return data.get("events", [])
    except Exception as e:
        print(f"[ESPN] Error fetching {date_str}: {e}")
        return []


def fetch_espn_matches(lookback_days: int = 4) -> list[dict]:
    from datetime import timedelta
    today = datetime.now(timezone.utc).date()
    all_events, seen_ids = [], set()
    for delta in range(lookback_days, -1, -1):
        day = today - timedelta(days=delta)
        for event in _espn_fetch_date(day.strftime("%Y%m%d")):
            eid = event.get("id")
            if eid not in seen_ids:
                seen_ids.add(eid)
                all_events.append(event)

    matches = []
    for event in all_events:
        comp        = event.get("competitions", [{}])[0]
        status_type = comp.get("status", {}).get("type", {})
        status      = status_type.get("name", "")
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        hs  = home.get("score")
        as_ = away.get("score")
        matches.append({
            "date":       event.get("date", "")[:10],
            "home_team":  _espn_name(home["team"]["displayName"]),
            "away_team":  _espn_name(away["team"]["displayName"]),
            "home_score": float(hs)  if hs  not in (None, "") else None,
            "away_score": float(as_) if as_ not in (None, "") else None,
            "status":     status,
            "clock":      comp.get("status", {}).get("displayClock", ""),
            "tournament": "FIFA World Cup",
            "country":    "United States",
            "neutral":    True,
            "source":     "espn",
        })
    return matches


# ── Unified fetch ──────────────────────────────────────────────────────────────

def fetch_matches() -> list[dict]:
    """Try football-data.org first; fall back to ESPN if it fails."""
    matches = fetch_fd_matches()
    if matches is not None:
        print(f"[football-data.org] {len(matches)} matches fetched.")
        return matches
    print("[football-data.org] failed — falling back to ESPN...")
    matches = fetch_espn_matches()
    print(f"[ESPN] {len(matches)} matches fetched.")
    return matches


# ── Status helpers ─────────────────────────────────────────────────────────────

def _is_finished(m: dict) -> bool:
    return m["status"] in FD_FINISHED or m["status"] in ESPN_COMPLETED

def _is_live(m: dict) -> bool:
    return m["status"] in FD_LIVE or m["status"] == "STATUS_IN_PROGRESS"

def _is_scheduled(m: dict) -> bool:
    return m["status"] in FD_SCHEDULED or m["status"] == "STATUS_SCHEDULED"


# ── Scoreboard display ─────────────────────────────────────────────────────────

def print_scoreboard(matches: list[dict]):
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    print(f"\n{'='*54}")
    print(f"  WC 2026 Scoreboard  [{now}]")
    print(f"{'='*54}")

    for label, fn in [("LIVE", _is_live), ("FINISHED", _is_finished), ("UPCOMING", _is_scheduled)]:
        group = [m for m in matches if fn(m)]
        if not group:
            continue
        print(f"\n  {label}:")
        for m in group:
            if _is_scheduled(m):
                print(f"    {m['date']}  {m['home_team']} vs {m['away_team']}")
            else:
                hs  = int(m["home_score"]) if m["home_score"] is not None else "?"
                as_ = int(m["away_score"]) if m["away_score"] is not None else "?"
                clock = f"  {m['clock']}" if m["clock"] and _is_live(m) else ""
                print(f"    {m['home_team']} {hs} - {as_} {m['away_team']}{clock}")
    print()


# ── Patch results into CSV ─────────────────────────────────────────────────────

def patch_results(df: pd.DataFrame, live_matches: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    changes = []

    for m in live_matches:
        if not _is_finished(m):
            continue
        if m["home_score"] is None or m["away_score"] is None:
            continue

        match_date = pd.to_datetime(m["date"])
        # Match within ±1 day: live sources report UTC kickoff, which can roll to
        # the next calendar day vs the scheduled date — an exact match would
        # otherwise insert a duplicate row instead of updating the fixture.
        day_gap = (df["date"] - match_date).abs().dt.days
        mask = (
            (df["home_team"] == m["home_team"])
            & (df["away_team"] == m["away_team"])
            & (day_gap <= 1)
            & (df["tournament"] == "FIFA World Cup")
        )

        label = f"{m['date']}  {m['home_team']} {int(m['home_score'])}-{int(m['away_score'])} {m['away_team']}"

        if mask.any():
            if pd.isna(df.loc[mask, "home_score"].iloc[0]):
                df.loc[mask, "home_score"] = m["home_score"]
                df.loc[mask, "away_score"] = m["away_score"]
                changes.append(f"UPDATED  {label}")
        else:
            new_row = pd.DataFrame([{
                "date":       match_date,
                "home_team":  m["home_team"],
                "away_team":  m["away_team"],
                "home_score": m["home_score"],
                "away_score": m["away_score"],
                "tournament": m["tournament"],
                "city":       "",
                "country":    m["country"],
                "neutral":    m["neutral"],
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            changes.append(f"INSERTED {label}")

    df = df.sort_values("date").reset_index(drop=True)
    return df, changes


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_once(retrain: bool = False) -> bool:
    matches = fetch_matches()
    if not matches:
        print("No data returned.")
        return False

    print_scoreboard(matches)

    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    df_updated, changes = patch_results(df, matches)

    if changes:
        print("Ingested:")
        for c in changes:
            print(f"  {c}")
        df_updated.to_csv(RESULTS_CSV, index=False)
        print(f"  Saved → {RESULTS_CSV}")
        if retrain:
            print("\nRetraining model...")
            subprocess.run([sys.executable, "src/train.py", "--quick"])
        return True
    else:
        finished = [m for m in matches if _is_finished(m)]
        if finished:
            print(f"  {len(finished)} completed match(es) already recorded — no update needed.")
        else:
            print("  No completed matches yet.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Retrain after patching new results")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                        help="Poll every N seconds until stopped")
    args = parser.parse_args()

    if args.watch > 0:
        print(f"Watching for results every {args.watch}s — Ctrl+C to stop.\n")
        while True:
            try:
                run_once(retrain=args.retrain)
                time.sleep(args.watch)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        run_once(retrain=args.retrain)


if __name__ == "__main__":
    main()
