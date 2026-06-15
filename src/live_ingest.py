"""
Live match ingestion — fetches WC 2026 results from ESPN and
patches international_results.csv, then optionally retrains the model.

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

RESULTS_CSV = Path("data/raw/international_results.csv")
ESPN_BASE = "http://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

# All statuses ESPN uses for a completed match
COMPLETED_STATUSES = {
    "STATUS_FINAL",
    "STATUS_FULL_TIME",
    "STATUS_FULL_PEN",
    "STATUS_EXTRA_TIME",
    "STATUS_PENALTY",
    "STATUS_FT",
}

# ESPN display names → names in international_results.csv
TEAM_NAME_MAP = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Trinidad & Tobago": "Trinidad and Tobago",
}


def normalise_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def _fetch_date(date_str: str) -> list[dict]:
    """Fetch ESPN scoreboard for a specific date (YYYYMMDD)."""
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
    """Fetch WC matches for today and the last lookback_days days."""
    from datetime import timedelta
    today = datetime.now(timezone.utc).date()
    all_events = []
    seen_ids = set()
    for delta in range(lookback_days, -1, -1):
        day = today - timedelta(days=delta)
        date_str = day.strftime("%Y%m%d")
        for event in _fetch_date(date_str):
            eid = event.get("id")
            if eid not in seen_ids:
                seen_ids.add(eid)
                all_events.append(event)

    matches = []
    for event in all_events:
        comp = event.get("competitions", [{}])[0]
        status_type = comp.get("status", {}).get("type", {})
        status = status_type.get("name", "")

        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_score = home.get("score")
        away_score = away.get("score")

        matches.append({
            "date": event.get("date", "")[:10],
            "home_team": normalise_name(home["team"]["displayName"]),
            "away_team": normalise_name(away["team"]["displayName"]),
            "home_score": float(home_score) if home_score not in (None, "") else None,
            "away_score": float(away_score) if away_score not in (None, "") else None,
            "status": status,
            "clock": comp.get("status", {}).get("displayClock", ""),
            "period": comp.get("status", {}).get("period", 0),
            "tournament": "FIFA World Cup",
            "country": "United States",
            "neutral": True,
        })

    return matches


def print_scoreboard(matches: list[dict]):
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    print(f"\n{'='*54}")
    print(f"  WC 2026 Scoreboard  [{now}]")
    print(f"{'='*54}")

    for label, status_filter in [
        ("LIVE", lambda s: s == "STATUS_IN_PROGRESS"),
        ("FINISHED", lambda s: s in COMPLETED_STATUSES),
        ("UPCOMING", lambda s: s == "STATUS_SCHEDULED"),
    ]:
        group = [m for m in matches if status_filter(m["status"])]
        if not group:
            continue
        print(f"\n  {label}:")
        for m in group:
            if m["status"] == "STATUS_SCHEDULED":
                print(f"    {m['date']}  {m['home_team']} vs {m['away_team']}")
            else:
                hs = int(m["home_score"]) if m["home_score"] is not None else "?"
                as_ = int(m["away_score"]) if m["away_score"] is not None else "?"
                clock = f"  {m['clock']}" if m["clock"] and m["status"] == "STATUS_IN_PROGRESS" else ""
                print(f"    {m['home_team']} {hs} - {as_} {m['away_team']}{clock}")
    print()


def patch_results(df: pd.DataFrame, live_matches: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Insert or update completed match scores. Returns (updated_df, change_list)."""
    df = df.copy()
    changes = []

    for m in live_matches:
        if m["status"] not in COMPLETED_STATUSES:
            continue
        if m["home_score"] is None or m["away_score"] is None:
            continue

        match_date = pd.to_datetime(m["date"])
        mask = (
            (df["home_team"] == m["home_team"])
            & (df["away_team"] == m["away_team"])
            & (df["date"].dt.date == match_date.date())
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
                "date": match_date,
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "home_score": m["home_score"],
                "away_score": m["away_score"],
                "tournament": m["tournament"],
                "city": "",
                "country": m["country"],
                "neutral": m["neutral"],
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            changes.append(f"INSERTED {label}")

    df = df.sort_values("date").reset_index(drop=True)
    return df, changes


def run_once(retrain: bool = False) -> bool:
    print("Fetching from ESPN...")
    matches = fetch_espn_matches()
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
        finished = [m for m in matches if m["status"] in COMPLETED_STATUSES]
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
