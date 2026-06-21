"""
Predict match outcome and find value bets.

Usage:
  python src/predict.py --home "Brazil" --away "Morocco"
  python src/predict.py --home "Brazil" --away "Morocco" --odds-home 1.80 --odds-draw 3.40 --odds-away 4.50
"""

import difflib
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import (
    FEATURE_COLS, ELO_DEFAULT, MASTER_CSV_PATH, SQUAD_VALUES_PATH,
    load_fifa_rankings, build_rank_lookup, get_rank_at,
    _decay_mean,
)
from player_props import (
    load_player_stats, load_match_xg, load_team_stats, defensive_factors,
    props_by_market, split_expected_goals, team_attack_strength,
    PROP_THRESHOLD, TEAM_NAME_MAP,
)
from goal_timing import load_goal_timing, half_scoring_probs

MODEL_DIR   = Path("models")
RESULTS_CSV = Path("data/raw/international_results.csv")
MASTER_CSV  = Path(__file__).parent.parent.parent / "world cup match odds" / "data" / "processed" / "master.csv"

LABELS = ["Home Win", "Draw", "Away Win"]

# Common alternate names → dataset name
TEAM_ALIASES: dict[str, str] = {
    # Turkey
    "turkiye": "Turkey", "türkiye": "Turkey", "turky": "Turkey",
    # USA
    "usa": "United States", "us": "United States", "america": "United States",
    "united states of america": "United States",
    # Ivory Coast
    "cote d'ivoire": "Ivory Coast", "côte d'ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast", "ivory": "Ivory Coast",
    # Netherlands
    "holland": "Netherlands", "dutch": "Netherlands",
    # South Korea
    "korea": "South Korea", "korea republic": "South Korea",
    "republic of korea": "South Korea", "south korean": "South Korea",
    # North Korea
    "dprk": "North Korea", "democratic people's republic of korea": "North Korea",
    # Czech Republic
    "czechia": "Czech Republic", "czech": "Czech Republic",
    # Bosnia
    "bosnia": "Bosnia and Herzegovina", "herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    # DR Congo
    "congo dr": "DR Congo", "democratic republic of congo": "DR Congo",
    "democratic republic of the congo": "DR Congo", "drc": "DR Congo",
    "congo democratic": "DR Congo",
    # UAE
    "uae": "United Arab Emirates", "emirates": "United Arab Emirates",
    # Ireland
    "ireland": "Republic of Ireland", "roi": "Republic of Ireland",
    # Iran
    "ir iran": "Iran", "islamic republic of iran": "Iran",
    # Curacao (accent)
    "curacao": "Curaçao", "curaçao": "Curaçao",
    # Cape Verde
    "cape verde islands": "Cape Verde",
    # Trinidad
    "trinidad": "Trinidad and Tobago", "trinidad & tobago": "Trinidad and Tobago",
    # Serbia
    "yugoslavia": "Serbia",
    # North Macedonia
    "macedonia": "North Macedonia", "fyrom": "North Macedonia",
    # China
    "china pr": "China", "prc": "China",
    # Russia
    "russian federation": "Russia",
    # Saudi Arabia
    "ksa": "Saudi Arabia",
    # England / UK
    "uk": "England",
}


def _all_known_teams() -> list[str]:
    if not RESULTS_CSV.exists():
        return []
    df = pd.read_csv(RESULTS_CSV, usecols=["home_team", "away_team"])
    return sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))


def resolve_team(raw: str) -> str:
    """
    Map user input to the canonical dataset team name.
    1. Exact match
    2. Alias lookup (case-insensitive)
    3. Fuzzy match against all known teams — prompts user to confirm
    """
    stripped = raw.strip()

    # 1. Exact match
    known = _all_known_teams()
    if stripped in known:
        return stripped

    # 2. Alias lookup
    alias_key = stripped.lower()
    if alias_key in TEAM_ALIASES:
        resolved = TEAM_ALIASES[alias_key]
        print(f"    → recognised as '{resolved}'")
        return resolved

    # 3. Fuzzy match
    close = difflib.get_close_matches(stripped, known, n=3, cutoff=0.5)
    if len(close) == 1:
        print(f"    → did you mean '{close[0]}'? (using it)")
        return close[0]
    if close:
        print(f"    → closest matches: {', '.join(close)}")
        pick = input(f"    Enter exact name or press Enter to use '{close[0]}': ").strip()
        return pick if pick else close[0]

    # No match found — use as-is with a warning
    print(f"    ⚠ '{stripped}' not found in dataset — prediction may be less accurate")
    return stripped


# ── Models ─────────────────────────────────────────────────────────────────────

def load_models():
    with open(MODEL_DIR / "outcome_model.pkl", "rb") as f:
        outcome_model = pickle.load(f)
    with open(MODEL_DIR / "goals_model.pkl", "rb") as f:
        goals_model = pickle.load(f)
    with open(MODEL_DIR / "elo_ratings.pkl", "rb") as f:
        elo_ratings = pickle.load(f)
    return outcome_model, goals_model, elo_ratings


# ── Feature vector ─────────────────────────────────────────────────────────────

def _load_xg_lookup(n_games: int = 7) -> dict[str, list[tuple]]:
    """Returns {team: [(date, xg_for), ...]} sorted by date."""
    if not MASTER_CSV_PATH.exists():
        return {}
    xg_df = pd.read_csv(MASTER_CSV_PATH, parse_dates=["date"],
                         usecols=["date", "home_team", "away_team", "home_xg", "away_xg"])
    xg_df = xg_df.dropna(subset=["home_xg", "away_xg"]).sort_values("date")
    hist: dict[str, list] = {}
    for _, row in xg_df.iterrows():
        hist.setdefault(row["home_team"], []).append((row["date"], row["home_xg"]))
        hist.setdefault(row["away_team"], []).append((row["date"], row["away_xg"]))
    return hist


def _team_xg(hist: dict, team: str, before: pd.Timestamp, n: int = 7) -> float:
    past = [xg for d, xg in hist.get(team, []) if d < before][-n:]
    return _decay_mean(past) if past else np.nan


def _load_squad_values() -> dict[str, float]:
    if not SQUAD_VALUES_PATH.exists():
        return {}
    sv = pd.read_csv(SQUAD_VALUES_PATH).set_index("team")["squad_value_eur_m"].to_dict()
    return sv


def build_match_features(home, away, elo_ratings, rank_lookup, is_neutral=True):
    home_elo = elo_ratings.get(home, ELO_DEFAULT)
    away_elo = elo_ratings.get(away, ELO_DEFAULT)

    today = pd.Timestamp.now()
    if rank_lookup:
        home_rank, home_pts = get_rank_at(rank_lookup, home, today)
        away_rank, away_pts = get_rank_at(rank_lookup, away, today)
    else:
        home_rank = away_rank = home_pts = away_pts = np.nan

    rank_diff = (away_rank - home_rank) if not (np.isnan(home_rank) or np.isnan(away_rank)) else np.nan
    pts_diff  = (home_pts - away_pts)   if not (np.isnan(home_pts)  or np.isnan(away_pts))  else np.nan

    sv = _load_squad_values()
    DEFAULT_SV = float(np.nanmedian(list(sv.values()))) if sv else 100.0
    home_sv = sv.get(home, DEFAULT_SV)
    away_sv = sv.get(away, DEFAULT_SV)
    sv_ratio = home_sv / away_sv if away_sv > 0 else 1.0

    xg_hist = _load_xg_lookup()
    today = pd.Timestamp.now()
    home_xg = _team_xg(xg_hist, home, today)
    away_xg = _team_xg(xg_hist, away, today)

    D_SCORED, D_CONCEDED, D_WIN, D_PTS = 1.4, 1.1, 0.45, 1.2

    # Name-keyed so ordering always follows FEATURE_COLS (no positional drift).
    # Temporal/momentum signals are unknown at predict time → NaN (median-imputed).
    values = {
        "elo_diff": home_elo - away_elo,
        "home_elo_before": home_elo,
        "away_elo_before": away_elo,
        "elo_momentum_diff": np.nan,
        "home_elo_momentum": np.nan,
        "away_elo_momentum": np.nan,
        "rest_diff": np.nan,
        "home_matches_30d": np.nan,
        "away_matches_30d": np.nan,
        "home_avg_scored": D_SCORED, "home_avg_conceded": D_CONCEDED,
        "home_win_rate": D_WIN, "home_points_rate": D_PTS,
        "away_avg_scored": D_SCORED, "away_avg_conceded": D_CONCEDED,
        "away_win_rate": D_WIN, "away_points_rate": D_PTS,
        "h2h_home_win_rate": 0.5, "h2h_avg_goals": 2.5,
        "rank_diff": rank_diff, "pts_diff": pts_diff,
        "home_fifa_rank": home_rank, "away_fifa_rank": away_rank,
        "is_neutral": int(is_neutral), "is_world_cup": 1,
        "home_squad_value": home_sv, "away_squad_value": away_sv,
        "squad_value_ratio": sv_ratio,
        "home_avg_xg": home_xg, "away_avg_xg": away_xg,
    }
    missing = set(FEATURE_COLS) - set(values)
    assert not missing, f"build_match_features missing: {missing}"
    features = [values[c] for c in FEATURE_COLS]
    return np.array(features, dtype=float).reshape(1, -1)


# ── Poisson goal lines ─────────────────────────────────────────────────────────

def poisson_over(lam: float, threshold: float) -> float:
    """P(goals > threshold) using Poisson distribution."""
    k = int(math.floor(threshold))
    cdf = sum(math.exp(-lam) * (lam ** i) / math.factorial(i) for i in range(k + 1))
    return max(0.0, min(1.0, 1 - cdf))


def btts_prob(home_mu: float, away_mu: float) -> float:
    """P(both teams score >= 1 goal)."""
    p_home_scores = 1 - math.exp(-home_mu)
    p_away_scores = 1 - math.exp(-away_mu)
    return p_home_scores * p_away_scores


def goal_line_table(predicted_total: float, home_mu: float, away_mu: float) -> list[tuple]:
    """Returns rows of (label, probability) for all relevant lines."""
    lines = []
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
        prob = poisson_over(predicted_total, threshold)
        lines.append((f"Over {threshold}", prob))
    lines.append(("BTTS (yes)", btts_prob(home_mu, away_mu)))
    return lines


# ── Last 5 games ───────────────────────────────────────────────────────────────

def last_n_games(team: str, n: int = 5) -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])

    mask = (df["home_team"] == team) | (df["away_team"] == team)
    games = df[mask].sort_values("date").tail(n).copy()

    rows = []
    for _, r in games.iterrows():
        is_home = r["home_team"] == team
        opponent = r["away_team"] if is_home else r["home_team"]
        scored   = r["home_score"] if is_home else r["away_score"]
        conceded = r["away_score"] if is_home else r["home_score"]
        if scored > conceded:
            result = "W"
        elif scored == conceded:
            result = "D"
        else:
            result = "L"
        rows.append({
            "date":     r["date"].strftime("%b %d"),
            "ha":       "H" if is_home else "A",
            "opponent": opponent,
            "result":   result,
            "score":    f"{int(scored)}-{int(conceded)}",
            "tournament": r.get("tournament", ""),
        })

    return pd.DataFrame(rows)


# ── Discipline & set pieces ────────────────────────────────────────────────────

TEAM_STATS_CSV = Path("data/raw/team_stats.csv")


def discipline_stats(team: str, n: int = 10) -> dict | None:
    """API-Football team stats (primary), falling back to master.csv."""
    primary = _discipline_from_api(team)
    if primary is not None:
        return primary
    return _discipline_from_master(team, n)


def _discipline_from_api(team: str) -> dict | None:
    if not TEAM_STATS_CSV.exists():
        return None
    df = pd.read_csv(TEAM_STATS_CSV)
    df["team"] = df["team"].replace(TEAM_NAME_MAP)
    row = df[df["team"] == team]
    if row.empty:
        return None
    r = row.iloc[0]
    out = {k: (float(r[k]) if k in r and not pd.isna(r[k]) else None)
           for k in ("corners", "yellow", "red", "fouls")}
    return out if any(v is not None for v in out.values()) else None


def _discipline_from_master(team: str, n: int = 10) -> dict | None:
    if not MASTER_CSV.exists():
        return None

    df = pd.read_csv(MASTER_CSV, parse_dates=["date"])
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    games = df[mask].sort_values("date").tail(n)

    if games.empty:
        return None

    stat_cols = {
        "corners":  ("home_corners",   "away_corners"),
        "yellow":   ("home_yellow",    "away_yellow"),
        "red":      ("home_red",       "away_red"),
        "fouls":    ("home_shots",     "away_shots"),   # fouls not in data, use shots as proxy
    }

    # Remap fouls — master.csv has HF/AF for WC sheets but they got dropped in normalisation
    # Use available discipline cols
    avail = {}
    for label, (h_col, a_col) in stat_cols.items():
        vals = []
        for _, r in games.iterrows():
            is_home = r["home_team"] == team
            col = h_col if is_home else a_col
            if col in r and not pd.isna(r[col]):
                vals.append(r[col])
        avail[label] = round(np.mean(vals), 1) if vals else None

    return avail if any(v is not None for v in avail.values()) else None


# ── Value bets ─────────────────────────────────────────────────────────────────

# Filters from the out-of-sample backtest (src/backtest.py): on 743 matches the
# least-losing 1X2 subset was edge>5%, model prob>35%, odds<3.5 (~breakeven at
# best-available odds). Outside this band the model reliably loses to the book.
VALUE_MIN_EDGE = 0.05
VALUE_MIN_PROB = 0.35
VALUE_MAX_ODDS = 3.5


def find_value_bets(probs, odds, min_edge=VALUE_MIN_EDGE):
    bets = []
    for label, prob, odd in zip(LABELS, probs, odds):
        implied = 1.0 / odd
        edge = prob - implied
        in_band = prob >= VALUE_MIN_PROB and odd <= VALUE_MAX_ODDS
        if edge >= min_edge and in_band:
            kelly = (prob * odd - 1) / (odd - 1)
            bets.append({
                "outcome": label, "model_prob": prob,
                "implied_prob": implied, "edge": edge,
                "odds": odd, "quarter_kelly": max(0.0, kelly * 0.25),
            })
    return sorted(bets, key=lambda x: x["edge"], reverse=True)


# ── Display helpers ────────────────────────────────────────────────────────────

def bar(prob: float, width: int = 28) -> str:
    return "█" * int(prob * width)


def fmt_rank(rank_lookup, team, today):
    if not rank_lookup:
        return "?"
    r = get_rank_at(rank_lookup, team, today)[0]
    return "?" if np.isnan(r) else int(r)


def print_form(team: str, games: pd.DataFrame):
    print(f"\n  {team.upper()} — Last {len(games)} Games")
    if games.empty:
        print("    No recent data found.")
        return
    for _, r in games.iterrows():
        symbol = {"W": "✓", "D": "—", "L": "✗"}[r["result"]]
        tourn  = f"  [{r['tournament'][:20]}]" if r["tournament"] else ""
        print(f"    {r['date']}  {r['ha']}  vs {r['opponent']:<28}  {symbol} {r['result']}  {r['score']}{tourn}")
    results = games["result"].tolist()
    form_str = " ".join(results)
    scored   = sum(int(s.split("-")[0]) for s in games["score"])
    conceded = sum(int(s.split("-")[1]) for s in games["score"])
    print(f"    Form: {form_str}  |  Scored: {scored}  Conceded: {conceded}  "
          f"(avg {scored/len(games):.1f} / {conceded/len(games):.1f} per game)")


def print_player_props(home: str, away: str, home_mu: float, away_mu: float):
    stats = load_player_stats()
    if stats is None:
        print("\n  PLAYER PROPS — run `python src/fetch_player_stats.py` to enable")
        return

    # Each team's shot/foul props are scaled by the OPPONENT's defensive profile.
    team_stats = load_team_stats()
    home_opp_factors = defensive_factors(team_stats, away)
    away_opp_factors = defensive_factors(team_stats, home)

    print(f"\n  PLAYER PROPS  (per market — ▶ WC MW1 starter, ▷ started last 3; listed first; ★ = above {PROP_THRESHOLD:.0%})")
    print(f"  expected goals: {home} {home_mu:.2f} / {away} {away_mu:.2f}")
    print(f"  opp-defence adj  {home}: shots ×{home_opp_factors['shots']:.2f}, shot-quality ×{home_opp_factors['quality']:.2f}"
          f"  |  {away}: shots ×{away_opp_factors['shots']:.2f}, shot-quality ×{away_opp_factors['quality']:.2f}")
    for team, mu, opp_f in [(home, home_mu, home_opp_factors), (away, away_mu, away_opp_factors)]:
        markets = props_by_market(stats, team, mu, top_n=5, opp_factors=opp_f)
        print(f"\n  {team.upper()}")
        if not any(rows for _, rows in markets):
            print("    no player data")
            continue
        for label, rows in markets:
            if not rows:
                continue
            def mark(r):
                if r.get("started_first_wc"):
                    return "▶"
                return "▷" if r.get("recent_starter") else ""
            picks = "   ".join(
                f"{mark(r)}"
                f"{r['player'].split()[-1] if r['player'] else r['player']} "
                f"{r['prob']:.0%}{'★' if r['prob'] >= PROP_THRESHOLD else ''}"
                for r in rows
            )
            print(f"    {label:<16} {picks}")


def print_discipline(home: str, away: str):
    h_stats = discipline_stats(home)
    a_stats = discipline_stats(away)
    if not h_stats and not a_stats:
        return

    print("\n  DISCIPLINE & SET PIECES  (avg per game, recency-weighted last games)")
    print(f"  {'':24}  {'Corners':>8}  {'Fouls':>7}  {'Yellow':>7}  {'Red':>5}")

    def cell(stats, key):
        return f"{stats[key]:.1f}" if stats and stats.get(key) is not None else "N/A"

    def row(name, stats):
        print(f"  {name:<24}  {cell(stats,'corners'):>8}  {cell(stats,'fouls'):>7}  "
              f"{cell(stats,'yellow'):>7}  {cell(stats,'red'):>5}")

    row(home, h_stats)
    row(away, a_stats)


# ── Main ───────────────────────────────────────────────────────────────────────

def prompt(label: str, cast=str, optional: bool = False):
    while True:
        raw = input(f"  {label}: ").strip()
        if not raw and optional:
            return None
        if not raw:
            print("    Required — please enter a value.")
            continue
        if cast is float:
            try:
                return float(raw)
            except ValueError:
                print("    Enter a number (e.g. 2.10).")
                continue
        return cast(raw)


def main():
    print("\n  WC 2026 Match Predictor")
    print("  " + "─" * 30)
    home       = resolve_team(prompt("Home team"))
    away       = resolve_team(prompt("Away team"))
    print()
    odds_home  = prompt("Home odds  (press Enter to skip)", cast=float, optional=True)
    odds_away  = prompt("Away odds  (press Enter to skip)", cast=float, optional=True)
    odds_draw  = prompt("Draw odds  (press Enter to skip)", cast=float, optional=True)

    outcome_model, goals_model, elo_ratings = load_models()
    rankings    = load_fifa_rankings()
    rank_lookup = build_rank_lookup(rankings) if rankings is not None else None

    X = build_match_features(home, away, elo_ratings, rank_lookup, is_neutral=True)

    probs           = outcome_model.predict_proba(X)[0].tolist()
    predicted_total = float(goals_model.predict(X)[0])
    confidence      = max(probs)

    home_elo  = elo_ratings.get(home, ELO_DEFAULT)
    away_elo  = elo_ratings.get(away, ELO_DEFAULT)
    today     = pd.Timestamp.now()
    home_rank = fmt_rank(rank_lookup, home, today)
    away_rank = fmt_rank(rank_lookup, away, today)

    sv = _load_squad_values()
    DEFAULT_SV = float(np.nanmedian(list(sv.values()))) if sv else 100.0
    home_sv = sv.get(home, DEFAULT_SV)
    away_sv = sv.get(away, DEFAULT_SV)

    # Split predicted total into home/away expected goals — Elo blended with
    # squad attacking strength from the player-stats data (sharper split).
    player_stats = load_player_stats()
    match_xg     = load_match_xg()
    if player_stats is not None:
        home_attack = team_attack_strength(player_stats, home, match_xg)
        away_attack = team_attack_strength(player_stats, away, match_xg)
    else:
        home_attack = away_attack = None
    home_mu, away_mu = split_expected_goals(
        predicted_total, home_elo, away_elo, home_attack, away_attack
    )

    W = 58
    print(f"\n{'='*W}")
    print(f"  {home}  vs  {away}")
    print(f"  Elo:       {home_elo:.0f}  vs  {away_elo:.0f}"
          f"  |  FIFA Rank: #{home_rank}  vs  #{away_rank}")
    print(f"  Squad €M:  {home_sv:.0f}M  vs  {away_sv:.0f}M")
    print(f"  Model confidence: {confidence:.1%}")
    print(f"{'='*W}")

    # ── Outcome ────────────────────────────────────────────────────────────────
    print("\n  OUTCOME PROBABILITIES")
    for label, prob in zip(LABELS, probs):
        marker = " ◄" if prob == confidence else ""
        print(f"  {label:<12}  {prob:>5.1%}  {bar(prob)}{marker}")

    # ── Goals ──────────────────────────────────────────────────────────────────
    print(f"\n  GOALS  (predicted total: {predicted_total:.2f})")
    lines = goal_line_table(predicted_total, home_mu, away_mu)
    for label, prob in lines:
        indicator = ""
        if "Over" in label:
            threshold = float(label.split()[1])
            indicator = "  ← prediction sits here" if (
                threshold == 2.5 or
                (predicted_total > threshold and predicted_total <= threshold + 1)
            ) else ""
        print(f"  {label:<14}  {prob:>5.1%}  {bar(prob)}{indicator}")

    # ── First / second half scoring ────────────────────────────────────────────
    timing = load_goal_timing()
    if timing is not None:
        print("\n  HALF SCORING  (chance the team scores in each half)")
        print(f"  {'':18}{'1st half':>10}{'2nd half':>10}")
        for team, mu in [(home, home_mu), (away, away_mu)]:
            p1, p2 = half_scoring_probs(timing, team, mu)
            print(f"  {team[:18]:<18}{p1:>9.0%}{p2:>10.0%}")

    # ── Last 5 games ───────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    home_games = last_n_games(home)
    away_games = last_n_games(away)
    print_form(home, home_games)
    print_form(away, away_games)

    # ── Discipline ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print_discipline(home, away)

    # ── Player props ───────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print_player_props(home, away, home_mu, away_mu)

    # ── Value bets ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    has_odds = all(o is not None for o in [odds_home, odds_draw, odds_away])
    if has_odds:
        odds_list = [odds_home, odds_draw, odds_away]
        bets = find_value_bets(probs, odds_list)
        print("\n  VALUE BET ANALYSIS")
        if bets:
            for b in bets:
                print(f"\n  ► {b['outcome']}")
                print(f"    Model prob   {b['model_prob']:.1%}  vs  implied {b['implied_prob']:.1%}")
                print(f"    Edge         +{b['edge']:.1%}  at odds {b['odds']}")
                print(f"    Stake        {b['quarter_kelly']:.1%} of bankroll  (¼ Kelly)")
            print("\n    ⚠ Backtest note: 1X2 'value' bets were ~breakeven-to-negative")
            print("      out of sample even at best odds. Shop the best line; size small.")
        else:
            print("\n  No qualifying 1X2 value (edge≥5%, prob≥35%, odds≤3.5).")
    else:
        print("\n  Add --odds-home X --odds-draw X --odds-away X for value bet analysis")

    print(f"\n{'='*W}\n")


if __name__ == "__main__":
    main()
