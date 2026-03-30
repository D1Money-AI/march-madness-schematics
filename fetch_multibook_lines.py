"""
Pull closing-style lines from public APIs and merge into NCAA_Mens_Tournament_2026_Complete_Results.csv.

1) ESPN game summaries (no key): pickcenter is typically DraftKings full-game spread + total for
   completed NCAA tournament games (group 50, season type 3).

2) Optional The Odds API (ODDS_API_KEY env): current snapshot for basketball_ncaab — useful for
   upcoming games and extra books (FanDuel, BetMGM, William Hill US, etc.). Completed games usually
   disappear from the live odds feed; ESPN remains the practical backfill for those.

Does not scrape William Hill / Circa directly. Circa is often unavailable on aggregator APIs; if
The Odds API adds a book key, add it to BOOKMAKER_KEYS below.

Usage:
  python fetch_multibook_lines.py              # update CSV in place
  python fetch_multibook_lines.py --dry-run    # print match stats only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent
CSV_PATH = DIR / "NCAA_Mens_Tournament_2026_Complete_Results.csv"

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
)
ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
)

# The Odds API v4 — add book keys you are entitled to on your plan.
BOOKMAKER_KEYS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "williamhill_us",
    "betrivers",
    "wynnbet",
    "espnbet",
    "bovada",
    "lowvig",
    "mybookieag",
]

# Map CSV / full school names to ESPN summary team.location (lowercase).
CANON_MAP: dict[str, str] = {
    "university of maryland baltimore county": "umbc",
    "university of texas at austin": "texas",
    "north carolina state university": "nc state",
    "university of california los angeles": "ucla",
    "university of connecticut": "uconn",
    "university of southern california": "usc",
    "university of central florida": "ucf",
    "university of northern iowa": "northern iowa",
    "virginia commonwealth university": "vcu",
    "saint louis university": "saint louis",
    "university of hawaii": "hawaii",
    "university of missouri": "missouri",
    "miami university": "miami (oh)",
    "university of miami": "miami",
    "university of pennsylvania": "pennsylvania",
    "california baptist university": "california baptist",
    "university of north carolina": "north carolina",
    "long island university": "long island university",
    "st john's university": "st. john's",
    "st. john's university": "st. john's",
    "southern methodist university": "smu",
    "brigham young university": "byu",
    "texas christian university": "tcu",
    "university of akron": "akron",
    "university of idaho": "idaho",
    "university of nevada": "nevada",  # if used
    "university of florida": "florida",
    "university of kansas": "kansas",
    "university of kentucky": "kentucky",
    "university of louisville": "louisville",
    "university of nebraska": "nebraska",
    "university of tennessee": "tennessee",
    "university of virginia": "virginia",
    "university of wisconsin": "wisconsin",
    "university of arkansas": "arkansas",
    "university of alabama": "alabama",
    "university of arizona": "arizona",
    "university of illinois": "illinois",
    "university of iowa": "iowa",
    "university of michigan": "michigan",
    "university of houston": "houston",
    "university of georgia": "georgia",
    "university of dayton": "dayton",
    "texas a&m university": "texas a&m",
    "iowa state university": "iowa state",
    "michigan state university": "michigan state",
    "oklahoma state university": "oklahoma state",
    "north dakota state university": "north dakota state",
    "wright state university": "wright state",
    "tennessee state university": "tennessee state",
    "kennesaw state university": "kennesaw state",
    "utah state university": "utah state",
    "mcneese state university": "mcneese",
    "northern kentucky university": "northern kentucky",
    "southeast missouri state university": "southeast missouri state",
    "fairleigh dickinson university": "fairleigh dickinson",
    "oral roberts university": "oral roberts",
    "texas southern university": "texas southern",
    "arizona state university": "arizona state",
    "florida atlantic university": "florida atlantic",
    "george mason university": "george mason",
    "college of charleston": "college of charleston",
    "james madison university": "james madison",
    "grand canyon university": "grand canyon",
    "samford university": "samford",
    "sam houston state university": "sam houston",
    "montana state university": "montana state",
    "montana university": "montana",
    "prairie view a&m university": "prairie view a&m",
    "troy university": "troy",
    "santa clara university": "santa clara",
    "furman university": "furman",
    "howard university": "howard",
    "duke university": "duke",
    "gonzaga university": "gonzaga",
    "villanova university": "villanova",
    "xavier university": "xavier",
    "marquette university": "marquette",
    "depaul university": "depaul",
    "seton hall university": "seton hall",
    "providence college": "providence",
    "georgetown university": "georgetown",
    "baylor university": "baylor",
    "purdue university": "purdue",
    "indiana university": "indiana",
    "rutgers university": "rutgers",
    "northwestern university": "northwestern",
    "ohio state university": "ohio state",
    "penn state university": "penn state",
    "clemson university": "clemson",
    "syracuse university": "syracuse",
    "wake forest university": "wake forest",
    "florida state university": "florida state",
    "louisiana state university": "lsu",
    "ole miss": "ole miss",
    "university of mississippi": "ole miss",
    "mississippi state university": "mississippi state",
    "auburn university": "auburn",
    "vanderbilt university": "vanderbilt",
    "texas tech university": "texas tech",
    "oklahoma university": "oklahoma",
    "university of oklahoma": "oklahoma",
    "kansas state university": "kansas state",
    "west virginia university": "west virginia",
    "texas a&m-corpus christi": "texas a&m-corpus christi",
    "uc santa barbara": "uc santa barbara",
    "uc irvine": "uc irvine",
    "uc san diego": "uc san diego",
    "uc riverside": "uc riverside",
    "uc davis": "uc davis",
    "san diego state university": "san diego state",
    "san jose state university": "san jose state",
    "fresno state university": "fresno state",
    "boise state university": "boise state",
    "colorado state university": "colorado state",
    "new mexico university": "new mexico",
    "unlv": "unlv",
    "university of nevada las vegas": "unlv",
    "wyoming university": "wyoming",
    "air force academy": "air force",
    "navy": "navy",
    "army": "army",
    "colgate university": "colgate",
    "princeton university": "princeton",
    "harvard university": "harvard",
    "yale university": "yale",
    "columbia university": "columbia",
    "cornell university": "cornell",
    "dartmouth college": "dartmouth",
    "brown university": "brown",
    "lehigh university": "lehigh",
    "lafayette college": "lafayette",
    "bucknell university": "bucknell",
    "american university": "american",
    "navy midshipmen": "navy",
    "saint mary's college of california": "saint mary's",
    "saint peter's university": "saint peter's",
    "iona university": "iona",
    "manhattan college": "manhattan",
    "canisius college": "canisius",
    "niagara university": "niagara",
    "rider university": "rider",
    "siena college": "siena",
    "fairfield university": "fairfield",
    "monmouth university": "monmouth",
    "quinnipiac university": "quinnipiac",
    "hofstra university": "hofstra",
    "towson university": "towson",
    "delaware university": "delaware",
    "drexel university": "drexel",
    "elon university": "elon",
    "unc wilmington": "unc wilmington",
    "charleston cougars": "college of charleston",
    "high point university": "high point",
    "queens university of charlotte": "queens university",
    "unc asheville": "unc asheville",
    "florida gulf coast university": "florida gulf coast",
    "fgcu": "florida gulf coast",
    "stetson university": "stetson",
    "jacksonville university": "jacksonville",
    "north florida": "north florida",
    "lipscomb university": "lipscomb",
    "belmont university": "belmont",
    "murray state university": "murray state",
    "morehead state university": "morehead state",
    "eastern kentucky university": "eastern kentucky",
    "southeastern louisiana": "southeastern louisiana",
    "nicholls state university": "nicholls",
    "mcmurry": "mcmurry",
    "grambling state university": "grambling",
    "southern university": "southern",
    "alcorn state university": "alcorn state",
    "jackson state university": "jackson state",
    "bethune-cookman university": "bethune-cookman",
    "florida a&m university": "florida a&m",
    "north carolina a&t": "north carolina a&t",
    "north carolina central": "north carolina central",
    "delaware state university": "delaware state",
    "maryland eastern shore": "maryland eastern shore",
    "coppin state university": "coppin state",
    "morgan state university": "morgan state",
    "norfolk state university": "norfolk state",
    "south carolina state university": "south carolina state",
    "northwestern state university": "northwestern state",
    "stephen f. austin": "stephen f. austin",
    "abilene christian university": "abilene christian",
    "tarleton state university": "tarleton",
    "ut arlington": "ut arlington",
    "utep": "utep",
    "utsa": "utsa",
    "ut rio grande valley": "ut rio grande valley",
    "new mexico state university": "new mexico state",
}


def normalize_team(name: str) -> str:
    s = str(name).strip().lower()
    for prefix in ("university of ", "the "):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s.strip()


def canon_csv_team(name: str) -> str:
    s = normalize_team(name)
    s = re.sub(r"[^a-z0-9\s'\-&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in CANON_MAP:
        return CANON_MAP[s]
    if s in CANON_MAP.values():
        return s
    # longest key match
    best = None
    for k, v in CANON_MAP.items():
        if k in s or s in k:
            if best is None or len(k) > len(best[0]):
                best = (k, v)
    if best:
        return best[1]
    return s


def canon_espn_location(loc: str) -> str:
    if not loc:
        return ""
    s = loc.strip().lower()
    s = s.replace("hawai'i", "hawaii").replace("hawaiʻi", "hawaii")
    return s


def matchup_key_pair(a: str, b: str) -> frozenset[str]:
    return frozenset((canon_csv_team(a), canon_csv_team(b)))


def http_get_json(url: str, timeout: float = 35.0) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "MarchMadnessOddsFetcher/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_espn_tournament_index(
    dates_yyyymmdd: list[str],
    sleep_s: float = 0.12,
) -> dict[frozenset[str], dict]:
    """Map frozenset(canon, canon) -> {provider, overUnder, spread_str, spread_mag}."""
    out: dict[frozenset[str], dict] = {}
    seen_events: set[str] = set()

    for ymd in dates_yyyymmdd:
        q = urllib.parse.urlencode(
            {"dates": ymd, "group": 50, "seasontype": 3, "limit": 400}
        )
        url = f"{ESPN_SCOREBOARD}?{q}"
        try:
            data = http_get_json(url)
        except urllib.error.HTTPError as e:
            print(f"ESPN scoreboard HTTP {e.code} for {ymd}", file=sys.stderr)
            continue
        events = data.get("events") or []
        for ev in events:
            eid = str(ev.get("id", ""))
            if not eid or eid in seen_events:
                continue
            seen_events.add(eid)
            try:
                summ = http_get_json(f"{ESPN_SUMMARY}?event={eid}")
            except urllib.error.HTTPError:
                continue
            comps = (summ.get("header") or {}).get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            if len(competitors) < 2:
                continue
            locs: list[str] = []
            for c in competitors:
                loc = (c.get("team") or {}).get("location") or ""
                locs.append(canon_espn_location(loc))
            key = frozenset(locs)
            if len(key) < 2:
                continue

            pick = (summ.get("pickcenter") or None)
            if not pick:
                continue
            pc = pick[0]
            provider = (pc.get("provider") or {}).get("name") or "Unknown"
            ou = pc.get("overUnder")
            details = (pc.get("details") or "").strip()
            spread_val = pc.get("spread")
            if ou is None and spread_val is None and not details:
                continue

            # Spread string: use ESPN "details" when it looks like "DUKE -28.5"
            spread_str = details if re.search(r"-?\d", details) else ""
            if not spread_str and spread_val is not None:
                ho = pc.get("homeTeamOdds") or {}
                ao = pc.get("awayTeamOdds") or {}
                id_to_loc = {}
                for c in competitors:
                    tid = str((c.get("team") or {}).get("id", ""))
                    loc = canon_espn_location((c.get("team") or {}).get("location") or "")
                    id_to_loc[tid] = loc
                try:
                    sp = float(spread_val)
                except (TypeError, ValueError):
                    sp = None
                if sp is not None:
                    if ho.get("favorite"):
                        loc = id_to_loc.get(str(ho.get("teamId", "")))
                        spread_str = f"{loc} {sp:g}" if loc else str(sp)
                    elif ao.get("favorite"):
                        loc = id_to_loc.get(str(ao.get("teamId", "")))
                        spread_str = f"{loc} {-sp:g}" if loc else str(-sp)

            try:
                ou_f = float(ou) if ou is not None else None
            except (TypeError, ValueError):
                ou_f = None

            out[key] = {
                "provider": provider,
                "overUnder": ou_f,
                "spread_str": spread_str or "",
                "espn_event_id": eid,
            }
            time.sleep(sleep_s)
    return out


def odds_api_fetch() -> list[dict]:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        return []
    books = ",".join(BOOKMAKER_KEYS)
    params = urllib.parse.urlencode(
        {
            "apiKey": key,
            "regions": "us,us2",
            "markets": "spreads,totals",
            "oddsFormat": "american",
            "bookmakers": books,
        }
    )
    url = f"https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds?{params}"
    try:
        return http_get_json(url)  # type: ignore[return-value]
    except urllib.error.HTTPError as e:
        print(f"The Odds API HTTP {e.code}", file=sys.stderr)
        return []


def normalize_odds_api_team(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\s'\-&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return canon_csv_team(s)


def parse_odds_api_event(ev: dict) -> dict[frozenset[str], dict[str, dict]]:
    """Return map matchup_key -> {book_key: {total, spread_str}}."""
    home = (ev.get("home_team") or "").strip()
    away = (ev.get("away_team") or "").strip()
    key = frozenset((normalize_odds_api_team(away), normalize_odds_api_team(home)))
    per_book: dict[str, dict] = {}

    for bk in ev.get("bookmakers") or []:
        bkey = (bk.get("key") or "").lower()
        total_pt = None
        spread_str = ""
        spread_candidates: list[tuple[str, float]] = []
        for mkt in bk.get("markets") or []:
            k = (mkt.get("key") or "").lower()
            for outc in mkt.get("outcomes") or []:
                nm = (outc.get("name") or "").strip()
                pt = outc.get("point")
                if k == "totals" and nm.lower() == "over" and pt is not None:
                    try:
                        total_pt = float(pt)
                    except (TypeError, ValueError):
                        pass
                if k == "spreads" and pt is not None:
                    try:
                        p = float(pt)
                    except (TypeError, ValueError):
                        continue
                    spread_candidates.append((nm, p))
        if spread_candidates:
            team, p = min(spread_candidates, key=lambda x: x[1])
            spread_str = f"{team} {p:g}"
        if total_pt is not None or spread_str:
            per_book[bkey] = {
                "overUnder": total_pt,
                "spread_str": spread_str,
            }
    return {key: per_book} if per_book else {}


def merge_odds_api_into_rows(
    api_rows: list[dict],
) -> dict[frozenset[str], dict[str, dict]]:
    merged: dict[frozenset[str], dict[str, dict]] = {}
    for ev in api_rows:
        for k, v in parse_odds_api_event(ev).items():
            merged.setdefault(k, {}).update(v)
    return merged


def compute_market_row(row: pd.Series, book_ou_cols: list[str]) -> tuple[float | None, str]:
    """Median of numeric book O/Us (excluding 0); spread prefers MGM, then DraftKings, then first."""
    mgm_ou = pd.to_numeric(row.get("MGM Over/Under"), errors="coerce")
    values: list[float] = []
    if pd.notna(mgm_ou) and mgm_ou > 0:
        values.append(float(mgm_ou))
    for c in book_ou_cols:
        v = pd.to_numeric(row.get(c), errors="coerce")
        if pd.notna(v) and v > 0:
            values.append(float(v))
    if not values:
        return None, ""
    mid = float(statistics.median(values))

    spread_priority = [
        "MGM Point Spread",
        "BetMGM Point Spread",
        "DraftKings Point Spread",
        "FanDuel Point Spread",
        "William Hill Point Spread",
        "Circa Point Spread",
    ]
    for col in spread_priority:
        if col not in row.index:
            continue
        s = str(row.get(col) or "").strip()
        if not s or s.upper() == "TBD":
            continue
        s_up = s.upper().replace("'", "")
        if s_up in ("PK", "PICK", "PICKEM", "PICK EM"):
            return mid, "pk"
    for col in spread_priority:
        if col not in row.index:
            continue
        s = str(row.get(col) or "").strip()
        if s and s.upper() != "TBD" and re.search(r"-?\d", s):
            return mid, s
    return mid, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH)

    # Date range: infer from CSV
    dates: list[str] = []
    for raw in df["Date"].dropna().unique():
        s = str(raw).strip()
        parts = s.split("/")
        if len(parts) != 3:
            continue
        mo, da, yy = int(parts[0]), int(parts[1]), int(parts[2])
        y = 2000 + yy if yy < 100 else yy
        dates.append(f"{y:04d}{mo:02d}{da:02d}")
    dates = sorted(set(dates))

    print("Fetching ESPN tournament summaries for dates:", ", ".join(dates))
    espn_idx = fetch_espn_tournament_index(dates)
    print("ESPN unique matchups with pickcenter:", len(espn_idx))

    api_merged = merge_odds_api_into_rows(odds_api_fetch())
    if api_merged:
        print("The Odds API matchups with at least one book:", len(api_merged))
    elif os.environ.get("ODDS_API_KEY"):
        print("The Odds API returned no rows (check key / quota).", file=sys.stderr)
    else:
        print("Set ODDS_API_KEY to merge live multi-book odds (FanDuel, WH US, etc.).")

    new_cols = {
        "DraftKings Over/Under": float("nan"),
        "DraftKings Point Spread": "",
        "FanDuel Over/Under": float("nan"),
        "FanDuel Point Spread": "",
        "William Hill Over/Under": float("nan"),
        "William Hill Point Spread": "",
        "Circa Over/Under": float("nan"),
        "Circa Point Spread": "",
        "BetMGM Over/Under": float("nan"),
        "BetMGM Point Spread": "",
        "Market Over/Under": float("nan"),
        "Market Point Spread": "",
    }
    for c, default in new_cols.items():
        if c not in df.columns:
            df[c] = default

    matched = 0
    api_hit = 0
    for i, row in df.iterrows():
        if str(row.get("Team A", "")).strip().upper() == "TBD":
            continue
        ta, tb = row["Team A"], row["Team B"]
        k = matchup_key_pair(ta, tb)
        rec = espn_idx.get(k)
        if rec:
            matched += 1
            if rec.get("provider") and "draft" in str(rec["provider"]).lower():
                df.at[i, "DraftKings Over/Under"] = rec.get("overUnder")
                df.at[i, "DraftKings Point Spread"] = rec.get("spread_str") or ""
            else:
                df.at[i, "DraftKings Over/Under"] = rec.get("overUnder")
                df.at[i, "DraftKings Point Spread"] = rec.get("spread_str") or ""

        api_rec = api_merged.get(k)
        if api_rec:
            api_hit += 1
            mapping = [
                ("draftkings", "DraftKings Over/Under", "DraftKings Point Spread"),
                ("fanduel", "FanDuel Over/Under", "FanDuel Point Spread"),
                ("williamhill_us", "William Hill Over/Under", "William Hill Point Spread"),
                ("betmgm", "BetMGM Over/Under", "BetMGM Point Spread"),
            ]
            # Circa — try common keys
            for circa_key in ("circa_sports", "circa", "circasports"):
                if circa_key in api_rec:
                    cr = api_rec[circa_key]
                    df.at[i, "Circa Over/Under"] = cr.get("overUnder")
                    df.at[i, "Circa Point Spread"] = cr.get("spread_str") or ""
                    break
            for bkey, ou_col, sp_col in mapping:
                if bkey not in api_rec:
                    continue
                r = api_rec[bkey]
                if r.get("overUnder") is not None:
                    df.at[i, ou_col] = r["overUnder"]
                if r.get("spread_str"):
                    df.at[i, sp_col] = r["spread_str"]

    print("Rows matched to ESPN pickcenter:", matched, "of", len(df[df["Team A"] != "TBD"]))
    print("Rows with any Odds API book match:", api_hit)

    book_ou_cols = [
        "DraftKings Over/Under",
        "FanDuel Over/Under",
        "William Hill Over/Under",
        "Circa Over/Under",
        "BetMGM Over/Under",
    ]
    for i, row in df.iterrows():
        if str(row.get("Team A", "")).strip().upper() == "TBD":
            continue
        mou, msp = compute_market_row(row, book_ou_cols)
        if mou is not None:
            df.at[i, "Market Over/Under"] = mou
        if msp:
            df.at[i, "Market Point Spread"] = msp

    if args.dry_run:
        print("Dry run — not writing CSV.")
        return

    # Stable column order: original 8 + new + market
    base = [
        "Date",
        "Team A",
        "Team B",
        "Team A Score",
        "Team B Score",
        "Total Combined Score",
        "MGM Point Spread",
        "MGM Over/Under",
    ]
    extra = [c for c in df.columns if c not in base]
    extra.sort()
    cols = base + extra
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(CSV_PATH, index=False)
    print("Wrote", CSV_PATH)


if __name__ == "__main__":
    main()
