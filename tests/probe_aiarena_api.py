"""
probe_aiarena_api.py - Interactive diagnostic script for the aiarena.net REST API.

Purpose in the larger system:
    The ladder-replay pipeline (src_new/data_processing/fetch_bot_replays.py) has
    been returning zero downloaded replays even though Macro_ladder.json is
    populated with competition IDs. The suspected cause: the pipeline's
    `_query_full_game_competition_ids` accepts EVERY competition whose name
    lacks the substring "micro", and `_validate_competition_id` only checks
    HTTP 200 — so historical / closed / frozen competitions slip through.
    Rounds queried for those competitions return nothing, so no match IDs are
    produced, and nothing is downloaded.

    This script is a sandbox for inspecting the raw API responses so we can:
      1. See every field the API returns on each endpoint (not just the ones
         fetch_bot_replays.py currently uses).
      2. Identify which filter/query parameters (e.g. status, game_mode,
         type, is_open) actually work on /api/competitions/.
      3. Confirm which competitions are genuinely LIVE right now versus
         historical, so the pipeline can be tightened to only target those.
      4. Walk through the competition -> rounds -> matches -> results
         chain for a single competition to verify replay_file URLs exist.

How to use:
    Run with no args to execute every probe in order:
        python SC2-gamestate-extractor/tests/probe_aiarena_api.py

    Or run a specific probe:
        python SC2-gamestate-extractor/tests/probe_aiarena_api.py --probe competitions
        python SC2-gamestate-extractor/tests/probe_aiarena_api.py --probe rounds --comp-id 34
        python SC2-gamestate-extractor/tests/probe_aiarena_api.py --probe matches --round-id 1234
        python SC2-gamestate-extractor/tests/probe_aiarena_api.py --probe results --match-id 5678
        python SC2-gamestate-extractor/tests/probe_aiarena_api.py --probe filters
        python SC2-gamestate-extractor/tests/probe_aiarena_api.py --probe chain

    All probes save their raw JSON output into tests/api_probe_output/
    so you can diff / grep the captured data after the fact.

Dependencies:
    - requests, python-dotenv: already in the project's requirements.
    - AIARENA_TOKEN and AIARENA_NET_URL environment variables: loaded from .env
      at the project root (same as the production pipeline does).

Calls / imports from the project:
    - src_new.data_processing.fetch_bot_replays.authorize — reused so we hit
      the API with the exact same auth headers and base URL as production.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library imports
import argparse            # Parsing CLI flags so each probe can be run individually.
import json                # Pretty-printing and saving JSON responses for inspection.
import os                  # Creating the output directory and reading env vars.
import sys                 # Adjusting sys.path so the project's src_new package imports cleanly.
from pathlib import Path   # Cross-platform path manipulation.
from pprint import pformat # Human-readable dict/list formatting for terminal output.

# Third-party imports
import requests            # Actually hitting the aiarena.net REST API.

# ---------------------------------------------------------------------------
# Path setup — allow this script to import from src_new/ regardless of cwd
# ---------------------------------------------------------------------------
# Layout:
#   local-play-bootstrap-main/
#     SC2-gamestate-extractor/
#       src_new/data_processing/fetch_bot_replays.py   <-- we import from here
#       tests/probe_aiarena_api.py                     <-- THIS file
#
# Path(__file__).parent.parent resolves to SC2-gamestate-extractor/, which is
# the package root containing src_new/. Inserting it at position 0 of sys.path
# lets us `from src_new... import ...` below.
_EXTRACTOR_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXTRACTOR_ROOT))

# Reuse the production authorize() so we send the exact same headers /
# base URL the pipeline does — eliminates "works here but not there" confusion.
from src_new.data_processing.fetch_bot_replays import authorize  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Where probe responses are persisted. Kept relative to THIS script so the
# output lands next to the script regardless of the working directory the user
# runs from. Created lazily in save_json().
OUTPUT_DIR = Path(__file__).resolve().parent / "api_probe_output"

# Hard cap on how many pages of a paginated endpoint we will walk. The
# competitions endpoint is small (<100 rows), but we still need a ceiling so
# a misbehaving probe cannot spin forever. Bump this if you want a full dump.
MAX_PAGES = 5

# Hard cap on how many items to print to the terminal from a single page.
# Avoids flooding the console; the full response is always saved to disk.
MAX_PRINTED_ITEMS = 5

# How many seconds to wait on any single HTTP request before aborting. The
# aiarena API is fast (usually <1s), so 20s is generous but bounded.
REQUEST_TIMEOUT = 20


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def save_json(label: str, data) -> Path:
    """
    Persist `data` as pretty-printed JSON under OUTPUT_DIR/<label>.json.

    Saving every probe response means we can diff successive runs (e.g. "did
    the list of competitions change between yesterday and today?") and grep
    through the captured data without re-hitting the API.

    Calls:
        Path.mkdir — creates OUTPUT_DIR on first use.
        json.dump — serializes with indent=2 for readability.

    Args:
        label (str): Short descriptive name that becomes the filename stem
                     (e.g. "competitions_page1"). No extension, no slashes.
        data: Any JSON-serializable object (usually a dict or list).

    Returns:
        Path to the written file, so the caller can print it in logs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return out_path


def pretty(obj, limit_chars: int = 4000) -> str:
    """
    Render `obj` as a pretty string, truncated if it is gigantic.

    Args:
        obj: Any Python object.
        limit_chars (int): Maximum characters to render; longer output is
                           truncated with a suffix to keep terminal output
                           manageable.

    Returns:
        The (possibly truncated) pretty-printed string.
    """
    text = pformat(obj, width=120, sort_dicts=False)
    if len(text) > limit_chars:
        text = text[:limit_chars] + f"\n... [truncated {len(text) - limit_chars} chars] ..."
    return text


def banner(title: str) -> None:
    """Print a visually distinct section header for the terminal output."""
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def get_json(url: str, auth_header: dict, params: dict = None) -> dict:
    """
    Issue a GET against `url` with the aiarena auth header and return the JSON.

    Keeps request handling in one place so every probe logs HTTP status and
    response size the same way, and every probe times out identically.

    Args:
        url (str): Full URL (may or may not already contain query params).
        auth_header (dict): {"Authorization": "Token ..."} from authorize().
        params (dict): Optional extra query params (requests will URL-encode).

    Returns:
        Parsed JSON (dict or list).

    Raises:
        requests.HTTPError on non-2xx responses. The caller decides whether
        to surface the error or continue — for a diagnostic script we usually
        want the traceback to be loud.
    """
    resp = requests.get(url, headers=auth_header, params=params, timeout=REQUEST_TIMEOUT)
    print(f"  GET {resp.url}  -> {resp.status_code}  ({len(resp.content)} bytes)")
    resp.raise_for_status()
    return resp.json()


def paginate(start_url: str, auth_header: dict, max_pages: int = MAX_PAGES) -> list:
    """
    Walk a DRF-paginated endpoint, collecting every `results[]` item.

    The aiarena API uses Django REST Framework's default pagination shape:
        {"count": N, "next": "...", "previous": "...", "results": [...]}

    We stop once either there is no `next` URL or we've hit max_pages — this
    protects against runaway loops on endpoints that return thousands of rows.

    Calls:
        get_json — performs each individual page fetch.

    Args:
        start_url (str): Fully-qualified first page URL.
        auth_header (dict): Auth header from authorize().
        max_pages (int): Safety cap on pages to walk.

    Returns:
        Combined list of result objects across all fetched pages.
    """
    all_items = []
    url = start_url
    page = 0
    while url and page < max_pages:
        page += 1
        data = get_json(url, auth_header)
        items = data.get("results", [])
        all_items.extend(items)
        url = data.get("next")
    if url:
        # Means we bailed because of max_pages, not because we reached the end.
        print(f"  [note] Stopped paginating at page {page}; more pages exist at {url!r}.")
    return all_items


# ---------------------------------------------------------------------------
# Probe 1 — Competitions endpoint: what fields come back? what statuses exist?
# ---------------------------------------------------------------------------

def probe_competitions(auth_header: dict, base_url: str) -> list:
    """
    Dump the full list of competitions so we can see their real fields.

    The key question we're answering: what distinguishes a LIVE competition
    from a closed/frozen/historical one? Until we know which field carries
    that information (likely `status`), we cannot tighten the pipeline.

    Calls:
        paginate — walks /api/competitions/ across pages.
        save_json — persists the raw response for offline inspection.

    Args:
        auth_header (dict): Auth header from authorize().
        base_url (str): e.g. "https://aiarena.net/api".

    Returns:
        list of competition dicts (all pages combined).
    """
    banner("PROBE: competitions — raw field dump + status breakdown")
    url = f"{base_url}/competitions/"
    comps = paginate(url, auth_header)
    print(f"Total competitions returned: {len(comps)}")

    # Show a single full record so every field name is visible at a glance.
    # We pick the first one; users can grep the saved JSON for other examples.
    if comps:
        print("\n--- First competition (every field shown) ---")
        print(pretty(comps[0]))

    # Build a compact status summary. The .get() calls are defensive: if the
    # field does not exist on a given record we fall back to "(missing)" so
    # we notice schema gaps instead of silently dropping rows.
    print("\n--- Summary table (id | name | status | type | game_mode | date_opened | date_closed) ---")
    for c in comps:
        print(
            f"  {c.get('id'):>4} | {c.get('name', '')[:45]:<45} | "
            f"{str(c.get('status', '(missing)'))[:10]:<10} | "
            f"{str(c.get('type', '(missing)'))[:6]:<6} | "
            f"{str(c.get('game_mode', '(missing)'))[:10]:<10} | "
            f"{str(c.get('date_opened', '(missing)'))[:10]:<10} | "
            f"{str(c.get('date_closed', '(missing)'))[:10]}"
        )

    # Count of each distinct status so we can see how many are actually open.
    status_counts = {}
    for c in comps:
        s = c.get("status", "(missing)")
        status_counts[s] = status_counts.get(s, 0) + 1
    print("\n--- status value distribution ---")
    for s, n in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {s!r:<15} -> {n}")

    saved = save_json("competitions_all", comps)
    print(f"\nFull JSON saved to: {saved}")
    return comps


# ---------------------------------------------------------------------------
# Probe 2 — Competitions filter experiments: which query params actually work?
# ---------------------------------------------------------------------------

def probe_competition_filters(auth_header: dict, base_url: str) -> None:
    """
    Try a grab-bag of filter query params on /competitions/ and show counts.

    Django REST Framework only honors filters that are explicitly registered
    on the viewset. Unknown params are silently ignored, which means the
    result will simply match the un-filtered list. To spot real filters we
    compare each filtered count against the baseline count — anything
    meaningfully smaller probably works.

    Calls:
        get_json — single-page sample per filter (no pagination needed).

    Args:
        auth_header (dict): Auth header from authorize().
        base_url (str): e.g. "https://aiarena.net/api".
    """
    banner("PROBE: competitions filter experiments")
    url = f"{base_url}/competitions/"

    # Candidate filter params to try. If the API ignores an unknown param it
    # will just return the baseline count; a real filter will narrow results.
    candidate_filters = [
        {},                                          # baseline — no filter
        {"status": "open"},
        {"status": "closed"},
        {"status": "frozen"},
        {"status": "paused"},
        {"status": "closing"},
        {"is_open": "true"},
        {"game_mode": "full_game"},
        {"game_mode": "micro"},
        {"type": "L"},                               # DRF often uses 'L' for league
        {"type": "R"},                               # 'R' for round robin, perhaps
        {"ordering": "-date_opened"},                # sanity check: ordering works?
        {"search": "macro"},                         # DRF SearchFilter
        {"search": "micro"},
        {"name": "Macro"},
    ]

    baseline_count = None
    for params in candidate_filters:
        label = ", ".join(f"{k}={v}" for k, v in params.items()) or "(no filter)"
        try:
            data = get_json(url, auth_header, params=params)
        except requests.HTTPError as e:
            print(f"  {label:<40} -> HTTP error: {e}")
            continue
        count = data.get("count")
        if baseline_count is None and not params:
            baseline_count = count
        names = [c.get("name") for c in data.get("results", [])[:MAX_PRINTED_ITEMS]]
        print(f"  {label:<40} -> count={count}  sample={names}")

    print(
        "\nInterpretation: any filter whose `count` matches the baseline "
        f"({baseline_count}) is likely being ignored — those params are NOT "
        "real filters on this endpoint."
    )


# ---------------------------------------------------------------------------
# Probe 3 — Rounds for a single competition: are there any completed rounds?
# ---------------------------------------------------------------------------

def probe_rounds(auth_header: dict, base_url: str, comp_id: int) -> list:
    """
    Inspect rounds under a single competition to see if any are complete.

    `fetch_ladder_match_ids` only accepts `complete=true` rounds. If a
    competition has zero completed rounds, the pipeline will silently return
    no matches — exactly the symptom the user is seeing.

    Calls:
        get_json — one call with complete=true, one without, so we can see
                   the delta between "all rounds" and "completed rounds".
        save_json — persists the raw responses.

    Args:
        auth_header (dict): Auth header from authorize().
        base_url (str): e.g. "https://aiarena.net/api".
        comp_id (int): Competition ID to inspect.

    Returns:
        list of round dicts (first page, unpaginated) for further probing.
    """
    banner(f"PROBE: rounds for competition {comp_id}")

    # 1. How many rounds total exist for this competition?
    all_rounds = get_json(
        f"{base_url}/rounds/",
        auth_header,
        params={"competition": comp_id, "ordering": "-started"},
    )
    print(f"Rounds (all)      count={all_rounds.get('count')}")

    # 2. How many of those are complete? This is what the production pipeline
    #    actually reads — if this is 0 the competition is unusable.
    complete_rounds = get_json(
        f"{base_url}/rounds/",
        auth_header,
        params={"competition": comp_id, "complete": "true", "ordering": "-started"},
    )
    print(f"Rounds (complete) count={complete_rounds.get('count')}")

    results = complete_rounds.get("results", [])
    if results:
        print("\n--- First completed round (all fields) ---")
        print(pretty(results[0]))
    else:
        print("\n[WARN] No completed rounds — this competition yields zero match IDs.")

    save_json(f"rounds_all_comp{comp_id}", all_rounds)
    save_json(f"rounds_complete_comp{comp_id}", complete_rounds)
    return results


# ---------------------------------------------------------------------------
# Probe 4 — Matches for a single round
# ---------------------------------------------------------------------------

def probe_matches(auth_header: dict, base_url: str, round_id: int) -> list:
    """
    Inspect matches under a single round to see what shape they have.

    Args:
        auth_header (dict): Auth header from authorize().
        base_url (str): e.g. "https://aiarena.net/api".
        round_id (int): Round ID to inspect.

    Returns:
        list of match dicts (first page).
    """
    banner(f"PROBE: matches for round {round_id}")
    data = get_json(f"{base_url}/matches/", auth_header, params={"round": round_id})
    print(f"Matches count={data.get('count')}")

    results = data.get("results", [])
    if results:
        print("\n--- First match (all fields) ---")
        print(pretty(results[0]))
    save_json(f"matches_round{round_id}", data)
    return results


# ---------------------------------------------------------------------------
# Probe 5 — Results for a single match (this is where replay_file lives)
# ---------------------------------------------------------------------------

def probe_results(auth_header: dict, base_url: str, match_id: int) -> dict:
    """
    Fetch /results/?match=<id> to see whether a replay_file URL is present.

    `_download_single_replay` reads `result['results'][0]['replay_file']`,
    so if that slot is empty OR the results array is empty we download
    nothing. Surfacing this clearly is the whole point of this probe.

    Args:
        auth_header (dict): Auth header from authorize().
        base_url (str): e.g. "https://aiarena.net/api".
        match_id (int): Match ID to query.

    Returns:
        dict: Raw response JSON.
    """
    banner(f"PROBE: results for match {match_id}")
    data = get_json(f"{base_url}/results/", auth_header, params={"match": match_id})
    print(f"Results count={data.get('count')}")

    items = data.get("results", [])
    if not items:
        print("[WARN] No results returned — replay is not yet available for this match.")
    else:
        print("\n--- First result (all fields) ---")
        print(pretty(items[0]))
        replay_url = items[0].get("replay_file")
        print(f"\nreplay_file URL: {replay_url!r}")
        if not replay_url:
            print("[WARN] replay_file is empty/null — nothing to download for this match.")
    save_json(f"results_match{match_id}", data)
    return data


# ---------------------------------------------------------------------------
# Probe 6 — End-to-end chain on a single LIVE competition
# ---------------------------------------------------------------------------

def probe_full_chain(auth_header: dict, base_url: str) -> None:
    """
    Walk competition -> rounds -> matches -> results for the first live
    competition we can find.

    Picks the FIRST competition whose `status` looks open (status == 'open'
    is the first guess; we fall through to a looser heuristic if no such
    field exists) and then drills down exactly the way the production
    pipeline would. If every step returns data, the pipeline should work on
    that competition — and we've identified a concrete known-good target
    for tightening the filter in _query_full_game_competition_ids.

    Calls:
        probe_competitions — gets the full list so we can pick a live one.
        probe_rounds / probe_matches / probe_results — drill down.

    Args:
        auth_header (dict): Auth header from authorize().
        base_url (str): e.g. "https://aiarena.net/api".
    """
    banner("PROBE: end-to-end chain (competition -> rounds -> matches -> results)")

    # Cheap reuse: this also dumps the status table and saves JSON.
    comps = probe_competitions(auth_header, base_url)

    # Try to find a genuinely open competition by a few candidate heuristics.
    # We record WHICH heuristic matched so the user sees how we picked.
    live_comp = None
    for c in comps:
        if str(c.get("status", "")).lower() == "open":
            live_comp = c
            reason = "status == 'open'"
            break
    if live_comp is None:
        for c in comps:
            # Fallback: no close date set yet -> still running.
            if not c.get("date_closed"):
                live_comp = c
                reason = "date_closed is falsy"
                break
    if live_comp is None and comps:
        # Last resort: just use the first one so we can still see the shape.
        live_comp = comps[0]
        reason = "first competition in list (no live detector matched)"

    if live_comp is None:
        print("[ERR] No competitions at all — cannot continue the chain.")
        return

    print(f"\nPicked competition id={live_comp['id']} name={live_comp.get('name')!r}  ({reason})")

    rounds = probe_rounds(auth_header, base_url, live_comp["id"])
    if not rounds:
        print("[STOP] No completed rounds on chosen competition — chain halts.")
        return

    first_round_id = rounds[0]["id"]
    matches = probe_matches(auth_header, base_url, first_round_id)
    if not matches:
        print("[STOP] Round has no matches — chain halts.")
        return

    first_match_id = matches[0]["id"]
    probe_results(auth_header, base_url, first_match_id)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with a subcommand-like --probe flag."""
    p = argparse.ArgumentParser(
        description="Diagnostic probes against the aiarena.net REST API.",
    )
    p.add_argument(
        "--probe",
        choices=["all", "competitions", "filters", "rounds", "matches", "results", "chain"],
        default="all",
        help="Which probe to run (default: all).",
    )
    p.add_argument("--comp-id", type=int, help="Competition ID for --probe rounds.")
    p.add_argument("--round-id", type=int, help="Round ID for --probe matches.")
    p.add_argument("--match-id", type=int, help="Match ID for --probe results.")
    return p


def main() -> None:
    """
    Parse CLI args, fetch auth, and dispatch to the requested probe(s).

    Calls:
        authorize — reads AIARENA_TOKEN and AIARENA_NET_URL from the .env
                    at the project root. If either is missing, every API
                    call will 401/403 — we surface that early.
    """
    args = build_arg_parser().parse_args()

    auth_header, base_url = authorize()
    if not base_url or not auth_header.get("Authorization", "").replace("Token ", "").strip():
        print("[FATAL] AIARENA_TOKEN or AIARENA_NET_URL missing from .env — aborting.")
        sys.exit(2)

    # Strip trailing slash so our own f-strings can always add one.
    base_url = base_url.rstrip("/")
    print(f"Using base_url = {base_url}")
    print(f"Output dir     = {OUTPUT_DIR}")

    probe = args.probe
    if probe in ("all", "competitions"):
        probe_competitions(auth_header, base_url)
    if probe in ("all", "filters"):
        probe_competition_filters(auth_header, base_url)
    if probe == "rounds":
        if args.comp_id is None:
            print("[ERR] --probe rounds requires --comp-id")
            sys.exit(2)
        probe_rounds(auth_header, base_url, args.comp_id)
    if probe == "matches":
        if args.round_id is None:
            print("[ERR] --probe matches requires --round-id")
            sys.exit(2)
        probe_matches(auth_header, base_url, args.round_id)
    if probe == "results":
        if args.match_id is None:
            print("[ERR] --probe results requires --match-id")
            sys.exit(2)
        probe_results(auth_header, base_url, args.match_id)
    if probe in ("all", "chain"):
        probe_full_chain(auth_header, base_url)


if __name__ == "__main__":
    main()
