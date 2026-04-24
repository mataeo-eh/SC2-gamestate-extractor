import logging
import requests
from dotenv import load_dotenv
import os
import pprint
from tqdm import tqdm
import json
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv()

# Module-level logger — handlers are configured by setup_logging() in quickstart.py
# before any of these functions are called, so log output goes to both the
# timestamped log file (DEBUG+) and the console (INFO+) automatically.
logger = logging.getLogger(__name__)


def authorize():
    token = os.getenv("AIARENA_TOKEN")
    base_url = os.getenv("AIARENA_NET_URL")

    # Set up auth header with your token
    auth = {'Authorization': f'Token {token}'}
    return auth, base_url

def get_bot_ids_by_names(auth, base_url, bot_names: list, print_output: bool = True):
    """
    Finds and returns bot IDs for all given bot names in a single paginated pass.

    Rather than making one API pass per bot name, this function checks every
    remaining name on each page, so all names are resolved in one traversal.

    Args:
        auth: Authorization header
        base_url: Base URL for the API
        bot_names (list): List of bot name strings to look up
    Kwargs:
        print_output (bool): Whether to print found IDs (default = True)
    Returns:
        dict mapping each found bot name to its integer ID.
        Names that were not found are omitted from the result.
    """
    # Use a set so we can cheaply remove names once they are found
    remaining = set(bot_names)
    found = {}  # {bot_name: bot_id}

    url = f"{base_url}/bots/"
    pbar = None
    while url and remaining:
        response = requests.get(url, headers=auth)
        response.raise_for_status()
        data = response.json()

        # Initialize progress bar once we know the total
        if pbar is None:
            total = data.get("count")
            pbar = tqdm(total=total, desc="Fetching bots", unit="bots")

        # Check every bot on this page against every still-unresolved name
        for bot in data["results"]:
            bot_api_name = bot.get("name", "").lower()
            for name in list(remaining):  # list() so we can mutate remaining mid-loop
                if name.lower() in bot_api_name:
                    found[name] = bot["id"]
                    remaining.discard(name)
                    if print_output:
                        print(f"ID found for {name}: {bot['id']}")

        pbar.update(len(data["results"]))
        url = data.get("next")  # Move to next page

    if remaining and print_output:
        for name in remaining:
            print(f"Warning: no bot found matching '{name}'")

    return found


def fetch_bot_match_ids(auth, base_url, bot_ids: list, max_replays: int = None, print_output: bool = True):
    """ 
    Fetches a list of matches for a specific bot from the AI Arena API 
    Args: 
        auth: Authorization header
        base_url: Base URL for the API
        bot_ids (list): List of bot IDs to fetch matches for
    Kwargs:
        max_replays (int): Maximum number of matches to fetch (default = None)
        print_output (bool): Whether to print the output (default = True)
    Returns:
        Number of match ID's collected (int)
        List of match IDs collected (list)
    """
    match_ids = []
    # Make replays directory if it doesn't exist
    os.makedirs('replays', exist_ok=True)
    # Iterate over bot IDs
    for bot_id in tqdm(bot_ids, desc="Processing bots"):
        # Reset replay counter for each bot
        bot_match_count = 0
        # Get matches for a given bot id, ordered by newest first so that
        # max_replays fetches the most recent matches rather than the oldest.
        url = f'{base_url}/match-participations/?bot={bot_id}&ordering=-id'
        pbar = None
        while url and (max_replays is None or bot_match_count < max_replays):
            response = requests.get(url, headers=auth)
            response.raise_for_status()
            matches = response.json()

            # Initialize pbar once
            if pbar is None:
                total = matches.get("count")
                pbar = tqdm(total=total, desc="Fetching match ID's", unit="matches")
            
            # Iterate over matches to get each ID
            for match in matches['results']:
                # Add each match ID to the list of Id's
                match_ids.append(match['match'])
                bot_match_count += 1 # Increment the per-bot replay counter
                if max_replays and bot_match_count >= max_replays:
                    break  # Stop once we hit the limit

            # Update progress after processing each page
            pbar.update(len(matches['results']))

            url = matches.get('next') # Fetch the next page of matches

    if print_output:
        print(f"Total match IDs fetched: {len(match_ids)}")
        pprint.pprint(match_ids)
    return len(match_ids), match_ids




import time

def _download_single_replay(auth, base_url, match_id, print_output: bool = True, max_retries: int = 3):
    """
    Downloads a single replay for a given match ID. Designed to run inside a
    ThreadPoolExecutor worker. All I/O is network-bound so threads work well
    despite the GIL.

    Args:
        auth: Authorization header
        base_url: Base URL for the API
        match_id: Match ID to download the replay for
    Kwargs:
        print_output (bool): Whether to print the output (default = True)
        max_retries (int): Maximum number of retry attempts for recoverable errors (default = 3)
    Returns:
        Tuple of (match_id, success: bool, error_reason: str or None)
    """
    # Check if replay has already been downloaded
    filename = f'replays/match_{match_id}.SC2Replay'
    if os.path.exists(filename):
        return (match_id, True, None)  # Already exists, count as success

    for attempt in range(max_retries):
        try:
            # Get info for each match
            response = requests.get(f'{base_url}/results/?match={match_id}', headers=auth)

            # Handle recoverable errors with retry
            if response.status_code in [429, 500, 502, 503]:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
                continue

            # Handle non-recoverable client errors
            if response.status_code == 404:
                return (match_id, False, "404 Not Found")

            if 400 <= response.status_code < 500:
                return (match_id, False, f"HTTP {response.status_code}")

            response.raise_for_status()

            # Parse JSON with error handling
            try:
                result = response.json()
            except requests.exceptions.JSONDecodeError:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return (match_id, False, "Invalid JSON")

            # Get replay URL — results may be empty if the match has no replay yet
            if not result.get('results'):
                return (match_id, False, "No results returned (replay not available)")
            replay_url = result['results'][0]['replay_file']

            if replay_url:
                # Stream the replay download in chunks to reduce peak memory
                # when many downloads run concurrently.
                replay_response = requests.get(replay_url, stream=True)
                replay_response.raise_for_status()

                with open(filename, 'wb') as f:
                    for chunk in replay_response.iter_content(chunk_size=8192):
                        f.write(chunk)

                return (match_id, True, None)
            else:
                return (match_id, False, "No replay URL")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return (match_id, False, str(e))

    return (match_id, False, f"Max retries ({max_retries}) exceeded")


def download_replays(auth, base_url, match_ids: list, print_output: bool = True,
                     max_retries: int = 3, max_workers: int = 8):
    """
    Downloads replays for given match IDs using concurrent threads.

    Downloads are I/O-bound (network), so ThreadPoolExecutor provides true
    concurrency despite the GIL. Each worker handles one match ID with retry
    logic and streaming file writes.

    Args:
        auth: Authorization header
        base_url: Base URL for the API
        match_ids (list): List of match IDs to download replays for
    Kwargs:
        print_output (bool): Whether to print the output (default = True)
        max_retries (int): Maximum number of retry attempts per match (default = 3)
        max_workers (int): Number of concurrent download threads (default = 8).
                           Higher values increase throughput but use more memory
                           and connections. 8 is a good balance for most APIs.
    Returns:
        Number of replays downloaded (int)
    """
    num_replays = 0
    failed_matches = []

    # Filter out already-downloaded replays before submitting to thread pool
    pending_ids = [
        mid for mid in match_ids
        if not os.path.exists(f'replays/match_{mid}.SC2Replay')
    ]
    skipped = len(match_ids) - len(pending_ids)
    if skipped and print_output:
        print(f"Skipping {skipped} already-downloaded replays.")

    if not pending_ids:
        if print_output:
            print("All replays already downloaded.")
        return 0

    # Download replays concurrently using a thread pool.
    # ThreadPoolExecutor releases the GIL during I/O (network, disk), so
    # multiple threads download truly in parallel.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_match = {
            executor.submit(
                _download_single_replay, auth, base_url, mid,
                print_output=False, max_retries=max_retries,
            ): mid
            for mid in pending_ids
        }

        for future in tqdm(as_completed(future_to_match), total=len(pending_ids),
                           desc="Downloading replays", leave=False):
            match_id, success, error = future.result()
            if success:
                num_replays += 1
            else:
                failed_matches.append((match_id, error))

    # Report failures at the end
    if failed_matches and print_output:
        print(f"\n{'='*50}")
        print(f"Failed to download {len(failed_matches)} replays:")
        for match_id, reason in failed_matches:
            print(f"  - Match {match_id}: {reason}")
        print(f"{'='*50}\n")

    return num_replays

# ---------------------------------------------------------------------------
# Ladder-wide replay fetching (no specific bot required)
# ---------------------------------------------------------------------------

def _load_ladder_cache(cache_path: str) -> dict:
    """
    Load the Macro_ladder.json competition ID cache from disk.

    The cache is a dict of {date_string: [comp_id, ...]} that persists
    previously discovered full-game competition IDs across runs so the
    script avoids a full API scan on every invocation.

    Args:
        cache_path (str): Path to Macro_ladder.json
    Returns:
        dict: Cached competition ID registry, or {} if the file is absent/empty/corrupt
    """
    path = Path(cache_path)
    if not path.exists():
        logger.debug("Ladder cache not found at %s — starting with empty cache.", cache_path)
        return {}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(
                "Ladder cache at %s contained unexpected type %s — treating as empty.",
                cache_path, type(data).__name__
            )
            return {}
        logger.debug("Loaded ladder cache from %s (%d date entries).", cache_path, len(data))
        return data
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning(
            "Failed to read ladder cache at %s (%s) — treating as empty.", cache_path, exc
        )
        return {}


def _save_ladder_cache(cache_path: str, data: dict) -> None:
    """
    Persist the Macro_ladder.json competition ID cache to disk.

    Args:
        cache_path (str): Path to Macro_ladder.json
        data (dict): {date_string: [comp_id, ...]} registry to write
    """
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.debug("Ladder cache saved to %s.", cache_path)


def _query_full_game_competition_ids(auth, base_url, print_output: bool = True) -> list:
    """
    Query the aiarena API for currently-OPEN full-game competitions.

    Filters applied (a competition must satisfy ALL of these):
      - status == "open"    — only the currently-live season yields fresh
                              replays. Closed seasons remain reachable via
                              HTTP 200 but their S3 replay files have been
                              cleaned, so every download would fail silently.
      - game_mode == 1      — 1 is full-game (macro), 2 is micro. Filtering
                              on game_mode is more reliable than name-matching
                              on "micro" (which can break if naming changes).

    In practice aiarena runs exactly one macro ladder at a time, so this
    normally returns a list of length 1. Returning a list (not a single ID)
    keeps the caller's loop structure unchanged if that ever changes.

    Calls:
        authorize() — caller is responsible for passing auth and base_url
        requests.get — paginates through /api/competitions/

    Args:
        auth: Authorization header dict
        base_url (str): Base aiarena API URL
        print_output (bool): Whether to log at INFO (True) or DEBUG (False) level (default True)
    Returns:
        list: Integer competition IDs for all open full-game competitions
    """
    # Helper so callers can suppress INFO-level noise when running non-interactively.
    log = logger.info if print_output else logger.debug

    url = f"{base_url}/competitions/"
    comp_ids = []
    pbar = None

    log("Querying aiarena API for OPEN full-game competitions (status='open', game_mode=1)...")

    while url:
        response = requests.get(url, headers=auth)
        response.raise_for_status()
        data = response.json()

        if pbar is None:
            total = data.get("count", 0)
            pbar = tqdm(total=total, desc="Scanning competitions", unit="comp",
                        disable=not print_output)

        for comp in data["results"]:
            name = comp.get("name", "")
            status = comp.get("status")
            game_mode = comp.get("game_mode")

            if status == "open" and game_mode == 1:
                comp_ids.append(comp["id"])
                logger.debug(
                    "Accepted competition: '%s' (id=%s, status=%s, game_mode=%s)",
                    name, comp["id"], status, game_mode,
                )
            else:
                logger.debug(
                    "Skipped competition: '%s' (id=%s, status=%s, game_mode=%s)",
                    name, comp["id"], status, game_mode,
                )

        pbar.update(len(data["results"]))
        url = data.get("next")

    if pbar:
        pbar.close()

    log("Found %d open full-game competition(s).", len(comp_ids))
    return comp_ids


def _validate_competition_id(auth, base_url, comp_id: int) -> bool:
    """
    Check whether a competition ID is currently OPEN (live) via the API.

    Closed competitions still return HTTP 200 from the detail endpoint, so
    a plain status-code check is not enough to identify a usable comp:
    their S3 replay files have been cleaned and the download pipeline would
    silently collect zero replays. This function requires the competition
    to be reachable AND have status == "open" before treating it as valid.

    Calls:
        requests.get — hits /api/competitions/<comp_id>/

    Args:
        auth: Authorization header dict
        base_url (str): Base aiarena API URL
        comp_id (int): Competition ID to check
    Returns:
        bool: True if the competition exists and has status == "open", False otherwise
    """
    try:
        response = requests.get(f"{base_url}/competitions/{comp_id}/", headers=auth)
        if response.status_code != 200:
            logger.debug(
                "Competition ID %s returned HTTP %s — treating as invalid.",
                comp_id, response.status_code
            )
            return False

        # HTTP 200 alone is not sufficient: historical seasons are reachable
        # but no longer serve replays. Require status=='open' so we never
        # spend the pipeline's max_replays budget chasing cleaned-up matches.
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            logger.warning(
                "Competition ID %s returned non-JSON body (%s) — treating as invalid.",
                comp_id, exc
            )
            return False

        status = data.get("status")
        if status == "open":
            logger.debug("Competition ID %s is valid (status='open').", comp_id)
            return True
        logger.debug(
            "Competition ID %s is not open (status=%r) — treating as invalid.",
            comp_id, status
        )
        return False
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "Request error while validating competition ID %s: %s", comp_id, exc
        )
        return False


def fetch_ladder_match_ids(auth, base_url, max_replays: int, cache_path: str,
                           print_output: bool = True) -> tuple:
    """
    Collect match IDs from the full-game ladder without targeting specific bots.

    Uses Macro_ladder.json to avoid a full competition scan on every run:

    Cache resolution order:
      1. Load the cache; sort date keys descending (most recent first).
      2. For each date entry, validate each stored competition ID.
      3. Use the valid IDs from the most-recent date that has at least one valid ID.
      4. If every date entry is exhausted (all IDs invalid or cache is empty):
         query the API for fresh competition IDs, append them to the cache under
         today's date, and save. Old entries are kept for auditing.

    Match collection:
      For each valid competition, fetch completed rounds ordered most-recent first
      (?complete=true&ordering=-started), then fetch all matches per round. Stop
      as soon as max_replays match IDs have been accumulated.

    Calls:
        _load_ladder_cache — reads Macro_ladder.json
        _validate_competition_id — checks each cached ID against the API
        _query_full_game_competition_ids — full API scan when cache is stale
        _save_ladder_cache — persists updated cache to disk
        requests.get — paginates /api/rounds/ and /api/matches/

    Args:
        auth: Authorization header dict
        base_url (str): Base aiarena API URL
        max_replays (int): Maximum number of match IDs to collect (None = unlimited)
        cache_path (str): Path to Macro_ladder.json
        print_output (bool): Whether to log at INFO (True) or DEBUG (False) level (default True)
    Returns:
        Tuple of (count: int, match_ids: list) — same shape as fetch_bot_match_ids
    """
    # Route messages to INFO (visible on console) or DEBUG (file only) based on
    # print_output, mirroring how the bot pipeline uses the print_output flag.
    log = logger.info if print_output else logger.debug

    # --- Step 1: Resolve valid competition IDs ---
    cache = _load_ladder_cache(cache_path)
    valid_comp_ids = []

    if cache:
        # Work through date keys newest-first; stop at the first date
        # that yields at least one still-valid competition ID.
        sorted_dates = sorted(cache.keys(), reverse=True)
        for date_key in sorted_dates:
            candidate_ids = cache[date_key]
            logger.debug(
                "Validating %d cached competition ID(s) from %s...",
                len(candidate_ids), date_key
            )
            validated = [
                cid for cid in candidate_ids
                if _validate_competition_id(auth, base_url, cid)
            ]
            if validated:
                valid_comp_ids = validated
                log(
                    "Using %d valid cached competition ID(s) from %s.",
                    len(valid_comp_ids), date_key
                )
                break
            else:
                logger.warning(
                    "All competition IDs from %s are no longer valid — checking older entries.",
                    date_key
                )

    # Cache was empty or every entry had no reachable IDs — query fresh.
    if not valid_comp_ids:
        logger.warning(
            "No valid cached competition IDs found. Querying API for full-game competitions..."
        )
        valid_comp_ids = _query_full_game_competition_ids(auth, base_url, print_output=print_output)
        if not valid_comp_ids:
            logger.error("No full-game competitions found on aiarena — cannot fetch ladder replays.")
            raise RuntimeError(
                "No full-game competitions found on aiarena. Cannot fetch ladder replays."
            )

        # Append new IDs to the cache under today's date.
        # If today already has an entry (e.g. two runs on the same day),
        # merge and deduplicate while preserving order.
        today = datetime.date.today().isoformat()  # e.g. "2026-04-19"
        existing_today = cache.get(today, [])
        # dict.fromkeys preserves insertion order and removes duplicates
        merged = list(dict.fromkeys(existing_today + valid_comp_ids))
        cache[today] = merged
        _save_ladder_cache(cache_path, cache)
        log("Saved %d competition ID(s) to cache under %s.", len(valid_comp_ids), today)

    # --- Step 2: Collect match IDs from recent completed rounds ---
    os.makedirs('replays', exist_ok=True)
    match_ids = []

    for comp_id in valid_comp_ids:
        if max_replays is not None and len(match_ids) >= max_replays:
            break

        log("Fetching completed rounds for competition %s...", comp_id)

        # Request only completed rounds so every match inside has a replay.
        # ordering=-started puts the most recently started rounds first.
        rounds_url = (
            f"{base_url}/rounds/"
            f"?competition={comp_id}&complete=true&ordering=-started"
        )

        while rounds_url and (max_replays is None or len(match_ids) < max_replays):
            response = requests.get(rounds_url, headers=auth)
            response.raise_for_status()
            rounds_data = response.json()

            for round_obj in tqdm(rounds_data["results"], desc="Rounds", unit="round",
                                  disable=not print_output, leave=False):
                if max_replays is not None and len(match_ids) >= max_replays:
                    break

                round_id = round_obj["id"]
                logger.debug("Fetching matches for round %s...", round_id)
                matches_url = f"{base_url}/matches/?round={round_id}"

                while matches_url and (max_replays is None or len(match_ids) < max_replays):
                    match_response = requests.get(matches_url, headers=auth)
                    match_response.raise_for_status()
                    matches_data = match_response.json()

                    for match in matches_data["results"]:
                        match_ids.append(match["id"])
                        if max_replays is not None and len(match_ids) >= max_replays:
                            break

                    matches_url = matches_data.get("next")

            rounds_url = rounds_data.get("next")

    log("Total match IDs fetched from ladder: %d", len(match_ids))
    return len(match_ids), match_ids


def main_ladder(max_replays: int = None, cache_path: str = None, print_output: bool = True) -> None:
    """
    Entry point for fetching and downloading replays from the full-game ladder.

    Mirrors the structure of main() but targets the entire ladder instead of
    specific bots. Competition IDs are resolved via Macro_ladder.json so the
    API is only scanned when the cache is empty or all cached IDs are stale.

    Calls:
        authorize — reads AIARENA_TOKEN and AIARENA_NET_URL from environment
        fetch_ladder_match_ids — resolves competition IDs and collects match IDs
        download_replays — downloads the replay files (shared with bot pipeline)

    Args:
        max_replays (int): Maximum number of replays to download (default None = all)
        cache_path (str): Path to Macro_ladder.json. Defaults to
                          src_new/utils/Macro_ladder.json relative to this file.
        print_output (bool): Whether to log at INFO (True) or DEBUG (False) level (default True)
    """
    log = logger.info if print_output else logger.debug

    if cache_path is None:
        # Resolve the default cache path relative to this module's location
        # so it works regardless of the working directory the script is run from.
        cache_path = Path(__file__).parent.parent / "utils" / "Macro_ladder.json"

    logger.info("Starting ladder replay download pipeline (max_replays=%s).", max_replays)
    try:
        auth, url = authorize()
        _, match_ids = fetch_ladder_match_ids(
            auth, url, max_replays, str(cache_path), print_output=print_output
        )
        log("Finished fetching ladder match IDs.")
        if match_ids:
            num_replays = download_replays(auth, url, match_ids, print_output=print_output)
            logger.info("Ladder pipeline complete. Total replays downloaded: %d", num_replays)
        else:
            logger.warning("No match IDs were collected — no replays downloaded.")
    except Exception as e:
        logger.error("Ladder replay download pipeline failed: %s", e, exc_info=True)
        raise RuntimeError(f"Error in ladder replay download pipeline: {e}")


def main(bots: list, print_output = True, max_replays=None):
    """ 
    Main function to fetch and download bot replays 
    Args: 
        bots (list): List of bot names to fetch replays for
    Kwargs:
        print_output (bool): Whether to print the output (default = True)
        max_replays (int): Maximum number of replays to download per bot (default = None)
    """
    try:
        # Authorize API usage for this session
        auth, url = authorize()

        # Single paginated pass finds all bot IDs simultaneously
        bot_ids = list(get_bot_ids_by_names(auth, url, bots, print_output=print_output).values())

        # Fetch and download matches for the bot IDs
        match_ids = fetch_bot_match_ids(auth, url, bot_ids, max_replays=max_replays, print_output=print_output)[1]
        if print_output:
            print("Finished fetching match IDs.")
        if match_ids:
            # Download replays for the fetched match IDs
            num_replays = download_replays(auth, url, match_ids, print_output=print_output)
    except Exception as e:
        raise RuntimeError(f"Error in replay download pipeline: {e}")
    if print_output:
        print(f"Total replays downloaded: {num_replays}")



if __name__ == "__main__":
    #bots = ["really","why","what"]
    #main(bots, max_replays = 1, print_output=True)
    #auth, base_url = authorize()
    #bots = fetch_bot_id(auth, base_url, bot_name="really")
    #bots = fetch_bots_list(auth, base_url, max=10, bot="really")
    #print(bots)
    #fetch_bot_match_ids(auth, base_url, bot_ids=[934], print_output=True)
    pass
