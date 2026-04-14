import requests
from dotenv import load_dotenv
import os
import pprint
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv()


def authorize():
    token = os.getenv("AIARENA_TOKEN")
    base_url = os.getenv("AIARENA_NET_URL")

    # Set up auth header with your token
    auth = {'Authorization': f'Token {token}'}
    return auth, base_url

def get_bot_id_by_name(auth, base_url, bot_name: str, print_output: bool = True,):
    """
    Finds and returns the bot ID for a given bot name.
    Returns None if not found.
    """
    url = f"{base_url}/bots/"
    pbar = None
    while url:
        response = requests.get(url, headers=auth)
        response.raise_for_status()
        data = response.json()
        
        # Initialize progress bar once
        if pbar is None:
            total = data.get("count")
            pbar = tqdm(total=total, desc="Fetching bots", unit="bots")

        # Check each bot on this page
        for bot in data["results"]:
            if bot_name.lower() in bot.get("name", "").lower():
                if print_output:
                    print(f"ID found for {bot_name}: {bot['id']}")
                return bot["id"]  # Found it, return immediately
        
        # Update the progress bar
        pbar.update(len(data["results"]))

        url = data.get("next")  # Check next page
    
    return None  # Not found


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

            # Get replay URL
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

        bot_ids = [get_bot_id_by_name(auth, url, name, print_output=print_output) for name in bots]
        bot_ids = [id for id in bot_ids if id is not None]  # Filter out any not-found

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
    bots = ["really","why","what"]
    main(bots, max_replays = 1, print_output=True)
    #auth, base_url = authorize()
    #bots = fetch_bot_id(auth, base_url, bot_name="really")
    #bots = fetch_bots_list(auth, base_url, max=10, bot="really")
    #print(bots)
    #fetch_bot_match_ids(auth, base_url, bot_ids=[934], print_output=True)

