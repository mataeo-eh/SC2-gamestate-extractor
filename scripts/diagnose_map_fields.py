"""
Diagnostic script: print every map-related field from a replay's MPQ archives.

Usage (run from SC2-gamestate-extractor directory):
    ..\.venv\Scripts\python.exe scripts\diagnose_map_fields.py path\to\match.SC2Replay
"""

import sys
from pathlib import Path

import mpyq
from s2protocol import versions as s2versions


def decode_handle(handle_bytes: bytes) -> str:
    """Return a human-readable representation of a cache handle."""
    if not handle_bytes:
        return "<empty>"
    # Try reading as UTF-8 / latin-1 for any readable substring.
    try:
        decoded = handle_bytes.decode("utf-8", errors="replace")
    except Exception:
        decoded = handle_bytes.decode("latin-1", errors="replace")
    hex_repr = handle_bytes.hex()
    return f"utf8={decoded!r}  hex={hex_repr}"


def main(replay_path: str) -> None:
    path = Path(replay_path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    print(f"\n=== Diagnosing: {path.name} ===\n")

    archive = mpyq.MPQArchive(str(path))

    # --- Header ---
    header_content = archive.header["user_data_header"]["content"]
    header = s2versions.latest().decode_replay_header(header_content)
    base_build = header["m_version"]["m_baseBuild"]
    print(f"base_build       : {base_build}")
    print(f"data_build       : {header.get('m_dataBuildNum')}")
    print(f"full version dict: {header.get('m_version')}")

    protocol = s2versions.build(base_build)

    # --- replay.details ---
    print("\n--- replay.details ---")
    try:
        details_raw = archive.read_file("replay.details")
        details = protocol.decode_replay_details(details_raw)
        for key in ("m_title", "m_mapFileName", "m_description", "m_imageFilePath"):
            val = details.get(key)
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            print(f"  {key}: {val!r}")

        # Cache handles in details
        handles = details.get("m_cacheHandles", [])
        print(f"  m_cacheHandles ({len(handles)} entries):")
        for i, h in enumerate(handles):
            print(f"    [{i}] {decode_handle(h) if isinstance(h, bytes) else h!r}")

        mod_paths = details.get("m_modPaths", [])
        print(f"  m_modPaths ({len(mod_paths)} entries):")
        for i, p in enumerate(mod_paths):
            if isinstance(p, bytes):
                p = p.decode("utf-8", errors="replace")
            print(f"    [{i}] {p!r}")

    except Exception as e:
        print(f"  ERROR reading details: {e}")

    # --- replay.initData ---
    print("\n--- replay.initData ---")
    try:
        init_raw = archive.read_file("replay.initData")
        init_data = protocol.decode_replay_initdata(init_raw)
        game_desc = init_data["m_syncLobbyState"]["m_gameDescription"]

        print(f"  game_desc keys: {list(game_desc.keys())}")

        for key in ("m_mapFileName", "m_mapAuthorName", "m_mapSha256", "m_mapFileSyncChecksum"):
            val = game_desc.get(key)
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            if val is not None:
                print(f"  {key}: {val!r}")

        handles = game_desc.get("m_cacheHandles", [])
        print(f"  m_cacheHandles ({len(handles)} entries):")
        for i, h in enumerate(handles):
            print(f"    [{i}] {decode_handle(h) if isinstance(h, bytes) else h!r}")

    except Exception as e:
        print(f"  ERROR reading initData: {e}")

    # --- replay.attributes.events (if present) ---
    print("\n--- replay.attributes.events (map attributes) ---")
    try:
        attr_raw = archive.read_file("replay.attributes.events")
        attrs = protocol.decode_replay_attributes_events(attr_raw)
        # Look for map-related attributes (attrid 1001 is often map name)
        for a in attrs.get("scopes", {}).get(16, {}).values():
            for entry in (a if isinstance(a, list) else [a]):
                val = entry.get("value", b"")
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace").rstrip("\x00")
                print(f"  attrid={entry.get('attrid')} val={val!r}")
    except Exception as e:
        print(f"  (skipped or error: {e})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_map_fields.py <replay.SC2Replay>")
        sys.exit(1)
    main(sys.argv[1])
