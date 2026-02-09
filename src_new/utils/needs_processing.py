def needs_processing(raw_file, processed_file):
    if not processed_file.exists():
        return True
    return raw_file.stat().st_mtime > processed_file.stat().st_mtime