from pathlib import Path
from icecream import ic
import shelve
import os
import sys
import re
import time

file_store = None
memoize_shelve = None


class ValueMeta:
    def __init__(self):
        self.value = None
        self.last_written = None
        self.last_read = None
        self.write_count = 0
        self.access_count = 0


def set_file_store_path(file_store_path: str):
    global file_store, memoize_shelve
    file_store = f"{file_store_path}.shelve"

    # Create parent directories if they don't exist
    Path(file_store).parent.mkdir(parents=True, exist_ok=True)

    memoize_shelve = shelve.open(file_store, writeback=True)


def reset_store():
    global memoize_shelve
    if memoize_shelve is not None:
        memoize_shelve.close()
    if file_store is not None:
        os.remove(f"{file_store}.dir")
        os.remove(f"{file_store}.dat")
        os.remove(f"{file_store}.bak")
    memoize_shelve = shelve.open(file_store, writeback=True)


def _create_value(value):
    val = ValueMeta()
    val.value = value
    val.last_written = time.time()
    val.last_read = time.time()
    val.write_count = 0
    val.access_count = 0
    return val


def get(key: str, default):
    global memoize_shelve
    if key in memoize_shelve:
        value = memoize_shelve[key]
        value.access_count += 1
        value.last_read = time.time()
        memoize_shelve[key] = value
        return value.value
    else:
        value = _create_value(default)
        memoize_shelve[key] = value
        memoize_shelve.sync()
        return default


def has_key(key: str) -> bool:
    global memoize_shelve
    return key in memoize_shelve


def set(key: str, data):
    global memoize_shelve
    if key in memoize_shelve:
        old_value = memoize_shelve[key]
        old_value.value = data
        old_value.last_written = time.time()
        old_value.write_count += 1
    else:
        new_value = _create_value(data)
        memoize_shelve[key] = new_value
    memoize_shelve.sync()
    return data


def delete_keys(keys: list):
    global memoize_shelve
    for key in keys:
        if key in memoize_shelve:
            del memoize_shelve[key]
    memoize_shelve.sync()


def delete_unused_keys(older_than_seconds: int) -> list[str]:
    global memoize_shelve

    cleaned_keys = []

    keys = list(memoize_shelve.keys())
    for key in keys:
        value = memoize_shelve[key]
        if time.time() - value.last_read > older_than_seconds:
            del memoize_shelve[key]
            cleaned_keys.append(key)
    memoize_shelve.sync()
    return cleaned_keys


def close():
    global memoize_shelve
    if memoize_shelve is not None:
        memoize_shelve.close()
