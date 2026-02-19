import hashlib
import os
import shutil
import urllib.request
from functools import cache
from pathlib import Path
from typing import Final

STAMP_CACHE_DIR: Final[Path] = (
    Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "stamp"
)
# Directory is created on demand (inside functions that write to it)
# so that a bare import of this module does not cause filesystem I/O.


def download_file(*, url: str, file_name: str, sha256sum: str) -> Path:
    """Downloads a file, or loads it from cache if it has been downloaded before.

    The checksum is only verified on the initial download.  Once the file
    exists in the cache it is trusted as-is to avoid re-reading large weight
    files (which can be ~1 GB) on every run.
    """
    STAMP_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    outfile_path = STAMP_CACHE_DIR / file_name
    if outfile_path.is_file():
        # File already cached and verified on first download — skip re-hash.
        return outfile_path

    filename, _ = urllib.request.urlretrieve(url)
    with open(filename, "rb") as weight_file:
        digest = hashlib.file_digest(weight_file, "sha256")
    assert digest.hexdigest() == sha256sum, "hash of downloaded file did not match"
    shutil.move(filename, outfile_path)
    return outfile_path


def file_digest(file: str | Path) -> str:
    with open(file, "rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


@cache
def get_processing_code_hash(file_path: Path) -> str:
    """The hash of the entire process codebase.

    It is used to assure that features extracted with different versions of
    this code base can be identified as such after the fact.
    """
    hasher = hashlib.sha256()
    for py_file in sorted(file_path.parent.glob("*.py")):
        # Use file_digest to stream the file in chunks instead of reading
        # the entire source into memory at once.
        with open(py_file, "rb") as fp:
            hasher.update(hashlib.file_digest(fp, "sha256").digest())
    return hasher.hexdigest()
