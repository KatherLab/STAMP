import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Final

STAMP_CACHE_DIR: Final[Path] = (
    Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "stamp"
)

# If we imported this, we probably want to use it,
# so it's okay creating the directory now
STAMP_CACHE_DIR.mkdir(exist_ok=True, parents=True)


def download_file(*, url: str, file_name: str, sha256sum: str) -> Path:
    """Downloads a file, or loads it from cache if it has been downloaded before"""
    outfile_path = STAMP_CACHE_DIR / file_name
    if outfile_path.is_file():
        with open(outfile_path, "rb") as weight_file:
            digest = hashlib.file_digest(weight_file, "sha256")
        assert digest.hexdigest() == sha256sum, (
            f"{outfile_path} has the wrong checksum. Try deleting it and rerunning this script."
        )
    else:
        filename, _ = urllib.request.urlretrieve(url)
        with open(filename, "rb") as weight_file:
            digest = hashlib.file_digest(weight_file, "sha256")
        assert digest.hexdigest() == sha256sum, "hash of downloaded file did not match"
        shutil.move(filename, outfile_path)

    return outfile_path
