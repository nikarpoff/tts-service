"""
Model weights on disk.

The weights live outside the image and the repository: under ARTIFACTS_PATH when it
is set (a volume in the deployment) or in the user cache. A missing file is downloaded
once; with the files in place a start needs no network.
"""

import os
import shutil

from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from loguru import logger

from app.core.config import get_project, Settings
from app.core.exceptions import BackendError

DOWNLOAD_TIMEOUT_S = 60             # per socket operation, not for the whole file
HUGGINGFACE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"


def weights_dir(settings: Settings, backend: str) -> Path:
    """Directory of the backend weights: under ARTIFACTS_PATH or in the user cache."""
    if settings.ARTIFACTS_PATH is not None:
        return settings.ARTIFACTS_PATH / backend
    cache_home = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache_home) / get_project().name / backend


def huggingface_url(repo: str, path: str, revision: str = "main") -> str:
    return HUGGINGFACE_URL.format(repo=repo, revision=revision, path=path)


def fetch(url: str, target: Path) -> Path:
    """The file at `target`, downloaded from `url` when it is not there yet."""
    if target.is_file() and target.stat().st_size > 0:
        return target

    logger.info("Downloading '{}' to '{}'", url, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A partial download never passes for the file: it is renamed only when complete
    partial = target.with_name(target.name + ".part")
    try:
        request = Request(url, headers={"User-Agent": "tts-service"})
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response, open(partial, "wb") as output:
            shutil.copyfileobj(response, output)
    except (URLError, OSError) as e:
        partial.unlink(missing_ok=True)
        raise BackendError(f"Weights are not at '{target}' and cannot be downloaded from '{url}': {e}") from e

    partial.replace(target)
    return target
