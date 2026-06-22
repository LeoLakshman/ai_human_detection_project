"""
Downloads large model artifacts (deep learning .h5 files, embedding files) from
a GitHub Release on first run, and caches them locally in models/ afterward.

Why: GitHub blocks files over ~25MB in a normal `git push`, and our LSTM/CNN
.h5 files are ~30MB — so they're hosted as Release assets instead of being
committed to the repo directly. This is the standard approach GitHub itself
recommends for shipping large binaries alongside a repo.

Release: https://github.com/LeoLakshman/ai_human_detection_project/releases/tag/v1.0.0
"""
import os

GITHUB_RELEASE_BASE_URL = (
    "https://github.com/LeoLakshman/ai_human_detection_project/releases/download/v1.0.0"
)
GITHUB_RELEASE_API_URL = (
    "https://api.github.com/repos/LeoLakshman/ai_human_detection_project/releases/tags/v1.0.0"
)

# Update this list if you rename assets in the release, or add more (e.g. if
# gensim split word2vec.model into word2vec.model + word2vec.model.wv.vectors.npy
# because the vocabulary was large enough to trigger gensim's separate-array storage).
REMOTE_ASSETS = {
    "fnn_model.h5": "models/fnn_model.h5",
    "lstm_model.h5": "models/lstm_model.h5",
    "cnn_model.h5": "models/cnn_model.h5",
    "word2vec.model": "models/embedding_model/word2vec.model",
    "tokenizer.json": "models/embedding_model/tokenizer.json",
}


def _remote_asset_sizes():
    """Fetch {asset_name: size_in_bytes} for the release, once per process.
    Used to detect a locally-cached file that's stale relative to the release
    (e.g. a previous deploy downloaded an old asset before this one was fixed
    and re-uploaded) so we don't keep serving it forever just because it
    already exists on disk. Returns {} on any failure (offline, rate-limited,
    etc.) — callers fall back to trusting whatever's already on disk."""
    try:
        import requests
        r = requests.get(GITHUB_RELEASE_API_URL, timeout=10)
        r.raise_for_status()
        return {a["name"]: a["size"] for a in r.json().get("assets", [])}
    except Exception:
        return {}


def download_asset(asset_filename, dest_path, chunk_size=1024 * 1024, _st=None, remote_sizes=None):
    """Download a single asset from the GitHub release. Skips the download if
    dest_path already exists AND matches the release's current size for that
    asset; re-downloads (overwrites) if the local copy is stale or undersized.
    Returns True if the file is present locally after this call, False if it
    could not be obtained.
    `_st` is an optional streamlit module reference, used to show a progress bar in the app.
    """
    if remote_sizes is None:
        remote_sizes = _remote_asset_sizes()
    expected_size = remote_sizes.get(asset_filename)

    if os.path.exists(dest_path):
        if expected_size is None or os.path.getsize(dest_path) == expected_size:
            return True
        # Local copy exists but doesn't match the release anymore — fall through and re-download.

    try:
        import requests
    except ImportError as e:
        raise ImportError(
            "The 'requests' package is required to download model artifacts. "
            "Add `requests>=2.31` to requirements.txt and redeploy."
        ) from e

    url = f"{GITHUB_RELEASE_BASE_URL}/{asset_filename}"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code == 404:
                return False
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            progress = _st.progress(0.0) if _st is not None else None
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress is not None and total:
                            progress.progress(min(downloaded / total, 1.0))
            if progress is not None:
                progress.empty()
        os.replace(tmp_path, dest_path)
        return True
    except requests.RequestException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def ensure_assets(names, _st=None):
    """Ensure each of `names` (keys in REMOTE_ASSETS) is downloaded locally and
    up to date with the release. Returns dict {name: bool_success}."""
    remote_sizes = _remote_asset_sizes()  # one API call, reused for every asset below
    results = {}
    for name in names:
        dest = REMOTE_ASSETS.get(name)
        if dest is None:
            results[name] = False
            continue
        if _st is not None:
            with _st.spinner(f"Downloading {name} from GitHub Release..."):
                results[name] = download_asset(name, dest, _st=_st, remote_sizes=remote_sizes)
        else:
            results[name] = download_asset(name, dest, remote_sizes=remote_sizes)
    return results
