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


def download_asset(asset_filename, dest_path, chunk_size=1024 * 1024, _st=None):
    """Download a single asset from the GitHub release if dest_path doesn't already exist.
    Returns True if the file is present locally after this call (already-cached counts),
    False if it could not be obtained.
    `_st` is an optional streamlit module reference, used to show a progress bar in the app.
    """
    if os.path.exists(dest_path):
        return True

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
    """Ensure each of `names` (keys in REMOTE_ASSETS) is downloaded locally.
    Returns dict {name: bool_success}."""
    results = {}
    for name in names:
        dest = REMOTE_ASSETS.get(name)
        if dest is None:
            results[name] = False
            continue
        if _st is not None:
            with _st.spinner(f"Downloading {name} from GitHub Release (first run only)..."):
                results[name] = download_asset(name, dest, _st=_st)
        else:
            results[name] = download_asset(name, dest)
    return results
