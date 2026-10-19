"""Resolve the DrQA checkpoint and metadata files.

By default both files are pulled from the HuggingFace Hub into the shared
``huggingface_hub`` cache. Pass explicit paths to use local files instead.
"""
from __future__ import annotations

DEFAULT_MODEL_REPO = "Jarbas/DrQA_en"
DEFAULT_MODEL_FILE = "drqa_en_v1.1.pt"
DEFAULT_META_FILE = "meta.msgpack"


def download_model(model_path: str | None = None,
                   meta_path: str | None = None,
                   repo_id: str = DEFAULT_MODEL_REPO,
                   model_file: str = DEFAULT_MODEL_FILE,
                   meta_file: str = DEFAULT_META_FILE) -> tuple[str, str]:
    """Return local paths for the model checkpoint and metadata file.

    Any path already supplied is used as-is; the missing one is downloaded
    from the HuggingFace Hub through the shared cache.
    """
    from huggingface_hub import hf_hub_download

    if not model_path:
        model_path = hf_hub_download(repo_id=repo_id, filename=model_file)
    if not meta_path:
        meta_path = hf_hub_download(repo_id=repo_id, filename=meta_file)
    return model_path, meta_path
