from __future__ import annotations

import torch
from ovos_plugin_manager.templates.agents import ExtractiveQAEngine
from ovos_utils.log import log_deprecation

from drqa import DrQA
from drqa.download import DEFAULT_MODEL_REPO, download_model


class DrQAExtractiveQAEngine(ExtractiveQAEngine):
    """Extractive QA engine backed by a trained DrQA document reader.

    Config keys
    -----------
    model_path : str
        local checkpoint override — skips the HuggingFace Hub download.
    meta_path : str
        local ``meta.msgpack`` override — skips the HuggingFace Hub download.
    model_repo : str
        HuggingFace repo id (default: ``Jarbas/DrQA_en``).
    cuda : bool
        run on GPU (default: available).
    lang : str
        language (DrQA only ships an English model, default: ``en``).
    """

    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        config.setdefault("lang", "en")
        super().__init__(config)
        model_path, meta_path = download_model(
            self.config.get("model_path"),
            self.config.get("meta_path"),
            self.config.get("model_repo", DEFAULT_MODEL_REPO),
        )
        self.dr = DrQA(model_path=model_path, meta_path=meta_path,
                       cuda=self.config.get("cuda", torch.cuda.is_available()))

    def get_best_passage(self, evidence: str, question: str,
                         lang: str | None = None) -> str:
        """Return the span of ``evidence`` that best answers ``question``."""
        if not evidence or not question:
            return ""
        return self.dr.predict(evidence, question)


class DrQASolver(DrQAExtractiveQAEngine):
    """Deprecated alias kept for code importing the pre-engine name."""

    def __init__(self, config: dict | None = None):
        log_deprecation("DrQASolver is deprecated, use DrQAExtractiveQAEngine",
                        "1.0.0")
        super().__init__(config)
