# Prerelease quirks

Newest entry first, stamped with the version that introduced it. The list
resets at the first stable release, where it becomes that release's upgrade
notes.

## 0.0.1a1

- Distribution name: `ovos-drqa-plugin`. Dependencies live in
  `pyproject.toml`; `requirements.txt` is removed from the tree.
- The package registers `opm.agents.extractive_qa` with
  `ovos-drqa-plugin = drqa.opm:DrQAExtractiveQAEngine`. That class is the
  supported name; `DrQASolver` remains importable as a deprecated alias of it
  and logs a deprecation warning.
- The checkpoint and `meta.msgpack` come from the Hugging Face repo
  `Jarbas/DrQA_en` at first use, into the shared `huggingface_hub` cache.
  Plugin config keys `model_path`, `meta_path` and `model_repo` override the
  download.
- `drqa-interact` is the console command. It takes `--model-file`,
  `--meta-file`, `--model-repo` and `--cuda`.
- The reader runs on PyTorch 2.x: the legacy checkpoint loads with
  `torch.load(..., weights_only=False)`, packed sequences are built with
  `enforce_sorted=False`, and `torch.ger` is replaced by `torch.outer`.
- The msgpack reader uses `unpack(..., raw=False)` in place of the removed
  `load(..., encoding="utf8")` argument, so `msgpack-python` is no longer
  needed.
- CI builds on Python 3.10 to 3.12 only. Torch publishes no wheels for later
  CPython versions, so a wider matrix fails at dependency resolution.
- Only an English reader ships, and `lang` defaults to `en`.
