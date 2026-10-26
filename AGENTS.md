# AGENTS.md

Conventions for AI coding agents (internal and community) working in this
repository.

## What this repo is

An OpenVoiceOS fork of Facebook Research's DrQA, the PyTorch implementation of
*Reading Wikipedia to Answer Open-Domain Questions*. A trained document reader
answers a question by returning the span of a supplied evidence passage that
answers it (extractive question answering).

The shippable surface is the OVOS plugin: `drqa/opm.py` defines
`DrQAExtractiveQAEngine`, registered under the `opm.agents.extractive_qa`
entry-point group and distributed as `ovos-drqa-plugin`. The reader itself
lives in `drqa/model.py`, `drqa/rnn_reader.py`, `drqa/layers.py` and
`drqa/utils.py`. `train.py`, `prepro.py` and `dawnbench.md` are the inherited
upstream training path, kept for reference.

## Ground rules

- Work on a feature branch. Never push to `dev` or `master` directly.
- Open pull requests against `dev` as **drafts** until CI is green.
- One commit per PR. Squash before pushing if history accumulates.
- Use conventional commit prefixes (`feat:`, `fix:`, `refactor:`, `docs:`,
  `test:`, `chore:`). Reserve `feat:` for changes a user or downstream
  consumer can observe.
- Never hand-edit the `VERSION_MAJOR`/`MINOR`/`BUILD`/`ALPHA` block in
  `drqa/version.py`. CI computes and bumps the version from conventional
  commit history.
- Every PR and issue you write carries an AI-authorship disclosure at the top
  that names the exact model used and states the text is not human-reviewed.

## Dependencies

- Use `uv`, never `pip`, for installing and resolving dependencies.
- All dependency and metadata declarations live in `pyproject.toml`
  (`dependencies` for runtime, `[project.optional-dependencies].test` for the
  test extra). Nothing dependency-related belongs in a workflow file.
- Pin floors only, and allow prereleases (`>=X.Y.Za1`). Never install a
  dependency from a git URL; publish an alpha to PyPI and depend on that.
- The build-test matrix is the gh-automations default, Python 3.10 to 3.14.
  Torch publishes wheels for each of those versions.
- The spaCy English model (`en_core_web_md`, matching the pinned spaCy 3.x) is
  installed separately; it is what the tokenizer loads.

## Testing

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[test]"
python -m spacy download en_core_web_md
pytest
```

`pytest` runs the offline tests. The `network` marker covers the tests that
download the checkpoint from Hugging Face and run inference; ask for them with
`pytest -m network`. Default runs deselect them.

A regression test for a bug must be shown to fail against the code before the
fix and pass after it. A test that passes against unfixed code proves nothing.

## Repo-specific notes

- The reader is legacy ParlAI-derived code. When you touch model code, keep the
  modern PyTorch style already in place: packed sequences with
  `enforce_sorted=False` and a `total_length`, `torch.outer` in place of the
  removed `torch.ger`, `.detach()` in place of `.data`, and
  `torch.load(..., weights_only=False)` for the legacy checkpoint format.
- Model weights and `meta.msgpack` are downloaded through `huggingface_hub`'s
  shared cache (`drqa/download.py`), never into a per-app directory. The
  default repo is `Jarbas/DrQA_en`.
- Only an English reader ships (`lang` defaults to `en`); the retriever is
  TF-IDF and is documented for reference, not wired into the plugin.
