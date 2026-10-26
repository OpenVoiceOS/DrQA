# TODO — DrQA

## Open issues

GitHub issues are disabled on this repository, so the backlog is tracked here
and in the pull-request queue.

## Gaps

- [ ] No published release yet. The first `ovos-drqa-plugin` alpha goes out
      when the packaging branch merges to `dev`.
- [ ] The `network` tests (checkpoint download plus inference) do not run in
      CI; they are deselected by default and exercised by hand.
- [ ] `drqa/interact.py` (the `drqa-interact` REPL) has no automated test.
- [ ] The retriever half of `DrQA` (TF-IDF over a Wikipedia corpus) is kept for
      completeness but is not part of the shipped extractive-QA engine; it has
      no test and no documented entry point.
- [ ] The upstream training path (`train.py`, `prepro.py`, `dawnbench.md`)
      still targets the pre-fork workflow and is untested.

## Code TODOs

- `drqa/utils.py` — the spaCy tokenizer loads `en_core_web_md`; a missing
  model raises spaCy's own error. Give it a clear message that names the
  `python -m spacy download en_core_web_md` step.
