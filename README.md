# DrQA

A PyTorch implementation of the ACL 2017 paper
[Reading Wikipedia to Answer Open-Domain Questions](http://www-cs.stanford.edu/people/danqi/papers/acl2017.pdf)
(DrQA), packaged as an OVOS extractive QA engine.

Reading comprehension is the task of producing an answer given a question and
one or more pieces of evidence (usually natural-language paragraphs). Compared to
question answering over knowledge bases, reading comprehension models are more
flexible and show strong zero-shot potential.

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is the reference benchmark.
DrQA is conceptually simpler than most models of its generation but still yields
strong single-model accuracy.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- numpy, msgpack, spacy 3.x, tqdm
- huggingface_hub (model download)

Install a SpaCy English model once:

```bash
python3 -m spacy download en_core_web_md
```

## Install

```bash
pip install ovos-drqa-plugin
```

## Usage

### OVOS plugin

The package registers an `opm.agents.extractive_qa` engine, discovered by
[ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager):

```python
from drqa.opm import DrQAExtractiveQAEngine

engine = DrQAExtractiveQAEngine({"model_repo": "Jarbas/DrQA_en"})
answer = engine.get_best_passage(evidence, question)
```

The default checkpoint (`drqa_en_v1.1.pt`) and its `meta.msgpack` are downloaded
from HuggingFace into the shared cache. Pass `model_path` / `meta_path` to use
local files instead.

### Python

```python
from drqa import DrQA
from drqa.download import download_model

model_path, meta_path = download_model()
dr = DrQA(model_path, meta_path, cuda=False)

answer = dr.predict(
    "Super Bowl 50 was played on February 7, 2016, at Levi's Stadium.",
    "When was the game played?",
)
print(answer)  # February 7, 2016
```

### Command line

```bash
drqa-interact
```

Or `python -m drqa.interact`. Example interaction:

```
Evidence: Super Bowl 50 was an American football game ... played on February 7, 2016, at Levi's Stadium ...
Question: What day was the game played on?
Answer: February 7, 2016
```

## Training

Download SQuAD and GloVe, then preprocess and train:

```bash
bash download.sh
python prepro.py
python train.py -e 40 -bs 128 --save_last_only
```

## Attribution

Original implementation: https://github.com/hitvoice/DrQA

Most of the PyTorch model code is borrowed from
[Facebook/ParlAI](https://github.com/facebookresearch/ParlAI/) under a BSD-3
license (see `LICENSE`).
