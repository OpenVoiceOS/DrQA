DrQA
---
A pytorch implementation of the ACL 2017 paper [Reading Wikipedia to Answer Open-Domain Questions](http://www-cs.stanford.edu/people/danqi/papers/acl2017.pdf) (DrQA).

Reading comprehension is a task to produce an answer when given a question and one or more pieces of evidence (usually natural language paragraphs). Compared to question answering over knowledge bases, reading comprehension models are more flexible and have revealed a great potential for zero-shot learning.

[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension benchmark where there's only a single piece of evidence and the answer is guaranteed to be a part of the evidence. Since the publication of SQuAD dataset, there has been fast progress in the research of reading comprehension and a bunch of great models have come out. DrQA is one that is conceptually simpler than most others but still yields strong performance even as a single model.

The motivation for this project is to offer a clean version of DrQA for the machine reading comprehension task, so one can quickly do some modifications and try out new ideas.

## Requirements
- python >=3.5 
- pytorch >=0.4. Tested on pytorch 0.4 and pytorch 1.10
- numpy
- msgpack
- spacy 3.x

## Quick Start

- download SpaCy English language model `python3 -m spacy download en_core_web_md`
- download model - https://huggingface.co/Jarbas/DrQA_en


### Predict

```bash
python drqa/interact.py
```
Example interactions:
```
Evidence: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24-10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
Question: What day was the game played on?
Answer: February 7, 2016
Time: 0.0245s

Evidence: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24-10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
Question: What is the AFC short for?
Answer: The American Football Conference
Time: 0.0214s

Evidence: Beanie style with simple design. So cool to wear and make you different. It wears as peak cap and a fashion cap. It is good to match your clothes well during party and holiday, also makes you charming and fashion, leisure and fashion in public and streets. It suits all adults, for men or women. Matches well with your winter outfits so you stay warm all winter long.
Question: Is it for women?
Answer: It suits all adults, for men or women
Time: 0.0238s

This timing indicates that the dog was the first species to be domesticated in the time of hunter–gatherers, which predates agriculture. DNA sequences show that all ancient and modern dogs share a common ancestry and descended from an ancient, extinct wolf population which was distinct from the modern wolf lineage. Most dogs form a sister group to the remains of a Late Pleistocene wolf found in the Kessleroch cave near Thayngen in the canton of Schaffhausen, Switzerland, which dates to 14,500 years ago. The most recent common ancestor of both is estimated to be from 32,100 years ago. This indicates that an extinct Late Pleistocene wolf may have been the ancestor of the dog. with the modern wolf being the dog's nearest living relative

The dog is a classic example of a domestic animal that likely travelled a commensal pathway into domestication. The questions of when and where dogs were first domesticated have taxed geneticists and archaeologists for decades. Genetic studies suggest a domestication process commencing over 25,000 years ago, in one or several wolf populations in either Europe, the high Arctic, or eastern Asia. In 2021, a literature review of the current evidence infers that the dog was domesticated in Siberia 23,000 years ago by ancient North Siberians, then later dispersed eastward into the Americas and westward across Eurasia.

```
The last example is a randomly picked product description from Amazon (not in SQuAD).


### Python

```python
from drqa import DrQA

parser = argparse.ArgumentParser(
    description='Interact with document reader model.'
)
parser.add_argument('--model-file', help='path to model file')
parser.add_argument('--meta-file',  help='path to meta.msgpack file')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
args = parser.parse_args()

dr = DrQA(args.model_file, args.meta_file, cuda=args.cuda)
while True:
    evidence = input("evidence:")
    question = input("question:")
    answer = dr.predict(evidence, question)
    print(">", answer)
```

### About

Original Implementation: https://github.com/hitvoice/DrQA

Most of the pytorch model code is borrowed from [Facebook/ParlAI](https://github.com/facebookresearch/ParlAI/) under a BSD-3 license.
