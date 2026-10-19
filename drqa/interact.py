import argparse

import torch

from drqa import DrQA
from drqa.download import DEFAULT_MODEL_REPO, download_model
from drqa.utils import str2bool


def main():
    parser = argparse.ArgumentParser(
        description='Interact with the DrQA document reader model.'
    )
    parser.add_argument('--model-file', help='path to a local model checkpoint')
    parser.add_argument('--meta-file', help='path to a local meta.msgpack file')
    parser.add_argument('--model-repo', default=DEFAULT_MODEL_REPO,
                        help='HuggingFace repo id used when no local files are given')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    args = parser.parse_args()

    model_path, meta_path = download_model(args.model_file, args.meta_file,
                                           args.model_repo)
    dr = DrQA(model_path, meta_path, cuda=args.cuda)
    while True:
        try:
            evidence = input("evidence:")
            question = input("question:")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        answer = dr.predict(evidence, question)
        print(">", answer)


if __name__ == "__main__":
    main()
