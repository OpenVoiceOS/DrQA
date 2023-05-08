import argparse

import torch

from drqa import DrQA
from drqa.utils import str2bool

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Interact with document reader model.'
    )
    parser.add_argument('--model-file', help='path to model file')
    parser.add_argument('--meta-file', help='path to meta.msgpack file')
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
