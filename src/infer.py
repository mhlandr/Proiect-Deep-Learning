from __future__ import annotations

import argparse
from transformers import pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, help="Path to saved HF model dir (outputs/bert-ner)")
    p.add_argument("--text", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    ner = pipeline("token-classification", model=args.model_dir, tokenizer=args.model_dir, aggregation_strategy="simple")
    out = ner(args.text)
    for ent in out:
        print(f"{ent['word']}\t{ent['entity_group']}\tscore={ent['score']:.3f}\tspan=({ent['start']},{ent['end']})")


if __name__ == "__main__":
    main()
