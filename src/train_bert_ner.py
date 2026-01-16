from __future__ import annotations

import argparse
import numpy as np
import evaluate
from typing import Any, Dict, List

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from data_conll import load_conll2003_hf, load_conll2003_kaggle, build_label_space_from_strings


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="bert-base-cased")
    p.add_argument("--output_dir", type=str, default="outputs/bert-ner")
    p.add_argument("--max_length", type=int, default=256)

    # data source
    p.add_argument("--source", choices=["hf", "kaggle"], default="hf")
    p.add_argument("--kaggle_dir", type=str, default=None)

    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


seqeval = evaluate.load("seqeval")


def align_labels_with_tokens(labels: List[int], word_ids: List[int | None]) -> List[int]:
    """
    Standard HF alignment: label first sub-token, mask subsequent sub-tokens with -100.
    """
    aligned = []
    current_word = None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)
        elif wid != current_word:
            aligned.append(labels[wid])
            current_word = wid
        else:
            aligned.append(-100)
    return aligned


def main():
    args = parse_args()

    # 1) Load data
    if args.source == "hf":
        ds: DatasetDict = load_conll2003_hf()
        # HF conll2003 has integer ner_tags + feature names
        label_names = ds["train"].features["ner_tags"].feature.names
        id2label = {i: n for i, n in enumerate(label_names)}
        label2id = {n: i for i, n in enumerate(label_names)}
        tokens_field = "tokens"
        labels_field = "ner_tags"
        labels_are_strings = False
    else:
        if not args.kaggle_dir:
            raise SystemExit("--kaggle_dir is required when --source=kaggle")
        ds = load_conll2003_kaggle(args.kaggle_dir)
        label_space = build_label_space_from_strings(ds, field="ner_tags_str")
        id2label, label2id = label_space.id2label, label_space.label2id
        tokens_field = "tokens"
        labels_field = "ner_tags_str"
        labels_are_strings = True

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_and_align(batch: Dict[str, Any]) -> Dict[str, Any]:
        tokenized = tokenizer(
            batch[tokens_field],
            truncation=True,
            is_split_into_words=True,
            max_length=args.max_length,
        )

        if labels_are_strings:
            label_ids_batch = [[label2id[t] for t in tags] for tags in batch[labels_field]]
        else:
            label_ids_batch = batch[labels_field]

        aligned_labels = []
        for i in range(len(label_ids_batch)):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned_labels.append(align_labels_with_tokens(label_ids_batch[i], word_ids))

        tokenized["labels"] = aligned_labels
        return tokenized

    ds_tok = ds.map(tokenize_and_align, batched=True)

    # 3) Model
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []
        for pred_seq, lab_seq in zip(predictions, labels):
            cur_preds = []
            cur_labs = []
            for pred, lab in zip(pred_seq, lab_seq):
                if lab == -100:
                    continue
                cur_preds.append(id2label[int(pred)])
                cur_labs.append(id2label[int(lab)])
            true_predictions.append(cur_preds)
            true_labels.append(cur_labs)

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # 4) Train
    targs = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(ds_tok["test"])
    print("TEST METRICS:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
