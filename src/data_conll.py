from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from datasets import DatasetDict, Dataset, load_dataset


def load_conll2003_hf() -> DatasetDict:
    """
    Loads CoNLL-2003 via Hugging Face datasets.
    Returns DatasetDict with splits: train/validation/test.
    """
    return load_dataset("BramVanroy/conll2003")


def read_conll_file(path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Reads a CoNLL-style file (like Kaggle eng.train/testa/testb).
    Format: token + columns; NER tag usually last column.
    Sentences separated by blank lines. Ignores -DOCSTART- lines.
    """
    sentences: List[List[str]] = []
    labels: List[List[str]] = []

    cur_tokens: List[str] = []
    cur_tags: List[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            if cur_tokens:
                sentences.append(cur_tokens)
                labels.append(cur_tags)
                cur_tokens, cur_tags = [], []
            continue

        if line.startswith("-DOCSTART-"):
            continue

        parts = line.split()
        token = parts[0]
        tag = parts[-1]  # in CoNLL-2003 files, NER tag is last column
        cur_tokens.append(token)
        cur_tags.append(tag)

    if cur_tokens:
        sentences.append(cur_tokens)
        labels.append(cur_tags)

    return sentences, labels


def load_conll2003_kaggle(data_dir: str | Path) -> DatasetDict:
    """
    Expects files:
      data_dir/eng.train
      data_dir/eng.testa  (validation)
      data_dir/eng.testb  (test)
    Produces DatasetDict with fields: tokens (List[str]), ner_tags_str (List[str])
    """
    data_dir = Path(data_dir)
    train_s, train_t = read_conll_file(data_dir / "eng.train")
    val_s, val_t = read_conll_file(data_dir / "eng.testa")
    test_s, test_t = read_conll_file(data_dir / "eng.testb")

    def to_ds(sents, tags):
        return Dataset.from_dict({"tokens": sents, "ner_tags_str": tags})

    return DatasetDict(
        train=to_ds(train_s, train_t),
        validation=to_ds(val_s, val_t),
        test=to_ds(test_s, test_t),
    )


@dataclass
class LabelSpace:
    id2label: Dict[int, str]
    label2id: Dict[str, int]


def build_label_space_from_strings(dataset: DatasetDict, field: str = "ner_tags_str") -> LabelSpace:
    labels = set()
    for split in dataset.keys():
        for seq in dataset[split][field]:
            labels.update(seq)
    labels = sorted(labels)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return LabelSpace(id2label=id2label, label2id=label2id)
