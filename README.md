# Named Entity Recognition (NER) — CoNLL-2003

This project fine-tunes a Transformer (BERT) for token classification (NER) and compares it to a BiLSTM-CRF baseline.

Entity types in CoNLL-2003: PER, ORG, LOC, MISC.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
