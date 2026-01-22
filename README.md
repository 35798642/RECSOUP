# RECSOUP

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/RecSoup.git
cd RecSoup
pip install -r requirements.txt
````

---

## Project Overview

### REC

Contains code for training recommendation models to be merged.

* `runner/`, `models/`, `reader/`, `main.py` — training functions and model definitions
* `run_main.py` — main entry point for training models
* Example: **SASRec** training

  * Configuration: `/REC/configs/train_sasrec.yaml`
  * Modify the YAML file to change dataset, model, or training parameters

### SOUP

Core code for pruning and merging models.

* `recsoup.py` — main program
* `run_recsoup.py` — entry point for training, evaluation, and optional pruning
* Configuration: `config.yaml`
* `evaluation.py` — functions for calculating metrics used in experiments

---

## Usage

1. **Train individual models (REC)**

```bash
python run_main.py
```

2. **Prune, merge, and evaluate models (RECSOUP)**

```bash
python run_recsoup.py
```
## Acknowledgement

This project includes components adapted from open-source implementations released under permissive licenses (e.g., BSD 3-Clause, MIT).
Details can be found in THIRD_PARTY_LICENSES.md.
