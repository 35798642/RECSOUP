# RecSoup

Official implementation of:

**RecSoup: Efficient Model Merging Approach for Controllable Multi-Objective Recommendation**

**Accepted at CIKM 2026.**

RecSoup is a model-merging framework for controllable multi-objective
recommendation. It trains objective-specific recommendation models and
efficiently combines them to support different trade-offs among recommendation
objectives.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd RecSoup
pip install -r requirements.txt
```

## Project Structure

```text
RecSoup/
├── REC/
│   ├── configs/
│   ├── models/
│   ├── reader/
│   ├── runner/
│   ├── utils/
│   └── main.py
├── SOUP/
│   ├── config.yaml
│   ├── dict_moe.py
│   ├── evaluation.py
│   ├── merge_utils.py
│   ├── recsoup.py
│   ├── run_recsoup.py
│   └── utils.py
├── run_main.py
├── requirements.txt
├── README.md
└── THIRD_PARTY_LICENSES.md
```

## REC

`REC` contains the code for training the objective-specific recommendation
models that are subsequently merged by RecSoup.

Main components include:

* `models/`: recommendation model implementations.
* `reader/`: data loading and preprocessing utilities.
* `runner/`: training and evaluation pipelines.
* `utils/`: utility functions used by the recommendation framework.
* `configs/`: experiment configuration files.

For example, the SASRec training configuration is provided in:

```text
REC/configs/train_sasrec.yaml
```

Dataset, model, objective, and training parameters can be modified in the
corresponding configuration file.

### Training Objective-Specific Models

* `run_main.py`: entry point for training recommendation models.

```bash
cd REC
python run_main.py
```

Train the required objective-specific models before running the RecSoup
model-merging pipeline.

## SOUP

`SOUP` contains the core implementation of RecSoup, including model pruning,
preference-conditioned model merging, and evaluation.

Main components include:

* `recsoup.py`: core implementation of RecSoup.
* `run_recsoup.py`: entry point for model merging and evaluation.
* `dict_moe.py`: preference-conditioned weight-ensembling modules.
* `merge_utils.py`: model-merging utilities.
* `evaluation.py`: evaluation functions and metrics.
* `utils.py`: supporting utilities.
* `config.yaml`: configuration for RecSoup experiments.

### Model Merging and Evaluation

After the objective-specific recommendation models have been trained, run:

```bash
cd SOUP
python run_recsoup.py
```

Model paths, objective settings, pruning parameters, and model-merging
parameters can be configured in:

```text
SOUP/config.yaml
```

## Workflow

The overall workflow consists of two stages:

1. **Train objective-specific recommendation models.**

   Use the code under `REC` to independently optimize recommendation models
   for the required objectives.

2. **Merge objective-specific models with RecSoup.**

   Use the code under `SOUP` to prune, merge, and evaluate the trained
   objective-specific models under different preference settings.

## Data and Checkpoints

Datasets and trained model checkpoints are not included in this repository.

Please prepare the required datasets and train the corresponding
objective-specific models before running the model-merging pipeline. Dataset
and checkpoint paths should be configured according to the corresponding YAML
configuration files.

## Third-Party Code

This repository contains code adapted from or built upon several open-source
research codebases. Third-party source information and applicable license notices are provided in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

