# -*- coding: UTF-8 -*-
import os
import logging
import sys
import torch
import yaml
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from recsoup import RecSoupProgram

# -----------------------------
# load configuration
# -----------------------------
cfg_path = Path(__file__).parent / "config.yaml"
with open(cfg_path, "r") as f:
    cfg_dict = yaml.safe_load(f)


class ConfigObject:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigObject(v))
            else:
                setattr(self, k, v)

cfg = ConfigObject(cfg_dict)

# -----------------------------
# log initialization
# -----------------------------
version = cfg.experiment.version
result_dir=os.path.join(cfg.paths.result_dir, version)
log_file = Path(result_dir) / "recsoup.log"
os.makedirs(result_dir, exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Loaded RecSoup config:")
logging.info(cfg_dict)

# -----------------------------
# random seed setting
# -----------------------------
seed = cfg.seed if hasattr(cfg, "seed") else 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# -----------------------------
# run RecSoup program
# -----------------------------
program = RecSoupProgram(cfg)
if cfg.experiment.train:
    logging.info("Starting RecSoup training...")
    program.train()

if cfg.experiment.evaluate:
    logging.info("Starting RecSoup evaluation...")
    program.evaluate()

logging.info("RecSoup program finished.")