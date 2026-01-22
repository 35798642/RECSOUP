# -*- coding: UTF-8 -*-
import os
import sys
import logging
import pickle
import torch
import yaml
import time
import pandas as pd

from REC.runner import BaseRunner
from REC.reader import SeqReader
from REC.models import SASRec
from REC.utils import utils


# -------------------------------
# Helper class to convert dict -> object
# -------------------------------
class ConfigObject:
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigObject(v))
            else:
                setattr(self, k, v)


# -------------------------------
# Set random seed
# -------------------------------
def set_seed(seed: int):
    utils.init_seed(seed)


# -------------------------------
# Load YAML config
# -------------------------------
def load_config(yaml_file: str):
    with open(yaml_file, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return ConfigObject(cfg_dict)


# -------------------------------
# Save recommendation results
# -------------------------------
def save_rec_results(dataset, runner, topk, args):
    model_name_str = f"{args.model.name}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_path = os.path.join(args.paths.pred_dir, f"rec-{model_name_str}-{dataset.phase}-{timestamp}.csv")
    utils.check_dir(result_path)
    
    out_dict = runner.predict(dataset)
    predictions = out_dict['predictions']
    users, rec_items, rec_predictions = [], [], []
    
    for i in range(len(dataset)):
        info = dataset[i]
        users.append(info['user_id'])
        item_scores = zip(info['item_id'], predictions[i])
        sorted_lst = sorted(item_scores, key=lambda x: x[1], reverse=True)[:topk]
        rec_items.append([x[0] for x in sorted_lst])
        rec_predictions.append([x[1] for x in sorted_lst])
    
    rec_df = pd.DataFrame(columns=['user_id', 'rec_items', 'rec_predictions'])
    rec_df['user_id'] = users
    rec_df['rec_items'] = rec_items
    rec_df['rec_predictions'] = rec_predictions
    rec_df.to_csv(result_path, sep=args.data.sep, index=False)
    logging.info(f"{dataset.phase} Prediction results saved to {result_path}")


# -------------------------------
# Main
# -------------------------------
def main():
    # 1. Load YAML config
    args = load_config("./REC/configs/train_sasrec.yaml")

    # 2. Set seed
    set_seed(args.seed)

    # 3. Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using device: {args.device}")

    model_map = {
        "SASRec": SASRec.SASRec,
    }
    if args.model.name not in model_map:
        raise ValueError(f"Unknown model_name '{args.model.name}' in YAML config!")
    model_class = model_map[args.model.name]
    # 4. Load corpus
    corpus_path = os.path.join(args.data.path, args.data.dataset, model_class.reader + ".pkl")
    if not args.data.regenerate and os.path.exists(corpus_path):
        logging.info(f"Load corpus from {corpus_path}")
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        corpus = SeqReader.SeqReader(args)
        logging.info(f"Save corpus to {corpus_path}")
        pickle.dump(corpus, open(corpus_path, 'wb'))

    # 5. Initialize model via mapping
    model = model_class(args, corpus).to(args.device)
    logging.info(f"Model {args.model.name} initialized with #params: {model.count_variables()}")

    # 6. Save initial weights
    init_weights_path = os.path.join(args.paths.model_dir, args.data.dataset, f"init-{args.model.name}.pt")
    torch.save(model.state_dict(), init_weights_path)
    logging.info(f"Initial weights saved to {init_weights_path}")

    # 7. Build datasets
    data_dict = {}
    for phase in ['train', 'dev', 'test']:
        data_dict[phase] = model_class.Dataset(model, corpus, phase)
        data_dict[phase].prepare()

    # 8. Initialize Runner
    runner = BaseRunner.BaseRunner(args)

    # 9. Train + Eval + Test
    logging.info("Dev test before training: " + runner.print_res(data_dict['dev']))
    if args.training.load > 0:
        model.load_model()
    if args.train > 0:
        runner.train(data_dict)

    logging.info("Dev after training: " + runner.print_res(data_dict['dev']))
    logging.info("Test after training: " + runner.print_res(data_dict['test']))

    if args.training.save_final_results:
        save_rec_results(data_dict['dev'], runner, 100, args)
        save_rec_results(data_dict['test'], runner, 100, args)

    model.actions_after_train()
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
