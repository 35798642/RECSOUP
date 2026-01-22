import os
import itertools
import sys
import torch
import logging
import pickle
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader

from omegaconf import DictConfig
from torch import nn, Tensor
import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from REC.runner import BaseRunner
from utils import *
from merge_utils import *
from dict_moe import ParetoWeightEnsemblingModule

log = logging.getLogger(__name__)

class RecSoupProgram:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.version = cfg.experiment.version
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
        self.result_dir = os.path.abspath(os.path.join(cfg.paths.result_dir,self.version))
        os.makedirs(self.result_dir, exist_ok=True)
        self.load_models()

    def load_models(self):
        """
        Load corpus, pretrained model, and finetuned expert models.
        This implementation is STRICTLY aligned with the original REC load_model logic.
        """
        cfg = self.cfg
        device = self.device

        # --------------------------------------------------
        # 1. Load corpus (MUST be identical to REC)
        # --------------------------------------------------
        from REC.reader import SeqReader

        corpus_path = os.path.join(
            cfg.rec.path,
            cfg.rec.dataset,
            f"{cfg.rec.reader}.pkl"
        )
        print(corpus_path)
        print(os.path.exists(corpus_path))
        if not cfg.rec.regenerate and os.path.exists(corpus_path):
            log.info(f"Load corpus from {corpus_path}")
            with open(corpus_path, "rb") as f:
                corpus = pickle.load(f)
        else:
            log.info("Corpus not found, rebuilding via REC reader")
            reader_cls = SeqReader.SeqReader
            corpus = reader_cls(cfg)
            os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
            with open(corpus_path, "wb") as f:
                pickle.dump(corpus, f)

        self.corpus = corpus

        # --------------------------------------------------
        # 2. Resolve REC model class
        # --------------------------------------------------
        from REC.models import SASRec

        if cfg.model.name == "SASRec":
            model_cls = SASRec.SASRec
        else:
            raise ValueError(f"Unsupported model: {cfg.model}")

        # --------------------------------------------------
        # 3. Load pretrained (initial) model
        # --------------------------------------------------
        log.info("Loading pretrained model")

        pretrained_path = os.path.join(
            cfg.paths.rec_checkpoint_dir,
            cfg.model.name,
            cfg.rec.dataset,
            f"init-{cfg.model.name}.pt"
        )

        pretrained_model = model_cls(cfg, corpus).to(device)
        pretrained_model.load_state_dict(
            torch.load(pretrained_path, map_location=device),
            strict=False
        )
        pretrained_model.requires_grad_(False)

        self.pretrained_model = pretrained_model

        # --------------------------------------------------
        # 4. Load finetuned expert models
        # --------------------------------------------------
        log.info("Loading finetuned models")

        self.finetuned_models = {}

        merging_tasks = [
            "model_acc1.0_div0.0",
            "model_acc0.0_div1.0",
        ]

        filepaths = [
            os.path.join(
                cfg.paths.rec_checkpoint_dir,
                cfg.model.name,
                cfg.rec.dataset,
                task,
                "best.pt"
            )
            for task in merging_tasks
        ]

        if cfg.experiment.prune:
            log.info(f"Pruning model with threshold {cfg.experiment.prune_threshold} and self_cof {cfg.experiment.self_cof}")
            tv_flat_checks, flat_ptm, ptm_check, remove_keys, reshape_keys=load_tv(merging_tasks,filepaths,pretrained_path)
            pruned_tvs, _ = topk_NFS_mask_preserve_normfrac(tv_flat_checks, cfg.experiment.prune_threshold, cfg.experiment.self_cof,1-cfg.experiment.self_cof)

            for i,task_name in enumerate(merging_tasks):
                # restore: θ = θ_pretrained + Δθ_pruned
                restored_tv = pruned_tvs[i] + flat_ptm

                restored_sd = vector_to_state_dict(
                    restored_tv,
                    ptm_check,
                    remove_keys=remove_keys,
                )
                model = model_cls(cfg, corpus).to(device)
                model.load_state_dict(
                    restored_sd,
                    strict=False
                )
                model.requires_grad_(False)
                self.finetuned_models[task_name] = model
        else:
            for task, fp in zip(merging_tasks, filepaths):
                model = model_cls(cfg, corpus).to(device)
                model.load_state_dict(
                    torch.load(fp, map_location=device),
                    strict=False
                )
                model.requires_grad_(False)
                self.finetuned_models[task] = model

        # --------------------------------------------------
        # 5. Build Pareto MoE (weight ensembling)
        # --------------------------------------------------
        log.info("Building Pareto Weight-Ensembling MoE")

        self.model = ParetoWeightEnsemblingModule(
            base_model=self.pretrained_model,
            expert_models=[
                self.finetuned_models[task]
                for task in self.finetuned_models.keys()
            ],
            init_lambda=cfg.experiment.init_lambda,
            fix_base_model_and_experts=True,
            router_hidden_layers=cfg.experiment.router_hidden_layers,
        )

        from REC.reader import SeqReader
        self.data_dict = {}

        model = model_cls(cfg, corpus).to(device)
        for phase in ['train', 'dev', 'test']:
            self.data_dict[phase] = model.Dataset(model, corpus, phase)
            self.data_dict[phase].prepare()

    def compute_loss(self, model: nn.Module, ray: Tensor, losses):
        losses = torch.stack(losses)
        losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
        loss = torch.sum(ray * losses)
        return loss

    def train(self):
        cfg = self.cfg
        device = self.device
        W = cfg.experiment.window_size
        λ = cfg.experiment.reg_lambda
        T = len(self.finetuned_models)
        backbone = deepcopy(self.model)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, backbone.parameters()), lr=cfg.experiment.lr)

        train_log = {"step": [], "mean_loss": []}
        dataset = self.data_dict['train']

        for step_idx in tqdm(range(1, cfg.experiment.num_steps + 1)):
            rays = sample_preference_vectors(W, T, cfg.experiment.alpha, device)
            loss_matrix = []
            for i in range(W):
                α = rays[i]
                ParetoWeightEnsemblingModule.set_preference_vector(backbone, α)
                batch_losses = torch.zeros(T).to(device)
                dl = DataLoader(dataset, batch_size=cfg.experiment.batch_size, shuffle=True,collate_fn=dataset.collate_batch)
                 
                for batch in itertools.islice(dl, cfg.experiment.num_batch_per_ray):
                    batch = batch_to_gpu(batch, device)
                    outputs = backbone(batch)
                    forward_model = backbone.get_merged_model()
                    acc_loss = forward_model.loss(outputs)
                    div_loss = forward_model.diversity_loss(outputs)
                    batch_losses += torch.stack([acc_loss, div_loss])
                loss_matrix.append(batch_losses / cfg.experiment.num_batch_per_ray)
            loss_matrix = torch.stack(loss_matrix)
            total_loss = torch.mean(torch.sum(rays * loss_matrix, dim=1))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_log["step"].append(step_idx)
            train_log["mean_loss"].append(total_loss.item())
            if step_idx % cfg.experiment.save_interval == 0:
                torch.save({"model": backbone}, os.path.join(self.result_dir, f"model_step={step_idx}.pt"))
                pd.DataFrame(train_log).to_json(os.path.join(self.result_dir, "train_log.json"), orient="records", force_ascii=False)
        # save current training config
        with open(os.path.join(self.result_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f)

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluation following the original ParetoMoE evaluation logic,
        aligned with the current ParetoMoEProgram implementation.
        """
        from collections import defaultdict
        import re
        import numpy as np

        results = defaultdict(list)
        cfg = self.cfg
        device = self.device

        num_objectives = len(self.finetuned_models)

        # -------- iterate checkpoints  --------
        for step_idx in range(cfg.experiment.save_interval, cfg.experiment.num_steps + 1, cfg.experiment.save_interval):
            log.info(f"Evaluating step {step_idx}")

            ckpt_path = os.path.join(
                self.result_dir,
                f"model_step={step_idx}.pt"
            )
            state = torch.load(ckpt_path, map_location=device)

            # train 中是 {"model": backbone}
            if isinstance(state, dict) and "model" in state:
                model = state["model"]
            else:
                model = state

            model.eval()
            # -------- generate preference rays --------
            if cfg.experiment.num_evaluation_samples == "equal_weight":
                uniform_grid = np.array(
                    [[1.0 / num_objectives] * num_objectives],
                    dtype=np.float32,
                )
            else:
                uniform_grid = generate_simplex_grid(
                    num_objectives,
                    cfg.experiment.num_evaluation_samples,
                )

            # -------- evaluate each ray --------
            for ray_idx, ray in tqdm(
                enumerate(uniform_grid),
                total=len(uniform_grid),
                desc="evaluating samples",
            ):
                results["step"].append(step_idx)
                for i in range(len(ray)):
                    results[f"ray_{i}"].append(float(ray[i]))

                ray = torch.from_numpy(ray).to(device)

                # set preference vector
                ParetoWeightEnsemblingModule.set_preference_vector(model, ray)

                # merge model
                final_sd, routing_weights = model._merge_state_dict()

                # external evaluation
                eval_result = self.evaluate_model(
                    final_sd,
    
                )

                # -------- parse metrics --------
                alpha_ndcg5 = re.search(r"ALPHA_NDCG@5:([\d.]+)", eval_result).group(1)
                hr5 = re.search(r"HR@5:([\d.]+)", eval_result).group(1)
                ndcg5 = re.search(r",NDCG@5:([\d.]+)", eval_result).group(1)

                alpha_ndcg10 = re.search(r"ALPHA_NDCG@10:([\d.]+)", eval_result).group(1)
                hr10 = re.search(r"HR@10:([\d.]+)", eval_result).group(1)
                ndcg10 = re.search(r",NDCG@10:([\d.]+)", eval_result).group(1)

        
                results["α-NDCG@5"].append(float(alpha_ndcg5))
                results["HR@5"].append(float(hr5))
                results["NDCG@5"].append(float(ndcg5))

                results["α-NDCG@10"].append(float(alpha_ndcg10))
                results["HR@10"].append(float(hr10))
                results["NDCG@10"].append(float(ndcg10))

                # routing weights
                results["routing_weights"].append(
                    routing_weights.round(decimals=4).tolist()
                )

                # save results
                df = pd.DataFrame(results)
                df.to_csv(
                    os.path.join(self.result_dir, "result.csv"),
                    index=False,
                )

            log.info(f"Finished evaluation for step {step_idx}")
            log.info(df)

    def evaluate_model(self, final_sd):
        """
        Build a REC model from merged state_dict and run REC evaluation.
        This method is class-internal and aligned with REC pipeline.
        """
        import logging
        from REC.models import SASRec

        cfg = self.cfg
        corpus = self.corpus
        device = self.device

        # -------- resolve model class --------
        if cfg.model.name == "SASRec":
            model_cls = SASRec.SASRec
        else:
            raise ValueError(f"Unsupported model: {cfg.model.name}")

        # -------- build model --------
        model = model_cls(cfg, corpus).to(device)
        model.load_state_dict(final_sd, strict=False)

        logging.info("Model loaded from merged checkpoint")
        logging.info(f"#params: {model.count_variables()}")

        # -------- build evaluation dataset --------
        data_dict = {}
        for phase in ["test"]:
            data_dict[phase] = model.Dataset(model, corpus, phase)
            data_dict[phase].prepare()

        # -------- run REC runner --------
        runner = BaseRunner.BaseRunner(cfg)

        eval_res = runner.print_res(data_dict["test"])
        logging.info(os.linesep + "Test After Training: " + eval_res)

        return eval_res
