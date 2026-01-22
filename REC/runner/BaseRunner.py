# -*- coding: UTF-8 -*-
"""
This file contains code adapted from:

Paragon: Parameter Generation for Controllable Multi-Task Recommendation
Chenglei Shen, Jiahao Zhao, Xiao Zhang, Weijie Yu, Ming He, Jianping Fan
Proceedings of the 19th ACM Conference on Recommender Systems (RecSys), 2025.

Original source:
 	https://github.com/bubble65/Paragon.git

License:
<original license, e.g., MIT License>

Modifications:
- Refactored for experimental comparison with model merging methods.
"""
import os
import gc
import random
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
import pandas as pd
from ..utils import utils
from ..models.BaseModel import BaseModel
from functools import reduce
from collections import defaultdict
import time as pytime
from ..baseline import MMR


class BaseRunner(object):
	@staticmethod
	def evaluate_method(output_dict: dict, topk: list, metrics: list, id_multihot: dict) -> Dict[str, float]:
		predictions: np.ndarray = output_dict["predictions"]
		dataset_ids = output_dict["dataset_ids"]
		"""
		:param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
		:param dataset_ids: (-1, n_candidates) shape, the first column is the ground-truth item
		:param topk: top-K value list
		:param metrics: metric string list
		:return: a result dict, the keys are metric@topk
		"""
		evaluations = dict()
		# ↓ As we only have one positive sample, comparing with the first item will be more efficient. 
		gt_rank = (predictions >= predictions[:,0].reshape(-1,1)).sum(axis=-1)
		for k in topk:
			hit = (gt_rank <= k)
			for metric in metrics:
				key = '{}@{}'.format(metric, k)
				if metric == 'HR':
					evaluations[key] = hit.mean()
				elif metric == 'NDCG':
					evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
				elif metric == 'ALPHA_NDCG':
					scores = BaseRunner.best_alpha_nDCG(output_dict, id_multihot, 0.5, k)
					evaluations[key] = BaseRunner.alpha_ndcg(output_dict, id_multihot, 0.5,k, normaliztion = scores)
				else:
					raise ValueError('Undefined evaluation metric: {}.'.format(metric))
		return evaluations
	
	@staticmethod
	def alpha_ndcg(output_dict: dict, id_multihot: Dict[int, list], alpha, k, normaliztion) -> float:
		batch_predictions = output_dict["predictions"]
		batch_ids = output_dict["dataset_ids"]
		a_ndcg_list = []
		category =  []
		for i in range(batch_ids.shape[0]): 
			sorted_indices = np.argsort(batch_predictions[i])[::-1]  
			sorted_ids = batch_ids[i][sorted_indices]
			grade_list = np.zeros((len(sorted_ids), len(id_multihot[str(sorted_ids[0])])))
			for j, id_ in enumerate(sorted_ids):
				grade_list[j] = id_multihot[str(id_)] 
			grade_list = np.transpose(np.array(grade_list))
			
			category.append(BaseRunner.test_category(grade_list)[k])

			alpha, n_subtopics, n_docs = 1 - alpha, grade_list.shape[0], grade_list.shape[1]
			grade_list = (grade_list>0).astype(int)
			cum = reduce(lambda t,i:  [t[0]+(grade_list[:,i]),
									t[1]+np.sum(np.dot(np.power(alpha,t[0]), grade_list[:,i]))/np.log2(i+2)],
						range(k), [np.zeros(n_subtopics), 0])[1]
			a_ndcg_list.append(cum)

			result = [a / b for a, b in zip(a_ndcg_list, normaliztion)]
		return  np.mean(result)	 
	
	@staticmethod
	def test_category(grade_list):
		grad_array = np.array(grade_list)
		topics = grad_array.sum(axis=0).tolist()
		for i in range(1,len(topics)):
			topics[i] = topics[i]+topics[i-1]
		return topics

	@staticmethod	
	def best_alpha_nDCG(output_dict: dict, id_multihot: Dict[int, list], alpha, k):
		batch_predictions = output_dict["predictions"]
		batch_ids = output_dict["dataset_ids"]
		score_batch = []
		for i in range(batch_ids.shape[0]): 
			sorted_indices = np.argsort(batch_predictions[i])[::-1]  
			sorted_ids = batch_ids[i][sorted_indices]
			grade_list = np.zeros((len(sorted_ids), len(id_multihot[str(sorted_ids[0])]))) 
			for j, id_ in enumerate(sorted_ids):
				grade_list[j] = id_multihot[str(id_)]
			alpha, n_subtopics, n_docs = 1 - alpha, grade_list.shape[0], grade_list.shape[1]
			grade_list = (grade_list > 0).astype(int)
			mask = np.zeros(n_docs)
			discount = np.zeros(n_subtopics)
			score, rank, score_list = 0, [], []
			for i in range(k):
				scores = np.matmul(grade_list.T, np.power(alpha, discount)) + mask
				r = np.argmax(scores)
				discount += grade_list[:,r]
				score += scores[r]/np.log2(i+2)
				score_list.append(score)
				rank.append(r)
				mask[r] = np.finfo(np.float32).min 
			score_batch.append(score)
			
		return score_batch



	def __init__(self, args):

		self.accuracy_weight = args.accuracy_weight 
		self.diversity_weight = args.diversity_weight

	
		self.model_path_root = f"./checkpoints/{args.model.name}/{args.data.dataset}/model_acc{self.accuracy_weight}_div{self.diversity_weight}/"
		self.all_epoch = args.training.all_epoch
		self.save_epoch = args.training.save_epoch

		self.train_models = args.train
		self.check_epoch = args.training.check_epoch
		self.test_epoch = args.training.test_epoch
		self.early_stop = args.training.early_stop
		self.learning_rate = args.training.lr
		self.batch_size = args.training.batch_size
		self.eval_batch_size = args.testing.eval_batch_size
		self.l2 = args.training.l2
		self.optimizer_name = args.optimizer
		self.num_workers = args.num_workers
		self.pin_memory = args.pin_memory
		self.topk = [int(x) for x in args.eval.topk]
		self.metrics = [m.strip().upper() for m in args.eval.metric]
		self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[1]) if not len(args.eval.main_metric) else args.eval.main_metric # early stop based on main_metric
		self.main_topk = int(self.main_metric.split("@")[1])
		self.time = None  # will store [start_time, last_step_time]

		self.log_path = os.path.dirname(args.paths.log_file) # path to save predictions
		self.save_appendix = args.paths.log_file.split("/")[-1].split(".")[0] # appendix for prediction saving
		self.test_num = args.testing.test_num

		self.train_prefer_sampling = getattr(args, 'train_prefer_sampling', 'uniform')

	def _check_time(self, start=False):
		if self.time is None or start:
			self.time = [time()] * 2
			return self.time[0]
		tmp_time = self.time[1]
		self.time[1] = time()
		return self.time[1] - tmp_time

	def _build_optimizer(self, model):
		logging.info('Optimizer: ' + self.optimizer_name)
		print("learning_rate:",self.learning_rate)

		optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
			model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
		
		return optimizer
	
	def state_part(train_list, net):
		part_param = {}
		for name, weights in net.named_parameters():
			if name in train_list:
				part_param[name] = weights.detach().cpu()
		return part_param

	def normalize_loss(self, loss, min_val, max_val):
		return (loss - min_val) / (max_val - min_val) if max_val > min_val else loss

	def update_min_max(self, loss, min_val, max_val):
		return min(min_val, loss), max(max_val, loss)


	def train(self, data_dict: Dict[str, BaseModel.Dataset]):
		model = data_dict['train'].model
		if not os.path.exists(self.model_path_root):
			os.makedirs(self.model_path_root)

		main_metric_results, dev_results = list(), list()
		self._check_time(start=True)

		try:
			for epoch in range(self.all_epoch):
				self._check_time()
				gc.collect()
				torch.cuda.empty_cache()
				loss = self.fit(data_dict['train'], epoch=epoch + 1)
				if np.isnan(loss):
					logging.info("Loss is Nan. Stop training at %d."%(epoch+1))
					break
				training_time = self._check_time()
				# Observe selected tensors
				if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
					utils.check(model.check_list)

				# Record dev results
				dev_result = self.evaluate(data_dict['dev'], [self.main_topk], self.metrics)
				dev_results.append(dev_result)
				main_metric_results.append(dev_result[self.main_metric])
				logging_str = 'Epoch {:<5} loss={:<.7f} [{:<3.1f} s]	dev=({})'.format(
					epoch + 1, loss, training_time, utils.format_metric(dev_result))
				
				# Test
				if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
					test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
					logging_str += ' test=({})'.format(utils.format_metric(test_result))
				testing_time = self._check_time()
				logging_str += ' [{:<.1f} s]'.format(testing_time)

				# Save model and early stop
				if max(main_metric_results) == main_metric_results[-1] or \
						(hasattr(model, 'stage') and model.stage == 1):
					
			
					best_path = os.path.join(self.model_path_root, 'best.pt')
					os.makedirs(self.model_path_root, exist_ok=True)
					model.save_model(best_path)
					logging_str += ' *'
				
			
				os.makedirs(self.model_path_root, exist_ok=True)
				epoch_path = os.path.join(self.model_path_root, 'epoch{}.pt'.format(epoch + 1))
				model.save_model(epoch_path)
				logging_str += ' [saved: epoch{}.pt]'.format(epoch + 1)
				
				logging.info(logging_str)


		except KeyboardInterrupt:
			logging.info("Early stop manually")
			exit_here = input("Exit completely without evaluation? (y/n) (default n):")
			if exit_here.lower().startswith('y'):
				logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
				exit(1)

		# Find the best dev result across iterations
		best_epoch = main_metric_results.index(max(main_metric_results))
		logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
			best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
	
		best_model_path = os.path.join(self.model_path_root, 'best.pt')
		if os.path.exists(best_model_path):
			model.load_model(best_model_path)
			logging.info(f"Loaded best model from {best_model_path} for final evaluation.")
		else:
			logging.warning(f"Best model not found at {best_model_path}, skipping model reload.")

	
	def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
		model = dataset.model
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
			
		model.train()
		loss_lst = list()
		dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, 
						num_workers=self.num_workers, collate_fn=dataset.collate_batch, 
						pin_memory=self.pin_memory)
		
		for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), 
							ncols=100, mininterval=1):
			batch = utils.batch_to_gpu(batch, model.device)
		
			item_ids = batch['item_id']
			indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)        
			batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]

			model.optimizer.zero_grad()
			out_dict = model(batch)
			prediction = out_dict['prediction']
			if len(prediction.shape) == 2:
				restored_prediction = torch.zeros(*prediction.shape).to(prediction.device)
				restored_prediction[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices] = prediction   
				out_dict['prediction'] = restored_prediction
				out_dict['item_id'] = item_ids
				
			
			accuracy_loss = model.loss(out_dict)
			diversity_loss = model.diversity_loss(out_dict)
			
			acc_weight = self.accuracy_weight
			div_weight = self.diversity_weight
				
			loss = acc_weight * accuracy_loss + div_weight * diversity_loss
			loss.backward()
			model.optimizer.step()
			loss_lst.append(loss.detach().cpu().data.numpy())
			
		return np.mean(loss_lst).item()

	def eval_termination(self, criterion: List[float]) -> bool:
		if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
			return True
		elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
			return True
		return False

	def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		id_multihot = dataset.model.item_multihot_mapping
		output_dict = self.predict(dataset) ### {"predictions":predictions([2874（all) x 100]),"dataset_ids":dataset_ids([2874（all) x 100])} 
		return self.evaluate_method(output_dict, topks, metrics,id_multihot)

	def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> np.ndarray:
		"""
		The returned prediction is a 2D-array, each row corresponds to all the candidates,
		and the ground-truth item poses the first.
		Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
				 predictions like: [[1,3,4], [2,5,6]]
		"""
		dataset.model.eval()
		predictions = list()
		dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
		dataset_ids = []
		for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
			dataset_ids.extend(batch['item_id'].cpu().numpy().tolist())
			if hasattr(dataset.model,'inference'):
				prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			else:
				prediction = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
			predictions.extend(prediction.cpu().data.numpy())
		predictions = np.array(predictions)
		dataset_ids = np.array(dataset_ids)

		if dataset.model.test_all:
			rows, cols = list(), list()
			for i, u in enumerate(dataset.data['user_id']):
				clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
				idx = list(np.ones_like(clicked_items) * i)
				rows.extend(idx)
				cols.extend(clicked_items)
			predictions[rows, cols] = -np.inf
		return {"predictions":predictions,"dataset_ids":dataset_ids}

	def print_res(self, dataset: BaseModel.Dataset) -> str:
		"""
		Construct the final result string before/after training
		:return: test result string
		"""
		result_dict = self.evaluate(dataset, self.topk, self.metrics)
		res_str = '(' + utils.format_metric(result_dict) + ')'
		return res_str
	
	def easy_print_res(self, dataset: BaseModel.Dataset) -> str:
		"""
		Construct the final result string before/after training
		:return: test result string
		"""
		result_dict = self.evaluate(dataset, self.topk, self.metrics)
		easy_result_dict = {}
		easy_result_dict['NDCG@5'] = result_dict['NDCG@5']
		easy_result_dict['ALPHA_NDCG@5'] = result_dict['ALPHA_NDCG@5']
		easy_result_dict['NDCG@10'] = result_dict['NDCG@10']
		easy_result_dict['ALPHA_NDCG@10'] = result_dict['ALPHA_NDCG@10']
		res_str = easy_result_dict
		return res_str
	
	
	def MMR_post_process(self, dataset: BaseModel.Dataset, accweight = 0.5):
		"""
		Evaluate the results for an input dataset.
		:return: result dict (key: metric@k)
		"""
		id_multihot = dataset.model.item_multihot_mapping
		grouped_users_updated= dataset.model.grouped_users_updated
		output_dict = self.predict(dataset) 
		baseline = MMR(test_num=len(output_dict["predictions"]))  
		output_dict = baseline.post_process(output_dict, id_multihot, lamda=accweight)
		result_dict = self.evaluate_method(output_dict,self.topk, self.metrics, id_multihot,grouped_users_updated)

		easy_result_dict = {}
		easy_result_dict['HR@5'] = result_dict['HR@5']
		easy_result_dict['NDCG@5'] = result_dict['NDCG@5']
		easy_result_dict['ALPHA_NDCG@5'] = result_dict['ALPHA_NDCG@5']
		easy_result_dict['HR@10'] = result_dict['HR@10']
		easy_result_dict['NDCG@10'] = result_dict['NDCG@10']
		easy_result_dict['ALPHA_NDCG@10'] = result_dict['ALPHA_NDCG@10']
		res_str = utils.format_metric(easy_result_dict)
	
		return res_str