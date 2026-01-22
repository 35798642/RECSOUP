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
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List
import json
from ..utils import utils
from ..reader import BaseReader
from functools import reduce
from time import time
from typing import Dict, List

class BaseModel(nn.Module):
	reader, runner = None, None  # choose helpers in specific model classes
	extra_log_args = []


	@staticmethod
	def init_weights(m):
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.normal_(m.bias, mean=0.0, std=0.01)
		elif isinstance(m, nn.Embedding):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
	
	def __init__(self, args, corpus: BaseReader):
		super(BaseModel, self).__init__()
		self.device = args.device
		self.model_path = args.paths.model_path
		self.buffer = args.buffer
		self.optimizer = None
		self.check_list = list()  # observe tensors in check_list every check_epoch

	"""
	Key Methods
	"""
	def _define_params(self):
		pass

	def forward(self, feed_dict: dict) -> dict:
		"""
		:param feed_dict: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		pass

	def loss(self, out_dict: dict) -> torch.Tensor:
		pass

	"""
	Auxiliary Methods
	"""
	def customize_parameters(self) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'bias' in name:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict

	def save_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		utils.check_dir(model_path)
		torch.save(self.state_dict(), model_path)
   
	def load_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		self.load_state_dict(torch.load(model_path))
		logging.info('Load model from ' + model_path)

	def count_variables(self) -> int:
		total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
		return total_parameters

	def actions_after_train(self):  # e.g., save selected parameters
		pass

	"""
	Define Dataset Class
	"""
	class Dataset(BaseDataset):
		def __init__(self, model, corpus, phase: str):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test

			self.buffer_dict = dict()
			self.data = corpus.data_df[phase].to_dict('list')

		def __len__(self):
			if type(self.data) == dict:
				for key in self.data:
					return len(self.data[key])
			return len(self.data)

		def __getitem__(self, index: int) -> dict:
			if self.model.buffer and self.phase != 'train':
				return self.buffer_dict[index]
			
			return self._get_feed_dict(index)

		# ! Key method to construct input data for a single instance
		def _get_feed_dict(self, index: int) -> dict:
			pass

		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase != 'train':
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self._get_feed_dict(i)

		# Called before each training epoch (only for the training dataset)
		def actions_before_epoch(self):
			pass

		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]) -> dict:
			feed_dict = dict()
			for key in feed_dicts[0]:
				if isinstance(feed_dicts[0][key], np.ndarray):
					tmp_list = [len(d[key]) for d in feed_dicts]
					if any([tmp_list[0] != l for l in tmp_list]):
						stack_val = np.array([d[key] for d in feed_dicts], dtype=object)
					else:
						stack_val = np.array([d[key] for d in feed_dicts])
				else:
					stack_val = np.array([d[key] for d in feed_dicts])
				if stack_val.dtype == object:  # inconsistent length (e.g., history)
					feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
				else:
					feed_dict[key] = torch.from_numpy(stack_val)
			feed_dict['batch_size'] = len(feed_dicts)
			feed_dict['phase'] = self.phase
			return feed_dict

class GeneralModel(BaseModel):
	reader, runner = 'BaseReader', 'BaseRunner'

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.user_num = corpus.n_users
		self.item_num = corpus.n_items
		self.num_neg = args.num_neg
		self.dropout = args.dropout
		self.test_all = args.test_all
		if args.data.dataset == 'MovieLens_1M':
			multihot_path = os.path.join(args.data.path, args.data.dataset, 'ML_1MTOPK', 'newid2multihot.json')
			self.item_multihot_mapping = json.load(open(multihot_path, 'r'))
		elif args.data.dataset == 'Grocery_and_Gourmet_Food':
			multihot_path = os.path.join(args.data.path, args.data.dataset, args.data.dataset, 'newid2multihot.json')
			self.item_multihot_mapping = json.load(open(multihot_path, 'r'))
		elif args.dataset == 'Toys_and_Games':
			multihot_path = os.path.join(args.path, args.dataset, args.dataset, 'newid2multihot.json')
			self.item_multihot_mapping = json.load(open(multihot_path, 'r'))
		else:
			logging.error('No such multihot embedding of dataset: {}'.format(args.dataset))
			
		self.item_multihot_embeddings = self.multihot_embedding(self.item_multihot_mapping).to(self.device)

	
	def multihot_embedding(self, item_multihot_mapping):
		num_items = max(int(k) for k in self.item_multihot_mapping.keys()) + 1
		embedding_dim = len(next(iter(self.item_multihot_mapping.values())))
		item_embeddings = torch.zeros((num_items, embedding_dim), dtype=torch.float32, device='cpu')
		for key, value in self.item_multihot_mapping.items():
			item_embeddings[int(key)] = torch.tensor(value, dtype=torch.float32)
		return item_embeddings
	
	def loss(self, out_dict: dict) -> torch.Tensor:
		"""
		BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
		"Recurrent neural networks with top-k gains for session-based recommendations"
		:param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
		:return:
		"""
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
		return loss
	

	def diversity_loss(self, out_dict: dict) -> torch.Tensor:
		start_time = time()
		predictions = out_dict['prediction']  # [B, N] preds 
		ids = out_dict['item_id']  # [B, N] item ids
		category = self.item_multihot_embeddings[ids.long()]  # [B, N, M]

		predictions_1 = predictions.unsqueeze(2)  # [B, N, 1]
		predictions_2 = predictions.unsqueeze(1)  # [B, 1, N]
		sub_prediction = (predictions_2 - predictions_1)  # [B, N, N]
		R = 0.5 + torch.sigmoid(sub_prediction / 0.1).sum(dim=2)  # [B, N]
		
		category_expanded = category.unsqueeze(2)  # [B, N, 1, M]
		sub_prediction_expanded = sub_prediction.unsqueeze(3)  # [B, N, N, 1]
		sigmoid_sub = torch.sigmoid(sub_prediction_expanded / 0.1)  # [B, N, N, 1]
		category_product = (category_expanded * sigmoid_sub).sum(dim=1) - 0.5 * category  # [B, N, M]
		decay_cat = torch.pow(0.5, category_product)  # [B, N, M]
		sum_cat = (category * decay_cat).sum(dim=2)  # [B, N]
		loss = -torch.sum(sum_cat / (torch.log(R + 1) / torch.log(torch.tensor(2.0, device='cpu')))) / predictions.shape[0]
		return loss

	class Dataset(BaseModel.Dataset):
		def _get_feed_dict(self, index):
			user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
			if self.phase != 'train' and self.model.test_all:
				neg_items = np.arange(1, self.corpus.n_items)
			elif self.phase != 'train':
				neg_items = self.data['neg_items'][index] # negative items are pre-sampled
			elif self.phase == 'train':
				neg_items = self.data['neg_items'][index][:self.model.num_neg] # negative items are pre-sampled
			item_ids = np.concatenate([[target_item], neg_items]).astype(int)
			feed_dict = {
				'user_id': user_id,
				'item_id': item_ids
			}
			
			return feed_dict

		# Sample negative items for all the instances 
		def actions_before_epoch(self):
			neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
			for i, u in enumerate(self.data['user_id']):
				clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
				# clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
				for j in range(self.model.num_neg):
					while neg_items[i][j] in clicked_set:
						neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
			self.data['neg_items'] = neg_items

class SequentialModel(GeneralModel):
	reader = 'SeqReader'

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.model.history_max

	class Dataset(GeneralModel.Dataset):  # 
		def __init__(self, model, corpus, phase):
			super().__init__(model, corpus, phase)
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key],dtype=object)[idx_select].tolist()

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]  
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			feed_dict['history_times'] = np.array([x[1] for x in user_seq])
			feed_dict['lengths'] = len(feed_dict['history_items'])
			return feed_dict
