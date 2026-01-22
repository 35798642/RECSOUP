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
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from ..utils import utils
class BaseReader(object):
  
    def __init__(self, args):
        self.sep = args.data.sep
        self.prefix = args.data.path
        self.dataset = args.data.dataset

        self._read_data()

        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            print(os.path.join(self.prefix, self.dataset))
            if self.dataset == 'MovieLens_1M':
                self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset ,'ML_1MTOPK',key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
            elif self.dataset == 'Grocery_and_Gourmet_Food':
                self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset ,self.dataset,key+ '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
            elif self.dataset == 'Toys_and_Games':
                self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset ,self.dataset,key+ '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        key_columns = ['user_id','item_id','time']
        if 'label' in self.data_df['train'].columns: # Add label for CTR prediction
            key_columns.append('label')
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        for key in ['dev', 'test']: 
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))
        if 'label' in key_columns:
            positive_num = (self.all_df.label==1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
				positive_num, positive_num/self.all_df.shape[0]*100))
        
