import os
import copy
import time
import json
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from data import DataLoader, read_csv_and_get_values, create_vocab_and_data
from model import TransformerModel
from utils import WarmupScheduler


class Trainer:
	def __init__(
			self,
			data_path: str,
			param_path: str,
			random_state: int = 1234,
		):

		# set seed and device
		self.set_seed(random_state)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# read data files
		smiles, self.chem_value, mvalue, self.scaler = read_csv_and_get_values(data_path)
		self.vocab, self.smiles_token = create_vocab_and_data(smiles)
		self.set_parameters(param_path)

		# scale target data
		self.mean_value, self.std_value = mvalue.mean(), mvalue.std()
		self.target = (mvalue - self.mean_value) / self.std_value

		self.ntokens = len(self.vocab)  # size of vocabulary
		self.nchem_feature = len(self.chem_value[0])

		self.criterion = nn.MSELoss()

		# for test time
		self.save_preprocessor()

	def train_loop(self) -> None:
		self.model.train()  # turn on train mode
		total_loss = 0.
		log_interval = 200
		batch_size = self.train_data.batch_size
		start_time = time.time()

		num_batches = len(self.train_data) // batch_size
		for i, batch in enumerate(self.train_data):
			smiles, pad_mask, chem_values, targets = batch
			output = self.model(smiles, pad_mask, chem_values)
			loss = self.criterion(output, targets)

			self.optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
			self.optimizer.step()
			self.scheduler.step()

			total_loss += loss.item()
			if i % log_interval == 0 and i > 0:
				lr = self.scheduler.get_last_lr()[0]
				ms_per_batch = (time.time() - start_time) * 1000 / log_interval
				cur_loss = total_loss / log_interval
				print(f'| epoch {self.epoch:3d} | {i:5d}/{num_batches:5d} batches | '
					f'lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f} | '
					f'loss {cur_loss:5.3f}')
				total_loss = 0
				start_time = time.time()

	def eval_loop(self) -> float:
		self.model.eval()  # turn on evaluation mode
		total_loss = 0.
		with torch.no_grad():
			for batch in self.val_data:
				smiles, pad_mask, chem_values, targets = batch
				output = self.model(smiles, pad_mask, chem_values)
				total_loss += smiles.size(0) * self.criterion(output, targets).item()
		return total_loss / len(self.val_data)

	def train_and_eval(self) -> None:
		kf = KFold(n_splits=self.nmodels, shuffle=True, random_state=self.random_state)
		for i, (train_index, valid_index) in enumerate(kf.split(self.smiles_token)):
			# data
			# train_smiles, valid_smiles, train_chem, valid_chem, train_target, valid_target = \
				# train_test_split(self.smiles_token, self.chem_value, self.target, test_size=0.2, random_state=self.random_state)
			train_smiles = [self.smiles_token[idx] for idx in train_index]
			valid_smiles = [self.smiles_token[idx] for idx in valid_index]
			train_chem, valid_chem = self.chem_value[train_index], self.chem_value[valid_index]
			train_target, valid_target = self.target[train_index], self.target[valid_index]

			self.train_data = DataLoader(train_smiles, train_chem, train_target, self.vocab, batch_size=self.batch_size, device=self.device)
			self.val_data = DataLoader(valid_smiles, valid_chem, valid_target, self.vocab, batch_size=self.batch_size, device=self.device, test=True)

			# models
			self.model = TransformerModel(self.ntokens, self.emsize, self.nhead, self.d_hid, self.nlayers, self.nchem_feature, self.dropout, self.linear_dropout).to(self.device)
			self.optimizer = self.model.configure_optimizer(lr=self.lr, weight_decay=self.weight_decay)
			self.scheduler = WarmupScheduler(self.optimizer, warmup_steps=2*len(self.train_data)//self.batch_size)

			best_val_loss = float('inf')
			best_model = None
			wait_count = 0
			self.epoch = 0

			while True:
				self.epoch += 1
				epoch_start_time = time.time()
				self.train_loop()

				val_loss = self.eval_loop()
				elapsed = time.time() - epoch_start_time
				print('-' * 89)
				print(f'| end of epoch {self.epoch:3d} | time: {elapsed:5.2f}s | '
					f'valid loss {val_loss:5.3f} | true valid loss {val_loss*self.std_value**2:5.2f}')
				print('-' * 89)

				if val_loss < best_val_loss:
					best_val_loss = val_loss
					best_model = copy.deepcopy(self.model)
					wait_count = 0
				else:
					wait_count += 1
				
				if wait_count == self.max_wait:
					self.save_model(best_model, f'model{i+1}')
					break
				elif wait_count == self.max_wait // 2:
					self.scheduler.coeff *= 0.8
					# self.scheduler.coeff *= 1.
				
	def set_seed(self, random_state: int) -> None:
		self.random_state = random_state
		random.seed(random_state)
		np.random.seed(random_state)
		torch.manual_seed(random_state)
		torch.cuda.manual_seed(random_state)
	
	def set_parameters(self, param_path: str) -> None:
		with open(param_path, 'r') as f:
			param = json.load(f)

		self.emsize = param['emsize']  # embedding dimension
		self.d_hid = param['d_hid']  # dimension of the feedforward network model in nn.TransformerEncoder
		self.nlayers = param['nlayers']  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
		self.nhead = param['nhead']  # number of heads in nn.MultiheadAttention
		self.dropout = param['dropout']  # dropout probability
		self.linear_dropout = param['linear_dropout']
		self.batch_size = param['batch_size']
		self.max_wait = param['max_wait']
		self.lr = param['lr']  # learning rate
		self.weight_decay = param['weight_decay']
		self.nmodels = param['nmodels']  # number of models to train

	def save_model(self, model: nn.Module, model_name: str) -> None:
		torch.save(model.state_dict(), os.path.dirname(__file__) +  f"/saved_models/{model_name}.pth")
		print("saved model!")

	def save_preprocessor(self) -> None:
		save_dict = dict(scaler=self.scaler, vocab=self.vocab, mean=self.mean_value, std=self.std_value)
		pickle.dump(save_dict, open(os.path.dirname(__file__) +  "/saved_models/preprocessor.pkl", "wb"))
		print("saved preprocessor!")
