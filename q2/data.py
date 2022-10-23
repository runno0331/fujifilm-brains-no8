from distutils.sysconfig import customize_compiler
import os
import re
from sklearn.utils import shuffle
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from mordred import Calculator, descriptors
import time


PAD = 0
UNK = 1
PRED = 2

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PRED_TOKEN = '<PRED>'

MORDRED_DESC = [
	# 'GATS3s', 'GATS6s', 'BCUTc-1h', 'BCUTc-1l', 'BCUTd-1h', 
	# 'BCUTd-1l', 'BCUTs-1l', 'BCUTv-1h', 'BCUTare-1l', 'BCUTp-1h', 
	# 'BCUTi-1h', 'BCUTi-1l', 'nBondsD', 'RPCG', 'C2SP2', 
	# 'SdsCH', 'SdssC', 'SdsN', 'SaaN', 'MAXdssC', 'MAXsssN', 
	# 'MINsssN', 'ETA_beta_ns_d', 'MDEN-12', 'MDEN-23', 'MID_N', 
	# 'piPC6', 'piPC7', 'piPC10', 'SMR', 'TopoPSA'
]



class BasicSmilesTokenizer(object):
	"""
		regex_pattern from https://deepchem.readthedocs.io/en/2.4.0/api_reference/tokenizers.html#basicsmilestokenizer
	"""
	def __init__(self):
		self.regex_pattern = '(\\[[^\\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|\n#|-|\\+|\\\\|\\/|:|~|@|\\?|>>?|\\*|\\$|\\%[0-9]{2}|[0-9])'
		self.regex = re.compile(self.regex_pattern)

	def tokenize(self, text):
		tokens = [token for token in self.regex.findall(text)]
		# tokens = list(text)
		return tokens


class SmilesVocab(object):
	def __init__(self, word2id, tokenizer=BasicSmilesTokenizer()):
		self.word2id = dict(word2id)
		self.id2word = {v: k for k, v in self.word2id.items()}    
		self.tokenizer = tokenizer
		
	def build_vocab(self, sentences, min_count=1):
		word_counter = {}
		for sentence in sentences:
			token_sentence = self.tokenizer.tokenize(sentence)
			for word in token_sentence:
				word_counter[word] = word_counter.get(word, 0) + 1
		
		for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
			if count < min_count:
				break
			_id = len(self.word2id)
			self.word2id.setdefault(word, _id)
			self.id2word[_id] = word 

	def smiles_to_ids(self, smiles):
		ids = [self.word2id.get(word, UNK) for word in self.tokenizer.tokenize(smiles)]
		ids = [PRED] + ids
		return ids

	def transform(self, smiles_data):
		data = [self.smiles_to_ids(smiles) for smiles in smiles_data]
		return data
	
	def __len__(self):
		return len(self.word2id)


def create_vocab_and_data(smiles_data):
	word2id = {
		PAD_TOKEN: PAD,
		UNK_TOKEN: UNK,
		PRED_TOKEN: PRED,
	}

	vocab = SmilesVocab(word2id=word2id)
	vocab.build_vocab(smiles_data)
	data = vocab.transform(smiles_data)

	return vocab, data


class DataLoader(object):
	def __init__(self, src, chem_value, target, vocab, batch_size, device, shuffle=True, random_state=1234, max_mask=10, test=False):
		self.data = list(zip(src, chem_value, target))

		self.batch_size = batch_size
		self.vocab = vocab
		self.device = device
		self.shuffle = shuffle
		self.random_state = random_state
		self.start_index = 0
		self.max_mask = max_mask
		self.mask_ratio = 0.1
		self.test = test
		
		self.reset()
	
	def reset(self):
		if self.shuffle:
			self.data = shuffle(self.data, random_state=self.random_state)
		self.start_index = 0
	
	def __len__(self):
		return len(self.data)
	
	def __iter__(self):
		return self
	
	def __next__(self):
		def preprocess_seqs(seqs):
			max_length = max([len(s) for s in seqs])
			if not self.test:
				for i in range(len(seqs)):
					for _ in range(self.max_mask):
						idx = np.random.randint(1, len(seqs[i]))
						# mask [*] randomly for generalization
						if np.random.rand() < self.mask_ratio and self.vocab.id2word[seqs[i][idx]][0] == '[':
							seqs[i][idx] = UNK
			data = [s + [PAD] * (max_length - len(s)) for s in seqs]
			# positions = [[pos+1 if w != PAD else 0 for pos, w in enumerate(seq)] for seq in data]
			masks = [[w == PAD for w in seq] for seq in data]
			data_tensor = torch.tensor(data, dtype=torch.long, device=self.device)
			mask_tensor = torch.tensor(masks, dtype=torch.bool, device=self.device)
			return data_tensor, mask_tensor            

		if self.start_index >= len(self.data):
			self.reset()
			raise StopIteration()

		src_seqs, chem_value, target = zip(*self.data[self.start_index:self.start_index+self.batch_size])
		src, pad_mask = preprocess_seqs(src_seqs)
		chem_value = torch.tensor(chem_value, dtype=torch.float, device=self.device)
		target = torch.tensor(target, dtype=torch.float, device=self.device).unsqueeze(-1)

		self.start_index += self.batch_size

		return src, pad_mask, chem_value, target


class TestDataLoader(object):
	def __init__(self, src, chem_value, vocab, batch_size):
		self.data = list(zip(src, chem_value))

		self.batch_size = batch_size
		self.vocab = vocab
		self.shuffle = shuffle
		self.start_index = 0
		
		self.reset()
	
	def reset(self):
		self.start_index = 0
	
	def __len__(self):
		return len(self.data)
	
	def __iter__(self):
		return self
	
	def __next__(self):
		def preprocess_seqs(seqs):
			max_length = max([len(s) for s in seqs])
			data = [s + [PAD] * (max_length - len(s)) for s in seqs]
			# positions = [[pos+1 if w != PAD else 0 for pos, w in enumerate(seq)] for seq in data]
			masks = [[w == PAD for w in seq] for seq in data]
			data_tensor = torch.tensor(data, dtype=torch.long)
			mask_tensor = torch.tensor(masks, dtype=torch.bool)
			return data_tensor, mask_tensor            

		if self.start_index >= len(self.data):
			self.reset()
			raise StopIteration()

		src_seqs, chem_value = zip(*self.data[self.start_index:self.start_index+self.batch_size])
		src, pad_mask = preprocess_seqs(src_seqs)
		chem_value = torch.tensor(chem_value, dtype=torch.float)

		self.start_index += self.batch_size

		return src, pad_mask, chem_value


def read_csv_and_get_values(path: str):
	df = pd.read_csv(path)
	smiles = df['SMILES'].values
	mvalue = df['λmax'].values
	mols_df = df['SMILES'].apply(Chem.MolFromSmiles)
	chem_df = df.drop(['SMILES', 'λmax'], axis=1)

	fps = [
		{'name': 'maccs', 'size': 167, 'func': AllChem.GetMACCSKeysFingerprint},
		# {'name': 'rdkit', 'size': 2048, 'func': Chem.RDKFingerprint},
		# {'name': 'atompair', 'size': 2048, 'func': AllChem.GetHashedAtomPairFingerprintAsBitVect},
		{'name': 'morgan', 'size': 1024, 'func': lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024)},
		{'name': 'avalon', 'size': 512, 'func': GetAvalonFP},
	]

	for fp in fps:
		start = time.time()
		name, size, func = fp['name'], fp['size'], fp['func']
		fp_value = np.zeros((len(df), size))
		print(name)
		for (idx, m) in enumerate(mols_df.values):
			DataStructs.ConvertToNumpyArray(
				func(m), fp_value[idx])

		fp_columns = [f'{name}-{i}' for i in range(size)]
		fp_df = pd.DataFrame(fp_value, columns=fp_columns, dtype=float)
		chem_df = pd.concat([chem_df, fp_df], axis=1)

		print(time.time() - start)
	
	print('mordred')
	
	start = time.time()
	# if os.path.exists('datasets/mordred.csv'):
	# 	mordred_df = pd.read_csv('datasets/mordred.csv')[MORDRED_DESC]
	# else:
	# 	calc_dummy = Calculator(descriptors, ignore_3D=False)
	# 	custom_descriptors = []
	# 	for desc in calc_dummy.descriptors:
	# 		if desc.__str__()  in MORDRED_DESC:
	# 			custom_descriptors.append(desc)
	# 	calc_real = Calculator(custom_descriptors, ignore_3D=False)
	# 	mordred_df = calc_real.pandas(mols_df)
	# 	for col in mordred_df.columns:
	# 		if mordred_df[col].dtypes == object:
	# 			mordred_df[col] = mordred_df[col].values.astype(np.float32)
	# chem_df = pd.concat([chem_df, mordred_df], axis=1)

	print(time.time() - start)

	# pickle.dump(chem_df.median(), open(os.path.dirname(__file__) +  "/saved_models/median.pkl", "wb"))
	# print("saved median!")

	chem_df = chem_df.fillna(0)
	drop_columns = []
	scale_columns = []
	for col in chem_df.columns:
		if len(chem_df[col].unique()) == 1:
			drop_columns.append(col)
		elif (chem_df[col].max() - chem_df[col].min()) / chem_df[col].std() > 50:
			drop_columns.append(col)
		elif not ('maccs' in col or 'morgan' in col or 'avalon' in col or 'rdkit' in col):
			scale_columns.append(col)
		# chem_df[col] = chem_df[col].fillna(chem_df[col].median())

	chem_df = chem_df.drop(columns=drop_columns)

	print("scaling")
	start = time.time()
	# scaler = StandardScaler()
	scaler = RobustScaler()
	chem_df[scale_columns] = scaler.fit_transform(chem_df[scale_columns])
	print(time.time() - start)

	chem_value = chem_df.values

	pickle.dump(drop_columns, open(os.path.dirname(__file__) +  "/saved_models/drop_col.pkl", "wb"))
	pickle.dump(scale_columns, open(os.path.dirname(__file__) +  "/saved_models/scale_col.pkl", "wb"))
	print('saved columns')

	return smiles, chem_value, mvalue, scaler
