import warnings
warnings.simplefilter('ignore', FutureWarning)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# import resource
# import os

# os.environ["OMP_NUM_THREADS"] = "1"
# resource.setrlimit(
#     resource.RLIMIT_DATA,
#     (500 * 1024 ** 2, -1))

import sys
import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from mordred import Calculator, descriptors

from data import TestDataLoader, MORDRED_DESC
from model import TransformerModel


def get_values(df, scaler):
	smiles = df['SMILES'].values
	chem_df = df.drop(['SMILES'], axis=1)
	mols_df = df['SMILES'].apply(Chem.MolFromSmiles)

	drop_columns = pickle.load(open(os.path.dirname(__file__) + "/saved_models/drop_col.pkl", "rb"))
	scale_columns = pickle.load(open(os.path.dirname(__file__) + "/saved_models/scale_col.pkl", "rb"))

	# median_df = pickle.load(open(os.path.dirname(__file__) + "/saved_models/median.pkl", "rb"))

	fps = [
		{'name': 'maccs', 'size': 167, 'func': AllChem.GetMACCSKeysFingerprint},
		# {'name': 'rdkit', 'size': 2048, 'func': Chem.RDKFingerprint},
		# {'name': 'atompair', 'size': 2048, 'func': AllChem.GetHashedAtomPairFingerprintAsBitVect},
		{'name': 'morgan', 'size': 1024, 'func': lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024)},
		{'name': 'avalon', 'size': 512, 'func': GetAvalonFP},
	]

	for fp in fps:
		name, size, func = fp['name'], fp['size'], fp['func']
		fp_value = np.zeros((len(df), size))
		for (idx, m) in enumerate(mols_df.values):
			DataStructs.ConvertToNumpyArray(
				func(m), fp_value[idx])

		fp_columns = [f'{name}-{i}' for i in range(size)]
		fp_df = pd.DataFrame(fp_value, columns=fp_columns, dtype=float)
		chem_df = pd.concat([chem_df, fp_df], axis=1)
		
	# calc_dummy = Calculator(descriptors, ignore_3D=False)
	# custom_descriptors = []
	# for desc in calc_dummy.descriptors:
	# 	if desc.__str__() in MORDRED_DESC:
	# 		custom_descriptors.append(desc)
	# calc_real = Calculator(custom_descriptors, ignore_3D=False)
	# mordred_df = calc_real.pandas(mols_df, quiet=True)
	# for col in mordred_df.columns:
	# 	if mordred_df[col].dtypes == object:
	# 		mordred_df[col] = mordred_df[col].values.astype(np.float32)
	# chem_df = pd.concat([chem_df, mordred_df], axis=1)
	
	chem_df.drop(columns=drop_columns, inplace=True)
	chem_df.fillna(0)

	# for col in chem_df.columns:
	# 	# chem_df[col] = chem_df[col].fillna(median_df[col])

	chem_df[scale_columns] = scaler.transform(chem_df[scale_columns])

	chem_value = chem_df.values

	return smiles, chem_value

input_data = []
for line in sys.stdin:
    input_data.append(line.strip().split(","))

input_df = pd.DataFrame(data=input_data[1:], columns=input_data[0])
preprocess_models = pickle.load(open(os.path.dirname(__file__) + "/saved_models/preprocessor.pkl", "rb"))
scaler = preprocess_models["scaler"]
vocab = preprocess_models["vocab"]
mean_value, std_value = preprocess_models["mean"], preprocess_models["std"]

smiles, chem_value = get_values(input_df, scaler)

smiles_token = vocab.transform(smiles)

with open(os.path.dirname(__file__) + '/parameter.json', 'r') as f:
	param = json.load(f)

dataloader = TestDataLoader(smiles_token, chem_value, vocab, batch_size=16)

model = TransformerModel(
	ntoken=len(vocab),
	d_model=param['emsize'],
	nhead=param['nhead'],
	d_hid=param['d_hid'],
	nlayers=param['nlayers'],
	nchem_feature=len(chem_value[0]),
	dropout=param['dropout'],
)

y_preds = []
for i in range(param["nmodels"]):
	model.load_state_dict(torch.load(os.path.dirname(__file__) + f"/saved_models/model{i+1}.pth", map_location='cpu'))
	model.eval()

	y_pred = []
	for batch in dataloader:
		batch_smiles, pad_mask, batch_chem_value = batch
		output = model(batch_smiles, pad_mask, batch_chem_value)
		output = list(output.squeeze(-1).detach().numpy())
		y_pred += output

	y_preds.append(y_pred)

y_preds = np.array(y_preds).mean(axis=0)
y_preds = y_preds * std_value + mean_value

for val in y_preds:
    print(val)
