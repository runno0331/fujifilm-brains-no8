import warnings
warnings.simplefilter('ignore', FutureWarning)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

from experiment import Experiment



EX = Experiment()

np.random.seed(1)

task_num = EX.get_task_num()

def get_next_index(exp_list, results, distance_list, num50):
    if len(results) < 5:
        idx = int(np.floor(5000 * np.random.rand()))
        while idx in exp_list:
            idx = int(np.floor(5000 * np.random.rand()))   
        return idx

    # best_sample = min(results, key=lambda x: x['rank'])
    results = sorted(results, key=lambda x: x['rank'])
    # best_sample = results[0]
    max_nsamples = (550 - len(results)) // 50
    # max_nsamples = 1
    
    distance = np.zeros(5000)

    count = 0
    ranks = []
    if num50 >= 30:
        pos_ratio = 0.8
    else:
        pos_ratio = 0.6

    for i in range(min(len(results) // 2, 50)):
    # for i in np.random.choice(len(results) // 5 + 1, min(20, len(results) // 5)):
        if i >= 0 and results[i]['rank'] > 2000:
            break
        if i >= 2 and results[i]['rank'] > 500:
            break
        
        if np.random.rand() < pos_ratio:
            continue

        ranks.append(results[i]['rank'])

        # print(i, results[i]['idx'])
        distance += distance_list[results[i]['idx']]
        count += 1
    
    neg_ranks = []
    max_num_neg = 30

    if num50 < 10:
        neg_ratio = 0.7
    elif num50 >= 30: # less exploration
        neg_ratio = 0.3
    # elif num50 >= 40: # much less exploration
        # neg_ratio = 0.3
    # elif num50 < 5 and len(results) < 300: # more exploration
        # neg_ratio = 0.7
    else:
        neg_ratio = 0.5
    
    for i in np.random.choice(len(results) // 4 + 1, min(max_num_neg, len(results) - count)):
        if np.random.rand() < neg_ratio:
            continue
        
        # if i > 10 and results[-(i+1)]['rank'] < 4000:
        #     break
        # if i > 5 and results[-(i+1)]['rank'] < 3000:
        #     break
        # if i > 1 and results[-(i+1)]['rank'] < 1000:
        #     break
        # if results[-(i+1)]['rank'] < 500:
        #     break
        
        neg_ranks.append(results[-(i+1)]['rank'])
        distance -= distance_list[results[-(i+1)]['idx']]
    
    # print('rank: ', ranks)
    # print('neg:  ', neg_ranks)
    for exp_idx in exp_list:
        distance[exp_idx] = np.inf

    min_cand_idx = np.argpartition(distance, max_nsamples)[:max_nsamples]
    next_index = np.random.choice(min_cand_idx)
    return next_index

def calculate_distance(data):
    fps_func = [
        lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048, useFeatures=True), # 5.9s
        # lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048, useFeatures=True), # 6.1s
        AllChem.GetMACCSKeysFingerprint, # 7.6s
        # AllChem.GetTopologicalTorsionFingerprint, # takes too much time 40.3s
        AllChem.GetHashedAtomPairFingerprintAsBitVect, # 7.7s
        # Chem.RDKFingerprint, # 8.68s
        # GetAvalonFP, # 8.7s
    ]

    mol_df = data['SMILES'].apply(Chem.MolFromSmiles).values
    distance_list = np.zeros((len(data), len(data)))

    for func in fps_func:
        fps = [func(m) for m in mol_df]

        for i in range(len(data)-1):
            # to reduce memory consumption
            distance_list[i, i+1:] += np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
            # distance_list[i, i+1:] += np.array(DataStructs.BulkDiceSimilarity(fps[i], fps[i+1:]))
    
    # ergf error

    data = data.drop(columns=['SMILES'])
    values = data.fillna(0).values
    values_scaled = StandardScaler().fit_transform(values)
    values_pca = PCA(n_components=20).fit_transform(values_scaled)

    # prevent memory consumption & calculation time
    for i in range(len(data)-1):
        distance_list[i, i+1:] -= np.linalg.norm(values_pca[i+1:] - values_pca[i], axis=1) / 50

    distance_list += distance_list.T
    distance_list = len(fps_func) - distance_list

    return distance_list

for task_idx in range(task_num):
    data = EX.get_data(task_idx+1)

    distance_list = calculate_distance(data)

    exp_list = []
    rank_list = []
    value_list = []

    results = []

    num50 = 0

    for roop_i in range(500):
        current_exp = get_next_index(exp_list, results, distance_list, num50)
        
        results_dict = EX.exp(task_id=task_idx+1, chem_id=current_exp)
        exp_list.append(current_exp)
        rank_list = np.append(rank_list, results_dict["rank"])
        value_list = np.append(value_list, results_dict["value"])

        if results_dict['rank'] <= 50:
            num50 += 1
        results.append({'idx': current_exp, 'rank': results_dict["rank"], 'value': results_dict["value"]})
    

    # print(list(map(int, rank_list))[:-50])
    # print(list(map(int, rank_list))[-50:])
    # print(list(map(int, rank_list)))
    # print(sorted(list(map(int, rank_list)))[:50])
    # print(sorted(exp_list))
