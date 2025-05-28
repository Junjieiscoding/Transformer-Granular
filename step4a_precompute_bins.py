# step4a_precompute_bins_v2.py
# Corrects Dask quantile calculation using groupby().apply().

import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import traceback # <<<--- 导入 traceback

# ---------------------- 0. 配置 ------------------------------------
output_dir = './output_dynamic'
os.makedirs(output_dir, exist_ok=True)

COHORT_KEY = 'aki'
N_BINS = 5
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
SEED = 42

cohort_file = os.path.join(output_dir, f'step1b_cohort_timestamps_{COHORT_KEY}.parquet')
events_file = os.path.join(output_dir, f'step2_merged_events_{COHORT_KEY}_core.parquet')
bin_edges_file = os.path.join(output_dir, f'step4ab_bin_edges_{COHORT_KEY}_{N_BINS}bins.pkl')

ITEMIDS_TO_BIN = [
    # <<<--- 粘贴你最终为 AKI 确定的数值特征列表 --->>>
    220045, 220181, 220210, 220277, 223762, 223901, 50820, 50971, 50912, 51006,
    50882, 50813, 51221, 51222, 51265, 51301, 50910, 50911, 51082, 51104,
    51097, 50964, 50835, 226559
]
NUMERIC_ITEMIDS_TO_BIN = [int(i) for i in ITEMIDS_TO_BIN]

print(f"--- Step 4a-PreComputeBins V2 ({COHORT_KEY.upper()}): Calculate Global Bin Edges ---")
# ... (打印配置信息) ...
print("-" * 70)

# ---------------------- 1. 识别训练集 Stays (保持不变) ---------------
# ... (与上一个版本相同的代码，用于获取 train_stay_ids_set) ...
print("Identifying training set stays...")
try:
    cohort_df = pd.read_parquet(cohort_file)
    all_stay_ids = cohort_df['stay_id'].unique()
    temp_indices = np.arange(len(all_stay_ids))
    train_val_indices, _ = train_test_split(temp_indices, test_size=TEST_SPLIT, random_state=SEED)
    if not train_val_indices.size: exit("Error: Train+Val set indices empty.")
    train_val_stay_ids = all_stay_ids[train_val_indices]
    relative_val_size = VAL_SPLIT / (1 - TEST_SPLIT) if (1 - TEST_SPLIT) > 0 else 0
    if not (0 < relative_val_size < 1): relative_val_size = 0.15
    if len(train_val_stay_ids) >= 2:
        train_indices_in_tv, _ = train_test_split(np.arange(len(train_val_stay_ids)), test_size=relative_val_size, random_state=SEED)
        original_train_indices = train_val_indices[train_indices_in_tv]
    else: original_train_indices = train_val_indices
    train_stay_ids = all_stay_ids[original_train_indices]
    train_stay_ids_set = set(train_stay_ids)
    print(f"Identified {len(train_stay_ids_set)} training stay_ids.")
    if not train_stay_ids_set: exit("Error: Training set is empty.")
except Exception as e: exit(f"确定训练集时出错: {e}\n{traceback.format_exc()}")
print("-" * 70)

# ---------------------- 2. 使用 Dask 读取训练集事件并计算分位数 (修正版) ---
print(f"Loading training events and calculating {N_BINS}-quantile bin edges using Dask...")
bin_edges = {}
quantiles_to_compute = np.linspace(0, 1, N_BINS + 1)

try:
    ddf = dd.read_parquet(
        events_file,
        columns=['stay_id', 'itemid', 'numeric_value'],
        filters=[('stay_id', 'in', list(train_stay_ids_set)),
                 ('itemid', 'in', NUMERIC_ITEMIDS_TO_BIN)]
    )
    ddf = ddf.dropna(subset=['numeric_value'])

    # --- 修正 Dask 分位数计算 ---
    # 使用 groupby().apply() 结合 pandas 的 quantile
    # 需要定义输出的元信息 (meta)
    meta = pd.Series(dtype='float64', name='numeric_value') # apply 输出的是 Series

    print(f"  Calculating {len(quantiles_to_compute)} quantiles for each feature using groupby.apply...")

    # 应用 quantile 函数到每个 itemid 组
    quantile_results_ddf = ddf.groupby('itemid')['numeric_value'].apply(
        lambda x: x.quantile(quantiles_to_compute),
        meta=meta
    )

    # --- 计算结果 ---
    print("  Computing quantiles (this may take time)...")
    with ProgressBar():
        quantile_results = quantile_results_ddf.compute() # Compute the results
    print("  Quantile computation finished.")
    # quantile_results 现在是一个 Pandas Series，索引是 MultiIndex (itemid, quantile_level)

    # 处理结果 (与之前相同)
    if isinstance(quantile_results.index, pd.MultiIndex):
        for itemid in NUMERIC_ITEMIDS_TO_BIN:
             if itemid in quantile_results.index.get_level_values('itemid'):
                 edges = quantile_results.loc[itemid].unique().tolist()
                 edges = sorted(edges)
                 if len(edges) >= 2:
                     final_edges = [-np.inf] + edges[1:-1] + [np.inf]
                     bin_edges[itemid] = final_edges
                 else: print(f"    警告: ItemID {itemid} 分位数不足 ({edges})，跳过。")
             else: print(f"    警告: ItemID {itemid} 未在结果中找到，跳过。")
    else: print("错误: 分位数计算结果格式不符合预期。")

except FileNotFoundError: exit(f"错误: 找不到核心事件文件 '{events_file}'。")
except ImportError: exit("错误: Dask 未安装。请运行 'pip install \"dask[dataframe]\"'。")
except Exception as e:
    print(f"错误: 使用 Dask 计算分位数时出错: {e}\n{traceback.format_exc()}") # traceback 现在已定义
print(f"\nComputed bin edges for {len(bin_edges)} features.")
print("-" * 70)

# ---------------------- 3. 保存分箱边界 (保持不变) -------------------
# ... (与上一个版本相同的保存代码) ...
if not bin_edges: print("错误：未能计算出任何有效的分箱边界，无法保存。")
else:
    print(f"Saving calculated bin edges to: {bin_edges_file}")
    try:
        with open(bin_edges_file, 'wb') as f: pickle.dump(bin_edges, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Bin edges saved successfully!")
    except Exception as e: exit(f"\n错误: 保存 Pickle 文件时出错: {e}")

print(f"\n--- Step 4a-PreComputeBins V2 ({COHORT_KEY.upper()}) Completed ---")
print(f"Next Step: Run 'step4b_generate_granular_sequences_batch.py'")
print(f"  loading bin edges from '{bin_edges_file}'.")