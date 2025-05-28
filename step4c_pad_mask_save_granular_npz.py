# step4c_pad_mask_save_granular_npz.py
# Reads temporary granular batch .pkl files from Step 4b (temp dir).
# Performs Padding (with -1) and Masking using NumPy. Matches outcomes.
# SKIPS SCALING for granular integer features.
# Saves final processed batches in NPZ format for model training.

import pickle
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
# from tensorflow.keras.preprocessing.sequence import pad_sequences # Not needed anymore
import traceback
import time

# ---------------------- 0. 配置 ------------------------------------
output_dir = './output_dynamic'
COHORT_KEY = 'aki' # <<<--- 明确为 AKI (或 'hf')

# --- 输入目录 ---
# Step 4b 生成的临时粒度批次文件目录
temp_granular_dir = os.path.join(output_dir, 'step4b_temp_granular')
# Step 3 的 Outcome 文件 (用于映射 y 值)
outcomes_file = os.path.join(output_dir, f'step3_dynamic_outcomes_{COHORT_KEY}.parquet')
# Step 4a 的 Bin Edges 文件 (用于确定 N_FEATURES)
bin_edges_file = os.path.join(output_dir, f'step4ab_bin_edges_{COHORT_KEY}_5bins.pkl') # Assume 5 bins

# --- 输出目录 ---
# 最终处理好的 NPZ 批次文件目录
processed_batch_dir = os.path.join(output_dir, f'step4c_processed_granular_batches_{COHORT_KEY}') # Step 4c Output
os.makedirs(processed_batch_dir, exist_ok=True)

# --- Padding 参数 ---
MAX_LEN = 240 # <<<--- 确认最大序列长度
PADDING_VALUE = -1 # <<<--- 使用 -1 作为填充值 for integer bin labels
PADDING_TYPE = 'post' # Options: 'pre', 'post'
TRUNCATING_TYPE = 'post' # Options: 'pre', 'post'

print(f"--- Step 4c (Pad/Mask Granular - NumPy): Process Temp Batches and Save NPZ ---")
print(f"Input temporary directory: {temp_granular_dir}")
print(f"Input outcomes file: {outcomes_file}")
print(f"Output processed NPZ batches directory: {processed_batch_dir}")
print(f"Max sequence length (MAX_LEN): {MAX_LEN}")
print(f"Padding value: {PADDING_VALUE}")
print(f"Padding Type: {PADDING_TYPE}, Truncating Type: {TRUNCATING_TYPE}")
print("-" * 70)

# ---------------------- 1. 确定特征数量 -------------------------
print("Determining number of granular features...")
try:
    with open(bin_edges_file, 'rb') as f: bin_edges = pickle.load(f)
    N_FEATURES_GRANULAR = len(bin_edges)
    if N_FEATURES_GRANULAR == 0: raise ValueError("No features found in bin edges file.")
    print(f"Determined {N_FEATURES_GRANULAR} granular features based on bin edges.")
except FileNotFoundError: exit(f"错误: 找不到分箱边界文件 '{bin_edges_file}'。请先运行 Step 4a。")
except Exception as e: exit(f"错误: 无法从分箱边界文件确定特征数量: {e}")
print("-" * 70)

# ---------------------- 2. 加载 Outcomes 数据用于映射 ------------
print("Loading outcomes data for mapping...")
try:
    outcomes_df = pd.read_parquet(outcomes_file)
    outcome_map = outcomes_df.set_index(['stay_id', 'prediction_hour'])['outcome_death_next_24h']
    print(f"Loaded {outcomes_df.shape[0]} outcome points.")
    del outcomes_df; gc.collect()
except FileNotFoundError: exit(f"错误: 找不到 outcome 文件 '{outcomes_file}'。")
except Exception as e: exit(f"加载 outcome 数据时出错: {e}")
print("-" * 70)

# ---------------------- 3. 查找并迭代处理临时批次文件 -------------
search_pattern = os.path.join(temp_granular_dir, f'temp_granular_features_{COHORT_KEY}_batch_*.pkl')
all_temp_batch_files = sorted(glob.glob(search_pattern))

if not all_temp_batch_files: exit(f"错误：在临时目录 '{temp_granular_dir}' 中找不到任何批次文件 (*_{COHORT_KEY}_batch_*.pkl)！")
print(f"Found {len(all_temp_batch_files)} temporary batch files to process.")
print("-" * 70)

print(f"Processing temporary batches and saving NPZ files to {processed_batch_dir}...")
failed_batches_processing = []
total_samples_processed = 0

for i, batch_file in enumerate(tqdm(all_temp_batch_files, desc="Processing Temp Batches")):
    batch_filename = os.path.basename(batch_file)
    original_batch_num = ''.join(filter(str.isdigit, batch_filename))
    output_npz_file = os.path.join(processed_batch_dir, f'processed_batch_{original_batch_num}.npz')

    try:
        with open(batch_file, 'rb') as f:
            # batch_data is a list of (stay_id, pred_hour, granular_feature_array<int32>)
            batch_data = pickle.load(f)

        if not batch_data: print(f"  Skipping empty batch file: {batch_filename}"); continue

        # --- Extract data from the loaded batch ---
        stay_ids_batch = [item[0] for item in batch_data]
        pred_hours_batch = [item[1] for item in batch_data]
        # List of variable-length integer arrays
        sequences_batch = [item[2] for item in batch_data]
        batch_size_current = len(sequences_batch)

        # --- Padding and Masking using NumPy ---
        # Initialize tensors/arrays with padding value/False
        X_padded_batch = np.full((batch_size_current, MAX_LEN, N_FEATURES_GRANULAR), PADDING_VALUE, dtype=np.int32)
        Mask_batch = np.zeros((batch_size_current, MAX_LEN), dtype=bool) # Mask is boolean

        for j, seq in enumerate(sequences_batch):
            # Handle potential empty sequences if they slipped through
            if seq is None or len(seq) == 0:
                 # print(f"Warning: Empty sequence found at index {j} in {batch_filename}")
                 continue # Skip this sequence, leave row as padding/False

            # Ensure correct number of features before padding/truncating
            if seq.shape[1] != N_FEATURES_GRANULAR:
                print(f"FATAL ERROR: Sequence at index {j} in {batch_filename} has incorrect feature dimension ({seq.shape[1]} vs {N_FEATURES_GRANULAR}). Exiting.")
                exit()

            # Truncate if necessary
            if len(seq) > MAX_LEN:
                if TRUNCATING_TYPE == 'post': seq_truncated = seq[:MAX_LEN]
                else: seq_truncated = seq[-MAX_LEN:] # Pre-truncating
                seq_len = MAX_LEN
            else:
                seq_truncated = seq
                seq_len = len(seq)

            # Copy data and set mask
            if seq_len > 0:
                if PADDING_TYPE == 'post':
                    X_padded_batch[j, :seq_len, :] = seq_truncated
                    Mask_batch[j, :seq_len] = True
                else: # Pre-padding
                    X_padded_batch[j, -seq_len:, :] = seq_truncated
                    Mask_batch[j, -seq_len:] = True

        # --- Scaling Skipped ---
        X_final_batch = X_padded_batch # Data is already granular integer labels

        # --- Get Outcomes ---
        batch_identifiers_df = pd.DataFrame({'stay_id': stay_ids_batch, 'prediction_hour': pred_hours_batch})
        batch_multi_index = pd.MultiIndex.from_frame(batch_identifiers_df)
        # Map outcomes, fill missing with -1 (should indicate an issue upstream)
        y_batch = outcome_map.reindex(batch_multi_index).fillna(-1).astype(np.int32).values

        # --- Filter out samples where outcome mapping failed ---
        valid_outcome_indices = (y_batch != -1)
        if np.sum(~valid_outcome_indices) > 0:
             # print(f"  Warning: Batch {original_batch_num} - Filtering {np.sum(~valid_outcome_indices)} samples with missing outcomes.")
             X_final_batch = X_final_batch[valid_outcome_indices]
             Mask_batch = Mask_batch[valid_outcome_indices]
             y_batch = y_batch[valid_outcome_indices]
             # Keep IDs consistent if needed for debugging/analysis later
             batch_identifiers_df = batch_identifiers_df[valid_outcome_indices]

        if X_final_batch.shape[0] == 0:
             print(f"  Skipping saving batch {original_batch_num}: No valid samples remaining after outcome mapping.")
             continue

        # --- Save processed batch as NPZ ---
        try:
             np.savez_compressed(output_npz_file,
                                 X=X_final_batch,      # Shape: (batch, MAX_LEN, N_FEATURES_GRANULAR), dtype: int32, Pad: -1
                                 Mask=Mask_batch,    # Shape: (batch, MAX_LEN), dtype: bool, False for padding
                                 y=y_batch,          # Shape: (batch,), dtype: int32
                                 # Optional: Save IDs for traceability
                                 stay_id=batch_identifiers_df['stay_id'].values,
                                 prediction_hour=batch_identifiers_df['prediction_hour'].values)
             total_samples_processed += len(y_batch)
        except Exception as e_save:
             print(f"错误: 保存 NPZ 批次 {output_npz_file} 时出错: {e_save}")
             failed_batches_processing.append(batch_filename + " (Save Failed)")

        # Clean up memory for this batch
        del batch_data, sequences_batch, X_padded_batch, Mask_batch, X_final_batch, y_batch, batch_identifiers_df
        if (i + 1) % 10 == 0: gc.collect() # GC every 10 batches

    except FileNotFoundError: print(f"错误: 找不到临时批次文件 {batch_file}，跳过。")
    except EOFError: print(f"错误: 文件 {batch_file} 可能不完整 (EOFError)，跳过。")
    except Exception as e: print(f"错误: 处理临时批次文件 {batch_file} 时发生未知错误: {e}"); traceback.print_exc(); failed_batches_processing.append(batch_filename + " (Processing Error)")

print(f"\n--- Step 4c (Pad/Mask Granular - NumPy) Completed ---")
print(f"Processed {len(all_temp_batch_files)} temporary pickle files.")
print(f"Total samples saved in NPZ files: {total_samples_processed}")
print(f"Final processed NPZ batches saved in: {processed_batch_dir}")
if failed_batches_processing: print("\n--- Failed Batches ---"); [print(f"  - {f}") for f in failed_batches_processing]
print(f"\nNext Step (Step 5): Use PyTorch DataLoader to load data from '{processed_batch_dir}' for model training.")
print(f"             Remember models will need an Embedding layer for the granular features (X is int32, padded with -1).")