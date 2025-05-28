# step4c_pad_mask_save_granular_npz_final.py
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
# from tensorflow.keras.preprocessing.sequence import pad_sequences # Using NumPy manual padding
import traceback
import time

# ---------------------- 0. 配置 ------------------------------------
output_dir = './output_dynamic'
COHORT_KEY = 'aki' # <<<--- 明确为 AKI

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

print(f"--- Step 4c (Pad/Mask Granular - NumPy Final): Process Temp Batches and Save NPZ ---")
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
total_samples_saved = 0
npz_file_index = 1

for i, batch_file in enumerate(tqdm(all_temp_batch_files, desc="Processing Temp Batches")):
    batch_filename = os.path.basename(batch_file)
    original_batch_num = ''.join(filter(str.isdigit, batch_filename))
    output_npz_file = os.path.join(processed_batch_dir, f'processed_batch_{original_batch_num}.npz')

    # Initialize lists for the current batch inside the loop
    stay_ids_batch = []
    pred_hours_batch = []
    sequences_batch = []

    try:
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f) # List of (stay_id, pred_hour, granular_feature_array<int32>)

        if not batch_data:
            print(f"  Skipping empty batch file: {batch_filename}")
            continue

        # --- Extract data safely ---
        for item in batch_data:
            # Basic validation of item structure
            if len(item) == 3 and isinstance(item[2], np.ndarray) and item[2].ndim == 2:
                # Check feature dimension before appending
                if item[2].shape[1] == N_FEATURES_GRANULAR:
                    stay_ids_batch.append(item[0])
                    pred_hours_batch.append(item[1])
                    sequences_batch.append(item[2].astype(np.int32)) # Ensure int32
                else:
                     print(f"Warning: Invalid feature dimension ({item[2].shape[1]} vs {N_FEATURES_GRANULAR}) in {batch_filename}. Skipping item.")
            # else: print(f"Warning: Invalid item structure in {batch_filename}. Skipping item.")

        batch_size_current = len(sequences_batch)
        if batch_size_current == 0:
            print(f"  Skipping batch file with no valid sequences: {batch_filename}")
            continue

        # --- Padding and Masking using NumPy ---
        X_padded_batch = np.full((batch_size_current, MAX_LEN, N_FEATURES_GRANULAR), PADDING_VALUE, dtype=np.int32)
        Mask_batch = np.zeros((batch_size_current, MAX_LEN), dtype=bool)

        for j, seq in enumerate(sequences_batch):
            if seq is None or len(seq) == 0: continue # Should not happen if filtered above

            if len(seq) > MAX_LEN: # Apply truncation
                if TRUNCATING_TYPE == 'post': seq_truncated = seq[:MAX_LEN]
                else: seq_truncated = seq[-MAX_LEN:]
                seq_len = MAX_LEN
            else:
                seq_truncated = seq
                seq_len = len(seq)

            if seq_len > 0: # Apply padding
                if PADDING_TYPE == 'post':
                    X_padded_batch[j, :seq_len, :] = seq_truncated
                    Mask_batch[j, :seq_len] = True
                else: # Pre-padding
                    X_padded_batch[j, -seq_len:, :] = seq_truncated
                    Mask_batch[j, -seq_len:] = True

        # --- Scaling Skipped ---
        X_final_batch = X_padded_batch

        # --- Get Outcomes ---
        # Ensure stay_ids_batch and pred_hours_batch are defined and populated correctly here
        if not stay_ids_batch or not pred_hours_batch or len(stay_ids_batch) != batch_size_current:
             print(f"Error: ID list mismatch before outcome mapping in {batch_filename}. Skipping.")
             failed_batches_processing.append(batch_filename + " (ID Mismatch)")
             continue

        batch_identifiers_df = pd.DataFrame({'stay_id': stay_ids_batch, 'prediction_hour': pred_hours_batch})
        batch_multi_index = pd.MultiIndex.from_frame(batch_identifiers_df)
        y_batch = outcome_map.reindex(batch_multi_index).fillna(-1).astype(np.int32).values

        # --- Filter out samples with missing outcomes ---
        valid_outcome_indices = (y_batch != -1)
        if np.sum(~valid_outcome_indices) > 0:
             # print(f"  Warning: Batch {original_batch_num} - Filtering {np.sum(~valid_outcome_indices)} samples with missing outcomes.")
             X_final_batch = X_final_batch[valid_outcome_indices]
             Mask_batch = Mask_batch[valid_outcome_indices]
             y_batch = y_batch[valid_outcome_indices]
             # Keep consistent IDs if saving them
             stay_ids_final = np.array(stay_ids_batch)[valid_outcome_indices]
             pred_hours_final = np.array(pred_hours_batch)[valid_outcome_indices]
        else:
             stay_ids_final = np.array(stay_ids_batch)
             pred_hours_final = np.array(pred_hours_batch)


        if X_final_batch.shape[0] == 0:
             print(f"  Skipping saving batch {original_batch_num}: No valid samples remaining after outcome mapping.")
             continue

        # --- Save processed batch as NPZ ---
        try:
             np.savez_compressed(output_npz_file,
                                 X=X_final_batch,      # Shape: (current_batch_valid_size, MAX_LEN, N_FEATURES_GRANULAR), dtype: int32
                                 Mask=Mask_batch,    # Shape: (current_batch_valid_size, MAX_LEN), dtype: bool
                                 y=y_batch,          # Shape: (current_batch_valid_size,), dtype: int32
                                 # Optional: Save IDs
                                 stay_id=stay_ids_final,
                                 prediction_hour=pred_hours_final)
             total_samples_saved += len(y_batch)
        except Exception as e_save: print(f"错误: 保存 NPZ 批次 {output_npz_file} 时出错: {e_save}"); failed_batches_processing.append(batch_filename + " (Save Failed)")

        # Clean up memory for this batch
        del batch_data, sequences_batch, X_padded_batch, Mask_batch, X_final_batch, y_batch
        del stay_ids_batch, pred_hours_batch, batch_identifiers_df, batch_multi_index # Delete lists too
        if (i + 1) % 10 == 0: gc.collect()

    except FileNotFoundError: print(f"错误: 找不到临时批次文件 {batch_file}，跳过。")
    except EOFError: print(f"错误: 文件 {batch_file} 可能不完整 (EOFError)，跳过。")
    except Exception as e: print(f"错误: 处理临时批次文件 {batch_file} 时发生未知错误: {e}"); traceback.print_exc(); failed_batches_processing.append(batch_filename + " (Processing Error)")

print(f"\n--- Step 4c (Pad/Mask Granular - NumPy Final) Completed ---")
print(f"Processed {len(all_temp_batch_files)} temporary pickle files.")
print(f"Total samples saved in NPZ files: {total_samples_saved}")
print(f"Final processed NPZ batches saved in: {processed_batch_dir}")
if failed_batches_processing: print("\n--- Failed Batches ---"); [print(f"  - {f}") for f in failed_batches_processing]
print(f"\nNext Step (Step 5): Use PyTorch DataLoader to load data from '{processed_batch_dir}' for model training.")
print(f"             Remember models will need an Embedding layer.")