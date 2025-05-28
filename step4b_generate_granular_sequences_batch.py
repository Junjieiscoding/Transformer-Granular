# step4b_generate_granular_sequences_batch_serial.py
# Final robust version for Step 4b.
# Applies binning using pre-computed edges.
# Generates variable-length granular sequences.
# Processes stays SERIALLY to avoid parallelization issues.
# Saves results in batches to temporary pickle files to manage memory.
# Includes fix for FutureWarning and improved error handling.

import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import pickle
import traceback
import time
import warnings # To suppress FutureWarning

# ---------------------- 0. 配置 ------------------------------------
# Suppress the specific FutureWarning about downcasting
# warnings.filterwarnings('ignore', category=FutureWarning, module='pandas.core.frame')
# More specific suppression if possible (might need exact message)
warnings.filterwarnings('ignore', message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated")


output_dir = './output_dynamic'
temp_dir = os.path.join(output_dir, 'step4b_temp_granular') # Temp dir for granular data
os.makedirs(temp_dir, exist_ok=True)

COHORT_KEY = 'aki' # <<<--- 明确为 AKI (或 'hf')

# 输入文件
cohort_file = os.path.join(output_dir, f'step1b_cohort_timestamps_{COHORT_KEY}.parquet')
events_file = os.path.join(output_dir, f'step2_merged_events_{COHORT_KEY}_core.parquet')
outcomes_file = os.path.join(output_dir, f'step3_dynamic_outcomes_{COHORT_KEY}.parquet')
# Attempt to load bin edges file, assuming 5 bins was calculated in Step 4a
bin_edges_file = os.path.join(output_dir, f'step4ab_bin_edges_{COHORT_KEY}_5bins.pkl')

# 输出文件 (最终合并的文件名 - 由下一个脚本 Step 4c 创建)
final_granular_features_file = os.path.join(output_dir, f'step4c_granular_features_variable_length_{COHORT_KEY}_core.pkl')
# 临时文件名前缀
temp_file_prefix = os.path.join(temp_dir, f'temp_granular_features_{COHORT_KEY}_batch_')

# --- 特征工程与分箱参数 ---
try:
    print(f"Loading bin edges from: {bin_edges_file}")
    with open(bin_edges_file, 'rb') as f: bin_edges = pickle.load(f)
    ITEMIDS_TO_BIN = sorted(list(bin_edges.keys()))
    if not ITEMIDS_TO_BIN: raise ValueError("Bin edges file is empty or invalid.")
    N_BINS = len(next(iter(bin_edges.values()))) - 1
    if N_BINS < 1: raise ValueError("Invalid number of bins detected from edges file.")
    N_FEATURES_GRANULAR = len(ITEMIDS_TO_BIN)
    print(f"Successfully loaded bin edges for {N_FEATURES_GRANULAR} features with {N_BINS} bins.")
except FileNotFoundError: exit(f"错误: 找不到分箱边界文件 '{bin_edges_file}'。请先运行 Step 4a。")
except Exception as e: exit(f"错误: 加载或解析分箱边界时出错: {e}")

# --- 分批保存参数 ---
BATCH_SIZE = 1000 # Process N stays before saving a temporary file (adjust based on memory)

print(f"--- Step 4b (Serial Processing): Generate Dynamic Granular Features ---")
print(f"Input events: {events_file}")
print(f"Input outcomes: {outcomes_file}")
print(f"Input bin edges: {bin_edges_file}")
print(f"Temporary batch output directory: {temp_dir}")
print(f"Final output file (to be merged later): {final_granular_features_file}")
print(f"Granulating {N_FEATURES_GRANULAR} core itemids into {N_BINS} bins.")
print(f"Processing Stays Serially in Batches of: {BATCH_SIZE}")
print("-" * 70)

# ---------------------- 1. 加载所需数据 -----------------------------
print("Loading data...")
try:
    cohort_df = pd.read_parquet(cohort_file)
    events_df = pd.read_parquet(events_file)
    events_df = events_df[events_df['itemid'].isin(ITEMIDS_TO_BIN)].copy()
    if events_df.empty: exit("错误：核心事件文件中没有需要进行粒化 (分箱) 的 ItemID 数据。")
    outcomes_df = pd.read_parquet(outcomes_file)
    intime_map = cohort_df.set_index('stay_id')['intime']
    print(f"Loaded {outcomes_df.shape[0]} prediction points.")
    print(f"Loaded {events_df.shape[0]} relevant core events for granulation.")
except Exception as e: exit(f"加载数据时出错: {e}\n{traceback.format_exc()}")
print("-" * 70)


# ---------------------- 2. 动态粒度特征生成函数 (内存优化版) -----------
def create_granular_features_for_stay(stay_events, stay_outcomes, stay_intime,
                                      itemids_to_bin_list, bin_edges_dict, n_bins):
    """Generates hourly GRANULAR (binned integer labels) & forward-filled features."""
    stay_results = []
    stay_id_for_error = stay_outcomes['stay_id'].iloc[0] if not stay_outcomes.empty else "Unknown"
    if stay_events.empty or stay_outcomes.empty or pd.isna(stay_intime): return stay_results

    final_granular_feature_cols = sorted([f'granule_{itemid}' for itemid in itemids_to_bin_list])
    n_features_expected = len(final_granular_feature_cols)

    try:
        # --- 1. Calculate hour and Apply Binning ---
        stay_events.loc[:, 'hour'] = ((stay_events['charttime'] - stay_intime).dt.total_seconds() / 3600).apply(np.floor).astype(int)
        for itemid in itemids_to_bin_list:
            granule_col_name = f'granule_{itemid}'
            if granule_col_name not in stay_events.columns: stay_events[granule_col_name] = pd.NA
            if itemid in bin_edges_dict:
                 edges = bin_edges_dict[itemid]
                 labels = range(n_bins)
                 mask_item = (stay_events['itemid'] == itemid)
                 if mask_item.any():
                      numeric_vals = stay_events.loc[mask_item, 'numeric_value'].astype(float)
                      bin_labels = pd.cut(numeric_vals, bins=edges, labels=False, right=True, include_lowest=True, duplicates='drop')
                      stay_events.loc[bin_labels.index, granule_col_name] = bin_labels.astype('Int64')

        # --- 2. Aggregate Binned Values per Hour (using last observation) ---
        stay_events.sort_values(by=['hour', 'charttime'], inplace=True)
        cols_to_agg = final_granular_feature_cols
        cols_that_exist = [col for col in cols_to_agg if col in stay_events.columns]
        if not cols_that_exist: return stay_results
        hourly_granular_df = stay_events.groupby('hour')[cols_that_exist].last()

        # --- 3. Ensure all hours and feature columns exist ---
        max_observed_event_hour = stay_events['hour'].max() if not stay_events.empty else -1
        max_hour_index = max(0, max_observed_event_hour)
        full_hour_index = pd.RangeIndex(start=0, stop=max_hour_index + 1, name='hour')
        hourly_granular_df = hourly_granular_df.reindex(full_hour_index)
        for col in final_granular_feature_cols:
            if col not in hourly_granular_df.columns: hourly_granular_df[col] = pd.NA
        hourly_granular_df = hourly_granular_df[final_granular_feature_cols]

        # --- 4. Forward Fill and Type Conversion (Optimized for Warning) ---
        # Use the optimized ffill logic from the warning fix
        filled_hourly_granular_df = hourly_granular_df.ffill()
        # Explicitly fill NA and convert type column by column
        for col in filled_hourly_granular_df.columns:
            filled_hourly_granular_df[col] = filled_hourly_granular_df[col].fillna(-1).astype(np.int32)
        # filled_hourly_granular_df = filled_hourly_granular_df.astype(np.int32) # Final ensure type

    except Exception as e_bin_agg:
        print(f"\nError granulating/aggregating stay {stay_id_for_error}: {e_bin_agg}")
        traceback.print_exc(); return stay_results

    # --- 5. Extract sequence for each prediction point ---
    max_final_hour_index = filled_hourly_granular_df.index.max() if not filled_hourly_granular_df.empty else -1
    for _, pred_row in stay_outcomes.iterrows():
        pred_hour = pred_row['prediction_hour']
        if pred_hour > max_final_hour_index: continue
        try:
            # Use .iloc for positional slicing which might be slightly faster if index is RangeIndex
            feature_array = filled_hourly_granular_df.iloc[0:pred_hour + 1].values
            # Shape check
            expected_shape = (pred_hour + 1, n_features_expected)
            if feature_array.shape == expected_shape:
                 stay_results.append((pred_row['stay_id'], pred_hour, feature_array))
            # else: print(f"WARN: Shape mismatch stay {pred_row['stay_id']} hr {pred_hour}.")
        except Exception as e_seq: print(f"\nError generating sequence stay {pred_row['stay_id']} hr {pred_hour}: {e_seq}")
    return stay_results


# ---------------------- 3. 迭代处理所有 Stays (串行, 分批保存) --------
print("Processing granular features serially (saving in batches)...")
all_results_list = [] # This will not be used to store ALL results anymore
grouped_events = events_df.groupby('stay_id')
grouped_outcomes = outcomes_df.groupby('stay_id')
stay_ids_to_process = outcomes_df['stay_id'].unique()
del events_df; gc.collect() # Release memory

current_batch_results = []
batch_num = 1
total_processed_points = 0
processed_stays_count = 0
start_time_processing = time.time()

for stay_id in tqdm(stay_ids_to_process, desc=f"Generating Granular Features for {COHORT_KEY.upper()}"):
    if stay_id not in grouped_events.groups or stay_id not in grouped_outcomes.groups: continue
    stay_events = grouped_events.get_group(stay_id).copy()
    stay_outcomes = grouped_outcomes.get_group(stay_id)
    stay_intime = intime_map.get(stay_id, pd.NaT)

    # Call the function to create granular features
    stay_granular_features = create_granular_features_for_stay(
        stay_events, stay_outcomes, stay_intime,
        ITEMIDS_TO_BIN, bin_edges, N_BINS
    )

    if stay_granular_features:
        current_batch_results.extend(stay_granular_features) # Add results for this stay to current batch
        total_processed_points += len(stay_granular_features)

    processed_stays_count += 1
    del stay_events, stay_outcomes, stay_granular_features # Clean up stay-specific data
    if processed_stays_count % 100 == 0: gc.collect() # More frequent GC

    # Save batch logic based on number of STAYS processed
    if processed_stays_count % BATCH_SIZE == 0 or processed_stays_count == len(stay_ids_to_process):
        if current_batch_results:
            batch_file_path = f"{temp_file_prefix}{batch_num}.pkl"
            print(f"\nSaving batch {batch_num} ({len(current_batch_results)} points from {processed_stays_count % BATCH_SIZE if processed_stays_count != len(stay_ids_to_process) else len(current_batch_results) % BATCH_SIZE} stays) to {batch_file_path}...")
            try:
                with open(batch_file_path, 'wb') as f: pickle.dump(current_batch_results, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Batch {batch_num} saved successfully.")
                current_batch_results = [] # Reset batch list for next batch
                batch_num += 1
                gc.collect()
            except Exception as e: print(f"\n错误: 保存临时批次文件 {batch_file_path} 时出错: {e}")

# --- End of loop ---
end_time_processing = time.time()
print(f"\nFinished generating features in {end_time_processing - start_time_processing:.2f} seconds.")
print(f"Total prediction points processed: {total_processed_points}")
num_temp_files = batch_num - 1
print(f"Generated {num_temp_files} temporary granular batch files in '{temp_dir}'.")

# Handle the very last batch if any left and not saved
if current_batch_results:
     batch_file_path = f"{temp_file_prefix}{batch_num}.pkl"
     print(f"\nSaving final batch {batch_num} ({len(current_batch_results)} points) to {batch_file_path}...")
     try:
         with open(batch_file_path, 'wb') as f: pickle.dump(current_batch_results, f, protocol=pickle.HIGHEST_PROTOCOL)
         print(f"Final batch saved successfully.")
         num_temp_files += 1
     except Exception as e: print(f"\n错误: 保存最终批次文件时出错: {e}")

print(f"Total temporary files created: {num_temp_files}")

print(f"\n--- Step 4b ({COHORT_KEY.upper()} - Granular Features - Batch/Serial) Completed ---")
print(f"\nNext Step: Merge batches ('step4c_merge_granular_batches.py') OR (Recommended) modify Step 4d")
print(f"         to load temp files from '{temp_dir}' directly, perform Padding (with value=-1),")
print(f"         generate Mask, match outcomes, and save final NPZ batches.")
print(f"         Scaling is NOT needed. Models (Step 5) need Embedding layer.")