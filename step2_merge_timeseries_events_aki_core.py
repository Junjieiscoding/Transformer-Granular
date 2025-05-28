import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import traceback

# ---------------------- 0. 配置 ------------------------------------
mimic_dir = './mimic-iv-3.1'
hosp_dir = os.path.join(mimic_dir, 'hosp')
icu_dir = os.path.join(mimic_dir, 'icu')
output_dir = './output_dynamic'
os.makedirs(output_dir, exist_ok=True)

# --- 输入文件 ---
COHORT_KEY = 'aki' # <<<--- 明确指定当前处理的队列
cohort_file = os.path.join(output_dir, f'step1b_cohort_timestamps_{COHORT_KEY}.parquet')

# 事件文件路径 (确保这是字典！)
event_files_info = {
    'chartevents': {'path': os.path.join(icu_dir, 'chartevents.csv.gz'), 'id_col': 'stay_id', 'value_col': 'valuenum'},
    'labevents': {'path': os.path.join(hosp_dir, 'labevents.csv.gz'), 'id_col': 'hadm_id', 'value_col': 'valuenum'},
    'outputevents': {'path': os.path.join(icu_dir, 'outputevents.csv.gz'), 'id_col': 'stay_id', 'value_col': 'value'}
    # Add more event types and files here if needed
}

# --- 输出文件 ---
# 输出文件名包含队列标识和 "core"
merged_events_file = os.path.join(output_dir, f'step2_merged_events_{COHORT_KEY}_core.parquet')

# --- 核心 ItemID 列表 (针对指定队列) ---
# <<<======================================================================>>>
# <<<===              请将这里替换为你和医生为 AKI 队列              ===>>>
# <<<===         最终确定的核心数值/可编码 ItemID 列表             ===>>>
# <<<======================================================================>>>
CORE_ITEMIDS_DICT = {
    'chartevents': [
        220045, # Heart Rate
        220181, # Arterial Blood Pressure mean (Invasive)
        220210, # Respiratory Rate
        220277, # O2 saturation pulseoxymetry
        223762, # Temperature Celsius
        223901, # GCS - Motor Response
    ],
    'labevents': [
        50820, # pH
        50971, # Potassium (Serum/Plasma)
        50912, # Creatinine (Serum/Plasma)
        51006, # Urea Nitrogen (BUN)
        50882, # Bicarbonate (HCO3)
        50813, # Lactate
        51221, # Hematocrit
        51222, # Hemoglobin (Lab CBC)
        51265, # Platelet Count
        51301, # WBC Count
        # 根据需要添加医生确认的其他数值指标 (CK, 尿液指标, Osmolality等)
        # 50910, 50911, 51082, 51104, 51097, 50964, 50835
    ],
    'outputevents': [
        226559, # Foley output
    ]
}

# 合并所有需要从原始事件表提取的 ItemID
all_core_itemids_to_extract = []
for item_list in CORE_ITEMIDS_DICT.values():
    all_core_itemids_to_extract.extend(item_list)
all_core_itemids_to_extract = list(set(all_core_itemids_to_extract))
N_FEATURES_CORE = len(all_core_itemids_to_extract) # 核心特征数量

print(f"--- Step 2 ({COHORT_KEY.upper()} - Core Features): Merge Time-Series Events ---")
print(f"Input cohort file: {cohort_file}")
print(f"Output file: {merged_events_file}")
print(f"Processing {N_FEATURES_CORE} core itemids for {COHORT_KEY.upper()} cohort.")
if N_FEATURES_CORE <= 15: # Print list only if it's reasonably short
     print("Core ItemIDs:", sorted(all_core_itemids_to_extract))
else:
     print(f"(List too long to display: {N_FEATURES_CORE} items)")
print("-" * 50)

# ---------------------- 1. 加载队列信息 -----------------------------
print(f"Loading {COHORT_KEY.upper()} cohort timestamp data...")
try:
    cohort_df = pd.read_parquet(cohort_file)
    cohort_info = cohort_df[['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime']].copy()
    cohort_info[['stay_id', 'hadm_id', 'subject_id']] = cohort_info[['stay_id', 'hadm_id', 'subject_id']].astype(int)
    valid_stay_ids = cohort_info['stay_id'].unique()
    valid_hadm_ids = cohort_info['hadm_id'].unique()
    # 创建唯一的 HAdmID -> StayID 映射
    hadm_stay_map_cohort = cohort_info.drop_duplicates(subset=['hadm_id'], keep='first').set_index('hadm_id')['stay_id']
    if not hadm_stay_map_cohort.index.is_unique:
        hadm_stay_map_cohort = hadm_stay_map_cohort[~hadm_stay_map_cohort.index.duplicated(keep='first')]
    print(f"Loaded {len(valid_stay_ids)} unique {COHORT_KEY.upper()} stays.")
except FileNotFoundError: exit(f"错误: 找不到队列文件 '{cohort_file}'。")
except Exception as e: exit(f"加载队列文件时出错: {e}")
print("-" * 50)

# ---------------------- 2. 逐个处理并合并事件文件 (只含核心特征) ---
all_events_list = []
total_processed_rows = 0

for event_type, info in event_files_info.items():
    event_file = info['path']
    id_col = info['id_col']
    value_col = info['value_col']
    core_itemids_for_type = CORE_ITEMIDS_DICT.get(event_type, []) # 使用核心列表

    print(f"Processing {event_type} for core features...")
    if not core_itemids_for_type: print(f"  No core features selected for {event_type}, skipping."); continue
    if not os.path.exists(event_file): print(f"  Warning: File {event_file} not found, skipping."); continue

    ids_to_filter = valid_stay_ids if id_col == 'stay_id' else valid_hadm_ids
    if len(ids_to_filter) == 0: print(f"  Cohort ID list empty, skipping."); continue

    try:
        dtype_spec = {id_col: 'Int64', 'itemid': 'Int64'}
        if value_col in ['valuenum', 'value']: dtype_spec[value_col] = 'Float64'
        cols_to_load = list(set([id_col, 'itemid', 'charttime', value_col]))

        chunk_list = []
        reader = pd.read_csv(event_file, compression='gzip', usecols=cols_to_load,
                             parse_dates=['charttime'], chunksize=10_000_000,
                             dtype=dtype_spec, on_bad_lines='skip')

        chunk_count = 0
        for chunk in reader:
            chunk_count += 1
            chunk.dropna(subset=[id_col, 'itemid', 'charttime', value_col], inplace=True)
            chunk = chunk[chunk[id_col].isin(ids_to_filter)]
            if chunk.empty: continue
            chunk = chunk[chunk['itemid'].isin(core_itemids_for_type)] # Filter by CORE list
            if chunk.empty: continue

            if event_type == 'labevents':
                 chunk['stay_id'] = chunk['hadm_id'].map(hadm_stay_map_cohort)
                 chunk.dropna(subset=['stay_id'], inplace=True)
                 chunk['stay_id'] = chunk['stay_id'].astype(int)
                 chunk = chunk[chunk['stay_id'].isin(valid_stay_ids)]
                 if chunk.empty: continue

            if 'stay_id' in chunk.columns and chunk['stay_id'].dtype != 'int64':
                 chunk['stay_id'] = chunk['stay_id'].astype('int64')

            chunk = pd.merge(chunk, cohort_info[['stay_id', 'intime', 'outtime']], on='stay_id', how='inner')
            chunk = chunk[(chunk['charttime'] >= chunk['intime']) & (chunk['charttime'] <= chunk['outtime'])]
            if chunk.empty: continue

            if value_col == 'valuenum': chunk['numeric_value'] = chunk[value_col]
            elif value_col == 'value' and event_type == 'outputevents': chunk['numeric_value'] = pd.to_numeric(chunk[value_col], errors='coerce')
            else: chunk['numeric_value'] = np.nan

            chunk_processed = chunk[['stay_id', 'itemid', 'charttime', 'numeric_value']].copy()
            chunk_processed.dropna(subset=['numeric_value'], inplace=True)

            if not chunk_processed.empty: chunk_list.append(chunk_processed)
            del chunk, chunk_processed; gc.collect()

        if chunk_list:
            event_df = pd.concat(chunk_list, ignore_index=True); all_events_list.append(event_df)
            rows_kept = len(event_df); total_processed_rows += rows_kept
            print(f"    Finished {event_type}. Kept {rows_kept} rows.")
            del event_df, chunk_list; gc.collect()
        else: print(f"    No relevant core events found for {event_type}.")
    except Exception as e: print(f"    Processing {event_type} failed: {e}"); traceback.print_exc()
print("-" * 50)

# ---------------------- 3. 合并所有核心事件 -------------------------
if not all_events_list: exit("错误：未能合并任何核心事件数据。")
print("Merging core events from all sources...")
merged_df = pd.concat(all_events_list, ignore_index=True); del all_events_list; gc.collect()
print("Sorting final merged core events data...")
merged_df.sort_values(by=['stay_id', 'charttime'], inplace=True)
print(f"\nTotal merged core events: {merged_df.shape[0]}")
print("Merged core events preview (first 5 rows):"); print(merged_df.head())
print("\nMerged core events schema:"); merged_df.info(memory_usage='deep')
print("-" * 50)

# ---------------------- 4. 保存合并后的核心事件表 -------------------
print(f"Saving merged core events data for {COHORT_KEY.upper()} to: {merged_events_file}")
try:
    merged_df['stay_id'] = merged_df['stay_id'].astype(int)
    merged_df['itemid'] = merged_df['itemid'].astype(int)
    merged_df['numeric_value'] = merged_df['numeric_value'].astype('Float64')
    merged_df.to_parquet(merged_events_file, index=False)
    print("Merged core events data saved successfully!")
except Exception as e: exit(f"\n错误: 保存合并文件时出错: {e}")

print(f"\n--- Step 2 ({COHORT_KEY.upper()} - Core Features) Completed ---")
print(f"Next Step (Step 3): Run Step 3 script for {COHORT_KEY.upper()} using '{cohort_file}'.")
print(f"Then proceed to Step 4 using '{merged_events_file}' (core features) and the Step 3 output for {COHORT_KEY.upper()}.")