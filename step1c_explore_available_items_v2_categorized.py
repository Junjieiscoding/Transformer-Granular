# step1c_explore_available_items_v2_categorized.py

import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict # Use defaultdict for grouping
import gc
from tqdm import tqdm

# ---------------------- 0. 配置 ------------------------------------
mimic_dir = './mimic-iv-3.1'
hosp_dir = os.path.join(mimic_dir, 'hosp')
icu_dir = os.path.join(mimic_dir, 'icu')
output_dir = './output_dynamic'
os.makedirs(output_dir, exist_ok=True)

# 输入文件: Step 1b 分层后的队列文件列表
cohort_files_to_explore = {
    'aki': os.path.join(output_dir, 'step1b_cohort_timestamps_aki.parquet'),
    'hf': os.path.join(output_dir, 'step1b_cohort_timestamps_hf.parquet')
}

# 事件文件路径
event_files_info = {
    'chartevents': {'path': os.path.join(icu_dir, 'chartevents.csv.gz'), 'id_col': 'stay_id', 'dict_type': 'd_items'},
    'labevents': {'path': os.path.join(hosp_dir, 'labevents.csv.gz'), 'id_col': 'hadm_id', 'dict_type': 'd_labitems'},
    'outputevents': {'path': os.path.join(icu_dir, 'outputevents.csv.gz'), 'id_col': 'stay_id', 'dict_type': 'd_items'} # Output items are often in d_items
}

# 字典文件 (这次需要 category)
d_items_file = os.path.join(icu_dir, 'd_items.csv.gz')
d_labitems_file = os.path.join(hosp_dir, 'd_labitems.csv.gz')

# 输出文件: 每个队列一个统计报告 (CSV 格式更方便后续处理)
output_report_prefix = os.path.join(output_dir, 'step1c_item_exploration_report_categorized_')

# --- 参数 ---
MIN_FREQUENCY_TO_REPORT = 10 # 只报告出现次数超过 N 次的 ItemID，避免过多噪音

print(f"--- Step 1c (V2 - Categorized): Explore ALL Available Items ---")
print(f"Output report prefix: {output_report_prefix}")
print(f"Minimum frequency to report: {MIN_FREQUENCY_TO_REPORT}")
print("-" * 60)

# ---------------------- 1. 加载字典文件 (包含 Category) -------------
item_details = {} # Store {itemid: {'label': ..., 'category': ..., 'source': ...}}
try:
    print("Loading d_items...")
    # Keep linksto to know if it's chartevents or other ICU tables
    d_items = pd.read_csv(d_items_file, compression='gzip', usecols=['itemid', 'label', 'category', 'linksto'])
    for _, row in d_items.iterrows():
        item_details[row['itemid']] = {'label': row['label'], 'category': row['category'], 'source': row['linksto']}
    del d_items
    print("Loading d_labitems...")
    d_labitems = pd.read_csv(d_labitems_file, compression='gzip', usecols=['itemid', 'label', 'category', 'fluid']) # Use fluid instead of linksto
    for _, row in d_labitems.iterrows():
        # Avoid overwriting if itemid exists in both (rare), prioritize non-lab? Or check source?
        if row['itemid'] not in item_details:
             item_details[row['itemid']] = {'label': row['label'], 'category': row['category'], 'source': 'labevents'} # Mark source as lab
    del d_labitems
    gc.collect()
    print(f"Loaded details for {len(item_details)} unique items.")
except Exception as e:
    print(f"Warning: Could not load item dictionary files completely: {e}. Reports may lack labels/categories.")
print("-" * 60)


# ---------------------- 2. 针对每个队列进行探索 --------------------
for cohort_key, cohort_file_path in cohort_files_to_explore.items():
    print(f"\n--- Exploring Cohort: {cohort_key.upper()} ---")
    print(f"Cohort file: {cohort_file_path}")
    if not os.path.exists(cohort_file_path): print("  队列文件不存在，跳过。"); continue

    try:
        cohort_df = pd.read_parquet(cohort_file_path)
        cohort_stay_ids = cohort_df['stay_id'].unique()
        cohort_hadm_ids = cohort_df['hadm_id'].unique()
        # Create hadm_id -> stay_id map specific to this cohort
        hadm_stay_map_cohort = cohort_df.drop_duplicates(subset=['hadm_id']).set_index('hadm_id')['stay_id']
        print(f"  Loaded {len(cohort_stay_ids)} stays for {cohort_key.upper()}.")
    except Exception as e: print(f"  加载队列文件时出错: {e}"); continue

    cohort_item_counts = Counter() # Still use Counter for overall frequency

    # --- 遍历事件文件 ---
    for event_type, info in event_files_info.items():
        event_file = info['path']
        id_col = info['id_col']
        print(f"  Processing {event_type}...")
        if not os.path.exists(event_file): print(f"    文件不存在，跳过。"); continue
        ids_to_filter = cohort_stay_ids if id_col == 'stay_id' else cohort_hadm_ids
        if len(ids_to_filter) == 0: print(f"    队列 ID 列表为空，跳过。"); continue

        try:
            dtype_spec = {id_col: 'Int64', 'itemid': 'Int64'}
            # Only load necessary columns for counting
            cols_to_load = [id_col, 'itemid']
            reader = pd.read_csv(event_file, compression='gzip', usecols=cols_to_load,
                                 chunksize=10_000_000, # Larger chunksize for counting
                                 dtype=dtype_spec, on_bad_lines='skip')

            for chunk in reader: # No tqdm here to reduce clutter per event type
                chunk.dropna(subset=[id_col, 'itemid'], inplace=True)
                chunk = chunk[chunk[id_col].isin(ids_to_filter)]
                if chunk.empty: continue

                if event_type == 'labevents':
                    chunk['stay_id'] = chunk['hadm_id'].map(hadm_stay_map_cohort)
                    chunk.dropna(subset=['stay_id'], inplace=True)
                    chunk = chunk[chunk['stay_id'].isin(cohort_stay_ids)] # Ensure stay_id is in cohort
                    if chunk.empty: continue

                # Count frequencies
                cohort_item_counts.update(chunk['itemid'].tolist())
                del chunk; gc.collect()
            print(f"    Finished processing {event_type}.")
        except Exception as e: print(f"    处理 {event_type} 时发生错误: {e}"); traceback.print_exc()

    # --- 分析和整理结果 ---
    print(f"\n  Analysis for Cohort: {cohort_key.upper()}")
    total_item_occurrences = sum(cohort_item_counts.values())
    num_unique_items = len(cohort_item_counts)
    print(f"  Total item occurrences found: {total_item_occurrences}")
    print(f"  Number of unique itemids found: {num_unique_items}")

    if num_unique_items == 0: print("  未找到任何相关的 ItemID。"); continue

    # --- 创建包含类别和标签的 DataFrame ---
    report_data = []
    for itemid, count in cohort_item_counts.items():
        if count >= MIN_FREQUENCY_TO_REPORT: # Apply frequency threshold
             details = item_details.get(itemid, {'label': 'N/A', 'category': 'Unknown', 'source': 'Unknown'})
             report_data.append({
                 'ItemID': itemid,
                 'Count': count,
                 'Label': details['label'],
                 'Category': details.get('category', 'Unknown'), # Use .get for safety
                 'SourceTable': details['source']
             })

    if not report_data:
        print(f"  未找到出现次数超过 {MIN_FREQUENCY_TO_REPORT} 的 ItemID。")
        continue

    report_df = pd.DataFrame(report_data)
    # 按类别，再按频率排序
    report_df.sort_values(by=['Category', 'Count'], ascending=[True, False], inplace=True)

    # --- 生成报告文件 (CSV) ---
    report_file_path = f"{output_report_prefix}{cohort_key}.csv" # Save as CSV
    print(f"  Generating exploration report (CSV) to: {report_file_path}")
    try:
        header = [
            f"ItemID Exploration Report for Cohort: {cohort_key.upper()}",
            f"Based on cohort file: {os.path.basename(cohort_file_path)}",
            f"Total unique stays in cohort: {len(cohort_stay_ids)}",
            f"Total relevant event occurrences (all items): {total_item_occurrences}",
            f"Number of unique ItemIDs found (all items): {num_unique_items}",
            f"Report includes items with frequency >= {MIN_FREQUENCY_TO_REPORT}."
        ]
        with open(report_file_path, 'w') as f:
             f.write("# " + "\n# ".join(header) + "\n") # Write header as comments
        report_df.to_csv(report_file_path, index=False, mode='a') # Append DataFrame after header
        print(f"  Report saved successfully.")
    except Exception as e:
        print(f"  保存报告文件时出错: {e}")

    del cohort_df, cohort_stay_ids, cohort_hadm_ids, hadm_stay_map_cohort
    del cohort_item_counts, report_data, report_df
    gc.collect()

print(f"\n--- Step 1c V2 Exploration Completed ---")
print("Please review the generated categorized report files (*.csv) using Excel or Pandas")
print("to select core features for Step 2 based on Category, Frequency, and Label.")