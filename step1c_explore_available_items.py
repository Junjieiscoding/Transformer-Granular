import pandas as pd
import numpy as np
import os
from collections import Counter
import gc

# ---------------------- 0. 配置 ------------------------------------
mimic_dir = './   mimic-iv-3.1'
hosp_dir = os.path.join(mimic_dir, 'hosp')
icu_dir = os.path.join(mimic_dir, 'icu')
output_dir = './output_dynamic'

# 输入文件: Step 1b 分层后的队列文件列表
cohort_files_to_explore = {
    'aki': os.path.join(output_dir, 'step1b_cohort_timestamps_aki.parquet'),
    'hf': os.path.join(output_dir, 'step1b_cohort_timestamps_hf.parquet')
}

# 事件文件路径
event_files_info = {
    'chartevents': {'path': os.path.join(icu_dir, 'chartevents.csv.gz'), 'id_col': 'stay_id', 'value_col': 'valuenum'},
    'labevents': {'path': os.path.join(hosp_dir, 'labevents.csv.gz'), 'id_col': 'hadm_id', 'value_col': 'valuenum'},
    'outputevents': {'path': os.path.join(icu_dir, 'outputevents.csv.gz'), 'id_col': 'stay_id', 'value_col': 'value'}
    # Add more if needed
}

# 字典文件 (用于获取标签)
d_items_file = os.path.join(icu_dir, 'd_items.csv.gz')
d_labitems_file = os.path.join(hosp_dir, 'd_labitems.csv.gz')

# 输出文件: 每个队列一个统计报告
output_report_prefix = os.path.join(output_dir, 'step1c_item_exploration_report_')

# --- 参数 ---
TOP_N_ITEMS_TO_SHOW = 100 # 显示频率最高的前 N 个 ItemID
SAMPLE_SIZE_FOR_VALUES = 1000 # 对高频 ItemID 抽样多少条记录看数值范围

print(f"--- Step 1c: Explore Available Items for Selected Cohorts ---")
print(f"Output report prefix: {output_report_prefix}")
print("-" * 60)

# ---------------------- 1. 加载字典文件 (Optional) -----------------
item_labels = {}
try:
    print("Loading d_items...")
    d_items = pd.read_csv(d_items_file, compression='gzip', usecols=['itemid', 'label'])
    item_labels.update(d_items.set_index('itemid')['label'].to_dict())
    del d_items
    print("Loading d_labitems...")
    d_labitems = pd.read_csv(d_labitems_file, compression='gzip', usecols=['itemid', 'label'])
    item_labels.update(d_labitems.set_index('itemid')['label'].to_dict())
    del d_labitems
    gc.collect()
    print(f"Loaded labels for {len(item_labels)} items.")
except Exception as e:
    print(f"Warning: Could not load item dictionary files: {e}. Reports will lack labels.")
print("-" * 60)


# ---------------------- 2. 针对每个队列进行探索 --------------------
for cohort_key, cohort_file_path in cohort_files_to_explore.items():
    print(f"\n--- Exploring Cohort: {cohort_key.upper()} ---")
    print(f"Cohort file: {cohort_file_path}")
    if not os.path.exists(cohort_file_path):
        print("  队列文件不存在，跳过。")
        continue

    try:
        cohort_df = pd.read_parquet(cohort_file_path)
        cohort_stay_ids = cohort_df['stay_id'].unique()
        cohort_hadm_ids = cohort_df['hadm_id'].unique()
        cohort_intime_map = cohort_df.set_index('stay_id')['intime']
        cohort_outtime_map = cohort_df.set_index('stay_id')['outtime']
        # Create hadm_id -> stay_id map specific to this cohort
        hadm_stay_map_cohort = cohort_df.drop_duplicates(subset=['hadm_id']).set_index('hadm_id')['stay_id']

        print(f"  Loaded {len(cohort_stay_ids)} stays for {cohort_key.upper()}.")
    except Exception as e:
        print(f"  加载队列文件时出错: {e}")
        continue

    cohort_item_counts = Counter() # Use Counter for efficient frequency counting
    cohort_item_samples = {} # Store sample values for top items {itemid: [values]}

    # --- 遍历事件文件 ---
    for event_type, info in event_files_info.items():
        event_file = info['path']
        id_col = info['id_col']
        value_col = info['value_col']
        print(f"  Processing {event_type}...")
        if not os.path.exists(event_file):
            print(f"    文件不存在，跳过。")
            continue

        ids_to_filter = cohort_stay_ids if id_col == 'stay_id' else cohort_hadm_ids
        if len(ids_to_filter) == 0:
             print(f"    队列 ID 列表为空，跳过。")
             continue

        try:
            # Define dtypes for efficiency
            dtype_spec = {id_col: 'Int64', 'itemid': 'Int64'}
            if value_col in ['valuenum', 'value']: # Assume numeric potential
                 dtype_spec[value_col] = 'Float64'

            reader = pd.read_csv(event_file, compression='gzip',
                                 usecols=[id_col, 'itemid', 'charttime', value_col], # Load minimal cols
                                 parse_dates=['charttime'], chunksize=5_000_000, # Smaller chunksize for exploration
                                 dtype=dtype_spec, on_bad_lines='skip') # Skip bad lines if any

            chunk_count = 0
            for chunk in reader:
                chunk_count += 1
                # print(f"    Processing {event_type} chunk {chunk_count}...")
                chunk.dropna(subset=[id_col, 'itemid', 'charttime', value_col], inplace=True)

                # Filter by relevant cohort IDs
                chunk = chunk[chunk[id_col].isin(ids_to_filter)]
                if chunk.empty: continue

                # Map hadm_id to stay_id for labevents
                if event_type == 'labevents':
                    chunk['stay_id'] = chunk['hadm_id'].map(hadm_stay_map_cohort)
                    chunk.dropna(subset=['stay_id'], inplace=True)
                    chunk['stay_id'] = chunk['stay_id'].astype(int)
                    # Ensure we only keep events for stays actually in this cohort
                    chunk = chunk[chunk['stay_id'].isin(cohort_stay_ids)]
                    if chunk.empty: continue

                # Filter events within ICU stay time (approximate using maps)
                # This merge is potentially slow, maybe do it *after* counting?
                # Let's count first, then optionally sample & filter time for value ranges
                # chunk = pd.merge(chunk, cohort_intime_map.rename('intime'), left_on='stay_id', right_index=True, how='inner')
                # chunk = pd.merge(chunk, cohort_outtime_map.rename('outtime'), left_on='stay_id', right_index=True, how='inner')
                # chunk = chunk[(chunk['charttime'] >= chunk['intime']) & (chunk['charttime'] <= chunk['outtime'])]
                # if chunk.empty: continue

                # --- Count ItemID frequencies ---
                cohort_item_counts.update(chunk['itemid'].tolist())

                # --- Collect sample values for high-frequency items later ---
                # For now, just count

                del chunk # Free memory
                gc.collect()

            print(f"    Finished processing {event_type} chunks.")

        except Exception as e:
            print(f"    处理 {event_type} 时发生错误: {e}")
            traceback.print_exc()

    # --- 分析统计结果 ---
    print(f"\n  Analysis for Cohort: {cohort_key.upper()}")
    total_item_occurrences = sum(cohort_item_counts.values())
    num_unique_items = len(cohort_item_counts)
    print(f"  Total item occurrences found: {total_item_occurrences}")
    print(f"  Number of unique itemids found: {num_unique_items}")

    if num_unique_items == 0:
        print("  未找到任何相关的 ItemID。")
        continue

    # 获取频率最高的 Top N ItemIDs
    top_items = cohort_item_counts.most_common(TOP_N_ITEMS_TO_SHOW)

    # --- (可选) 抽样获取 Top N ItemID 的数值范围 ---
    # This requires another pass through the data or smarter sampling during first pass
    # For simplicity, we will skip the value range sampling in this version
    # but the report will show the most frequent items.

    # --- 生成报告文件 ---
    report_file_path = f"{output_report_prefix}{cohort_key}.txt"
    print(f"  Generating exploration report to: {report_file_path}")
    try:
        with open(report_file_path, 'w') as f:
            f.write(f"ItemID Exploration Report for Cohort: {cohort_key.upper()}\n")
            f.write(f"Based on cohort file: {cohort_file_path}\n")
            f.write(f"Total unique stays in cohort: {len(cohort_stay_ids)}\n")
            f.write(f"Total relevant event occurrences: {total_item_occurrences}\n")
            f.write(f"Number of unique ItemIDs found: {num_unique_items}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Top {TOP_N_ITEMS_TO_SHOW} Most Frequent ItemIDs:\n")
            f.write("-" * 50 + "\n")
            f.write("Rank | ItemID   | Count      | Label (if available)\n")
            f.write("-" * 50 + "\n")
            for rank, (itemid, count) in enumerate(top_items, 1):
                label = item_labels.get(itemid, "N/A")
                f.write(f"{rank:<4} | {itemid:<8} | {count:<10} | {label}\n")
        print(f"  Report saved successfully.")
    except Exception as e:
        print(f"  保存报告文件时出错: {e}")

    del cohort_df, cohort_stay_ids, cohort_hadm_ids, cohort_intime_map, cohort_outtime_map
    del hadm_stay_map_cohort, cohort_item_counts, top_items
    gc.collect()


print(f"\n--- Step 1c Exploration Completed ---")
print("Please review the generated report files (*.txt) to help select core features for Step 2.")