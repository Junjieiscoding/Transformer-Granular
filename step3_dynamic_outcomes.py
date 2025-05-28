import pandas as pd
import numpy as np
import os
from datetime import timedelta
from tqdm import tqdm # Import tqdm for progress bar

# ---------------------- 0. 配置 ------------------------------------
output_dir = './output_dynamic'
os.makedirs(output_dir, exist_ok=True)

# --- 指定当前处理的队列 ---
COHORT_KEY = 'aki' # <<<--- 明确队列

# 输入文件 (Step 1b 的对应队列文件)
cohort_file = os.path.join(output_dir, f'step1b_cohort_timestamps_{COHORT_KEY}.parquet')

# 输出文件 
dynamic_outcome_file = os.path.join(output_dir, f'step3_dynamic_outcomes_{COHORT_KEY}.parquet')

# --- 参数 ---
PREDICTION_WINDOW_HOURS = 24 # 预测未来多少小时内的死亡

print(f"--- Step 3: Generate Dynamic Outcome Labels for Cohort: {COHORT_KEY.upper()} ---")
print(f"Input cohort file: {cohort_file}")
print(f"Output file: {dynamic_outcome_file}")
print(f"Prediction window: {PREDICTION_WINDOW_HOURS} hours")
print("-" * 60)

# ---------------------- 1. 加载队列时间戳数据 --------------------
print(f"Loading {COHORT_KEY.upper()} cohort timestamp data from Step 1b...")
try:
    cohort_df = pd.read_parquet(cohort_file)
    # 确保必要的 timestamp 列是 datetime 对象
    time_cols = ['intime', 'outtime', 'death_event_time']
    for col in time_cols:
        if col in cohort_df.columns:
            cohort_df[col] = pd.to_datetime(cohort_df[col])
        else:
            if col != 'death_event_time':
                 raise ValueError(f"错误: 队列文件中缺少必要的列 '{col}'")

    print(f"Loaded {cohort_df['stay_id'].nunique()} stays for {COHORT_KEY.upper()}.")
except FileNotFoundError:
    print(f"错误: 找不到队列文件 '{cohort_file}'。请确保 Step 1b 已为 '{COHORT_KEY}' 队列成功运行。")
    exit()
except Exception as e:
    print(f"加载队列文件时出错: {e}")
    exit()
print("-" * 60)

# ---------------------- 2. 生成动态目标变量 -------------------------
print(f"Generating dynamic outcome (death within next {PREDICTION_WINDOW_HOURS}h) for each hour...")
dynamic_outcomes = []

for _, stay_row in tqdm(cohort_df.iterrows(), total=cohort_df.shape[0], desc=f"Processing {COHORT_KEY.upper()} stays"):
    stay_id = stay_row['stay_id']
    intime = stay_row['intime']
    outtime = stay_row['outtime']
    death_event_time = stay_row.get('death_event_time', pd.NaT)

    if pd.isna(intime) or pd.isna(outtime) or intime >= outtime:
        continue

    total_los_hours = (outtime - intime).total_seconds() / 3600
    last_possible_prediction_hour = int(np.floor(total_los_hours - PREDICTION_WINDOW_HOURS))

    if last_possible_prediction_hour < 0:
        continue

    for t_hour in range(last_possible_prediction_hour + 1):
        prediction_time = intime + timedelta(hours=t_hour)
        prediction_end_time = prediction_time + timedelta(hours=PREDICTION_WINDOW_HOURS)
        outcome = 0
        if pd.notna(death_event_time):
            if death_event_time > prediction_time and death_event_time <= prediction_end_time:
                outcome = 1
        dynamic_outcomes.append({
            'stay_id': stay_id,
            'prediction_hour': t_hour,
            'prediction_time': prediction_time,
            'outcome_death_next_24h': outcome
        })

dynamic_outcomes_df = pd.DataFrame(dynamic_outcomes)

print(f"\nGenerated {len(dynamic_outcomes_df)} dynamic prediction time points for {COHORT_KEY.upper()}.")
if not dynamic_outcomes_df.empty:
    n_unique_stays_with_preds = dynamic_outcomes_df['stay_id'].nunique()
    positive_rate = dynamic_outcomes_df['outcome_death_next_24h'].mean()
    print(f"  Number of unique stays with predictions: {n_unique_stays_with_preds}")
    print(f"  Positive outcome (death) rate: {positive_rate:.4f}")
    print("\nDynamic outcomes DataFrame preview (first 5 rows):")
    print(dynamic_outcomes_df.head())
    print("\nDynamic outcomes DataFrame preview (last 5 rows):")
    print(dynamic_outcomes_df.tail())
else:
    print("警告：没有生成任何动态结果标签。")
print("-" * 60)

# ---------------------- 3. 保存结果 ---------------------------------
if not dynamic_outcomes_df.empty:
    print(f"Saving dynamic outcomes for {COHORT_KEY.upper()} to {dynamic_outcome_file}...")
    try:
        dynamic_outcomes_df['stay_id'] = dynamic_outcomes_df['stay_id'].astype(int)
        dynamic_outcomes_df['prediction_hour'] = dynamic_outcomes_df['prediction_hour'].astype(int)
        dynamic_outcomes_df['outcome_death_next_24h'] = dynamic_outcomes_df['outcome_death_next_24h'].astype(int)
        dynamic_outcomes_df.to_parquet(dynamic_outcome_file, index=False)
        print("Dynamic outcomes saved successfully!")
    except Exception as e:
        print(f"\n错误: 保存 Parquet 文件时出错: {e}")
        exit()
else:
    print("No dynamic outcomes generated, skipping save.")

# --- FIX: Correctly reference the expected input file for the next step ---
expected_step2_output = os.path.join(output_dir, f'step2_merged_events_{COHORT_KEY}_core.parquet')

print(f"\n--- Step 3 ({COHORT_KEY.upper()}) Completed ---")
print(f"Next Step (Step 4a): Use '{expected_step2_output}' (from Step 2 for {COHORT_KEY.upper()})") # Use the correct variable
print(f"  and '{dynamic_outcome_file}' (this file) to generate dynamic hourly features.")