import pandas as pd
import numpy as np
import os
from datetime import timedelta
import operator # For sorting dictionary

# ---------------------- 0. 配置 ------------------------------------
mimic_dir = './mimic-iv-3.1' # <<<--- 确保路径正确
hosp_dir = os.path.join(mimic_dir, 'hosp')
icu_dir = os.path.join(mimic_dir, 'icu')
output_dir = './output_dynamic'
os.makedirs(output_dir, exist_ok=True)

# 输入文件路径
admissions_file = os.path.join(hosp_dir, 'admissions.csv.gz')
patients_file = os.path.join(hosp_dir, 'patients.csv.gz')
icustays_file = os.path.join(icu_dir, 'icustays.csv.gz')
diagnoses_file = os.path.join(hosp_dir, 'diagnoses_icd.csv.gz')

# --- 队列定义参数 ---
MIN_AGE = 18
MIN_STAY_HOURS_FOR_COHORT = 24

# --- 疾病定义 (包含所有你想考虑的候选疾病及其代码) ---
# 我们将从这个完整列表中自动选择 Top 2
candidate_disease_definitions = {
    'hf': { # Heart Failure
        'name': 'Heart Failure',
        'icd9_prefix': ['428'], 'icd10_prefix': ['I50']},
    'sepsis': { # Sepsis
        'name': 'Sepsis',
        'icd9_code': ['99591', '99592', '78552'], 'icd10_prefix': ['A40', 'A41'], 'icd10_code_prefix': ['R652']},
    'arf': { # Acute Respiratory Failure
        'name': 'Acute Respiratory Failure',
        'icd9_code': ['51881', '51882', '51884'], 'icd10_prefix': ['J960', 'J962']},
    'aki': { # Acute Kidney Injury
        'name': 'Acute Kidney Injury',
        'icd9_prefix': ['584'], 'icd10_prefix': ['N17']},
    'pneumonia': {
        'name': 'Pneumonia',
        'icd9_prefix': ['480','481','482','483','484','485','486'], 'icd10_prefix': ['J12','J13','J14','J15','J16','J17','J18']},
    'stroke': {
        'name': 'Stroke/CVA',
        'icd9_prefix': ['430','431','432','433','434','435','436','437'], 'icd10_prefix': ['I6']},
    'ami': {
        'name': 'Acute Myocardial Infarction',
        'icd9_code': ['410'], 'icd10_prefix': ['I21', 'I22']},
    'copd': {
        'name': 'COPD',
        'icd9_code': ['491', '492', '496'], 'icd10_prefix': ['J44']},
     'gi_bleed': {
        'name': 'GI Bleed',
        'icd9_prefix': ['578'], 'icd10_code': ['K920', 'K921', 'K922']},
     'cirrhosis': {
        'name': 'Cirrhosis/Liver Failure',
        'icd9_prefix': ['571'], 'icd10_prefix': ['K70', 'K71', 'K72', 'K73', 'K74']},
    # 你可以继续添加其他你感兴趣的疾病定义...
}

# --- 自动选择 Top N 疾病 ---
N_TOP_DISEASES = 2 # <<<--- 设置你想选择的数量

print(f"--- Step 1 (Auto Stratify Top {N_TOP_DISEASES}): Define Cohort, Extract Timestamps, Stratify ---")
# ... (其他打印信息) ...
print("-" * 70)

# --- Function to Load and Apply Base Filters (保持不变) ---
def load_and_filter_base_cohort(admissions_file, patients_file, icustays_file, min_age, min_stay_hours):
    # ... (代码与上一个版本完全相同) ...
    print("Loading core tables...")
    try:
        adm = pd.read_csv(admissions_file, compression='gzip', usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag'], parse_dates=['admittime', 'dischtime', 'deathtime'])
        adm.dropna(subset=['hadm_id'], inplace=True); adm['hadm_id'] = adm['hadm_id'].astype(int)
        pat = pd.read_csv(patients_file, compression='gzip', usecols=['subject_id', 'gender', 'anchor_age', 'dod'], parse_dates=['dod'])
        pat.dropna(subset=['subject_id'], inplace=True); pat['subject_id'] = pat['subject_id'].astype(int)
        icu = pd.read_csv(icustays_file, compression='gzip', usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'], parse_dates=['intime', 'outtime'])
        icu.dropna(subset=['stay_id', 'hadm_id', 'intime', 'outtime', 'los'], inplace=True)
        icu = icu[icu['outtime'] > icu['intime']].copy(); icu['los_hours'] = icu['los'] * 24.0
        icu[['stay_id', 'hadm_id', 'subject_id']] = icu[['stay_id', 'hadm_id', 'subject_id']].astype(int)
    except Exception as e: print(f"Error loading core tables: {e}"); exit()
    print("Merging core tables...")
    icu_pat = pd.merge(icu, pat, on='subject_id', how='inner')
    cohort_base = pd.merge(icu_pat, adm, on=['subject_id', 'hadm_id'], how='inner')
    print(f"Initial merged cohort size: {cohort_base.shape[0]}")
    print("Applying base filters (Age, LOS, First Stay)...")
    n_initial = cohort_base.shape[0]
    cohort_filtered = cohort_base[cohort_base['anchor_age'] >= min_age].copy()
    n_after_age = cohort_filtered.shape[0]
    cohort_filtered = cohort_filtered[cohort_filtered['los_hours'] >= min_stay_hours].copy()
    n_after_los = cohort_filtered.shape[0]
    cohort_filtered = cohort_filtered.sort_values(by=['subject_id', 'intime']).drop_duplicates(subset=['subject_id'], keep='first').copy()
    n_after_first = cohort_filtered.shape[0]
    cohort_final = cohort_filtered.dropna(subset=['intime', 'outtime', 'admittime']).copy() # Require admittime
    n_final = cohort_final.shape[0]
    print(f"Filtering Steps: Initial({n_initial}) -> Age({n_after_age}) -> LOS({n_after_los}) -> FirstStay({n_after_first}) -> Timestamps({n_final})")
    if n_final == 0: exit("错误：基础筛选后没有符合条件的 ICU stay 记录！")
    cohort_final['death_event_time'] = cohort_final['deathtime'].combine_first(cohort_final['dod'])
    columns_final = ['subject_id', 'hadm_id', 'stay_id', 'anchor_age', 'gender', 'intime', 'outtime', 'los_hours', 'admittime', 'dischtime', 'death_event_time', 'hospital_expire_flag']
    cohort_final = cohort_final[columns_final].copy()
    cohort_final['gender'] = cohort_final['gender'].map({'M': 1, 'F': 0}).astype('Int64')
    print(f"Base cohort prepared with {n_final} unique stays.")
    return cohort_final

# --- 执行基础队列创建 ---
base_cohort_df = load_and_filter_base_cohort(admissions_file, patients_file, icustays_file, MIN_AGE, MIN_STAY_HOURS_FOR_COHORT)
print("-" * 70)

# --- 加载诊断信息 ---
print("Loading diagnoses data for stratification...")
try:
    diagnoses_df = pd.read_csv(diagnoses_file, compression='gzip', usecols=['hadm_id', 'icd_code', 'icd_version'],
                               dtype={'hadm_id': 'Int64', 'icd_version': 'Int64', 'icd_code': str})
    diagnoses_df.dropna(inplace=True)
    diagnoses_df['icd_code_norm'] = diagnoses_df['icd_code'].str.replace('.', '', regex=False).str.upper()
except Exception as e: print(f"Error loading diagnoses data: {e}"); exit()

cohort_hadm_ids = base_cohort_df['hadm_id'].unique()
diagnoses_cohort = diagnoses_df[diagnoses_df['hadm_id'].isin(cohort_hadm_ids)].copy()
print(f"Filtered diagnoses relevant to the cohort.")
print("-" * 70)

# --- 统计所有候选疾病的频率 ---
print("Calculating frequencies for all candidate diseases...")
disease_hadm_counts = {}

for key, definition in candidate_disease_definitions.items():
    disease_name = definition['name']
    is_disease = pd.Series(False, index=diagnoses_cohort.index)
    # (复制之前的 ICD 代码匹配逻辑)
    icd9_codes = definition.get('icd9_code', [])
    icd9_prefixes = definition.get('icd9_prefix', [])
    if icd9_codes: is_disease |= ((diagnoses_cohort['icd_version'] == 9) & diagnoses_cohort['icd_code_norm'].isin(icd9_codes))
    if icd9_prefixes:
        for prefix in icd9_prefixes: is_disease |= ((diagnoses_cohort['icd_version'] == 9) & diagnoses_cohort['icd_code_norm'].str.startswith(prefix))
    icd10_codes_prefix = definition.get('icd10_code_prefix', [])
    icd10_codes_exact = definition.get('icd10_code', [])
    icd10_prefixes = definition.get('icd10_prefix', [])
    if icd10_codes_prefix:
        for code in icd10_codes_prefix: is_disease |= ((diagnoses_cohort['icd_version'] == 10) & diagnoses_cohort['icd_code_norm'].str.startswith(code))
    if icd10_codes_exact: is_disease |= ((diagnoses_cohort['icd_version'] == 10) & diagnoses_cohort['icd_code_norm'].isin(icd10_codes_exact))
    if icd10_prefixes:
         for prefix in icd10_prefixes: is_disease |= ((diagnoses_cohort['icd_version'] == 10) & diagnoses_cohort['icd_code_norm'].str.startswith(prefix))

    hadm_ids_with_disease = diagnoses_cohort.loc[is_disease, 'hadm_id'].unique()
    disease_hadm_counts[key] = len(hadm_ids_with_disease) # Store count by key
    print(f"  - {disease_name} ({key}): {len(hadm_ids_with_disease)} unique HADM_IDs")

# --- 选择 Top N 疾病 ---
# Sort the dictionary by value (count) in descending order
sorted_disease_counts = sorted(disease_hadm_counts.items(), key=operator.itemgetter(1), reverse=True)

# Select the top N keys
top_n_disease_keys = [item[0] for item in sorted_disease_counts[:N_TOP_DISEASES]]

print(f"\nSelected Top {N_TOP_DISEASES} diseases based on HADM_ID count:")
selected_disease_definitions = {}
for key in top_n_disease_keys:
    selected_disease_definitions[key] = candidate_disease_definitions[key]
    # Generate output file path dynamically based on key
    selected_disease_definitions[key]['output_file'] = os.path.join(output_dir, f'step1b_cohort_timestamps_{key}.parquet')
    print(f"  - {selected_disease_definitions[key]['name']} ({key}) with {disease_hadm_counts[key]} HADM_IDs")
print("-" * 70)


# ---------------------- 执行分层并保存 Top N 队列 -------------
print(f"Stratifying and saving the Top {N_TOP_DISEASES} disease cohorts...")

# We need the hadm_ids for each selected disease again (could store them above)
for key, definition in selected_disease_definitions.items():
    disease_name = definition['name']
    output_file = definition['output_file']
    print(f"  Processing: {disease_name} ({key})")

    # Recalculate the hadm_ids for this specific disease
    is_disease = pd.Series(False, index=diagnoses_cohort.index)
    # (复制之前的 ICD 代码匹配逻辑)
    icd9_codes = definition.get('icd9_code', [])
    icd9_prefixes = definition.get('icd9_prefix', [])
    if icd9_codes: is_disease |= ((diagnoses_cohort['icd_version'] == 9) & diagnoses_cohort['icd_code_norm'].isin(icd9_codes))
    if icd9_prefixes:
        for prefix in icd9_prefixes: is_disease |= ((diagnoses_cohort['icd_version'] == 9) & diagnoses_cohort['icd_code_norm'].str.startswith(prefix))
    icd10_codes_prefix = definition.get('icd10_code_prefix', [])
    icd10_codes_exact = definition.get('icd10_code', [])
    icd10_prefixes = definition.get('icd10_prefix', [])
    if icd10_codes_prefix:
        for code in icd10_codes_prefix: is_disease |= ((diagnoses_cohort['icd_version'] == 10) & diagnoses_cohort['icd_code_norm'].str.startswith(code))
    if icd10_codes_exact: is_disease |= ((diagnoses_cohort['icd_version'] == 10) & diagnoses_cohort['icd_code_norm'].isin(icd10_codes_exact))
    if icd10_prefixes:
         for prefix in icd10_prefixes: is_disease |= ((diagnoses_cohort['icd_version'] == 10) & diagnoses_cohort['icd_code_norm'].str.startswith(prefix))

    hadm_ids_with_disease = diagnoses_cohort.loc[is_disease, 'hadm_id'].unique()

    # Filter the base cohort DataFrame
    disease_cohort_df = base_cohort_df[base_cohort_df['hadm_id'].isin(hadm_ids_with_disease)].copy()
    n_stays_disease = disease_cohort_df['stay_id'].nunique()
    print(f"    Final cohort for {disease_name}: {n_stays_disease} unique Stay_IDs.")

    # Save the stratified cohort
    if not disease_cohort_df.empty:
        print(f"    Saving cohort data to: {output_file}")
        try:
            disease_cohort_df.to_parquet(output_file, index=False)
            print(f"    Saved successfully.")
        except Exception as e:
            print(f"    错误: 保存 {disease_name} 队列文件时出错: {e}")
    else:
        print(f"    警告: 未找到符合 {disease_name} 标准的 Stays，未保存文件。")
    print("-" * 30)

print(f"\n--- Step 1 (Auto Stratify Top {N_TOP_DISEASES}) Completed ---")
print(f"Next Step (Step 2): For each generated cohort file (e.g., *_{key}.parquet),")
print(f"  run Step 2 script using the appropriate *minimal* feature list for that disease.")