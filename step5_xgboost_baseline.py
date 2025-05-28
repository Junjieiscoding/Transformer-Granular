# step5_xgboost_baseline.py
import os
import glob
import gc
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, confusion_matrix, log_loss
)
import xgboost as xgb
from tqdm import tqdm
import time
import warnings
import traceback
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免显示图形

# 抑制警告
warnings.filterwarnings('ignore')

# --- 全局配置 ---
OUTPUT_DIR = './output_dynamic'
COHORT_KEY = 'aki'
MODEL_NAME_XGB = 'XGBoost_Granular' # 用于日志和文件名

# --- 文件与目录路径 (与PyTorch脚本的结构对齐) ---
# 模型保存在与PyTorch模型相同的目录下
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'step5_models_pytorch_{COHORT_KEY}')
# 结果、日志、图表保存在与PyTorch结果相同的目录下
RESULTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'step5_results_pytorch_{COHORT_KEY}')

XGB_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, f'best_{MODEL_NAME_XGB}.model')
XGB_ES_CHECKPOINT_PATH = os.path.join(MODEL_OUTPUT_DIR, f'es_checkpoint_{MODEL_NAME_XGB}.model') # 早停时保存的最佳模型
XGB_TEST_RESULTS_PATH = os.path.join(RESULTS_OUTPUT_DIR, f'{MODEL_NAME_XGB.lower()}_test_summary.json')
XGB_TRAINING_LOG_PATH = os.path.join(RESULTS_OUTPUT_DIR, f'{MODEL_NAME_XGB.lower()}_training_log.tsv')
# 图表直接保存在RESULTS_OUTPUT_DIR下，以匹配PyTorch脚本行为
# XGB_PLOTS_DIR = os.path.join(RESULTS_OUTPUT_DIR, f'{MODEL_NAME_XGB.lower()}_plots') # 如果需要子目录

# 创建必要的目录
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
# if XGB_PLOTS_DIR: os.makedirs(XGB_PLOTS_DIR, exist_ok=True)


# --- 数据参数 ---
MAX_LEN = 240
PADDING_VALUE_INPUT = -1.0 # XGBoost DMatrix的 missing 参数会处理这个
TEST_SIZE = 0.15
VAL_SIZE = 0.15 # 相对于 (1 - TEST_SIZE) 的比例
SEED = 42
DATA_LOADER_BATCH_SIZE = 100  # 数据加载器分批加载原始数据的批大小

# --- XGBoost训练参数 ---
XGB_EPOCHS = 50 # 最大提升轮数
XGB_ES_PATIENCE = 10 # 早停轮数 (XGBoost中是rounds)


# --- 内存优化数据加载器 (与您提供的版本一致) ---
class GranularPklDataset:
    """内存优化数据集（按需加载PKL文件中的数据）"""
    def __init__(self, pkl_dir, outcomes_file, cohort_key_for_glob, max_len_feat, padding_value=-1.0):
        self.pkl_files = sorted(glob.glob(os.path.join(pkl_dir, f'*{cohort_key_for_glob}*.pkl')))
        if not self.pkl_files:
            raise ValueError(f"在目录: {pkl_dir} 中未找到任何匹配 '*{cohort_key_for_glob}*.pkl' 的PKL文件")
        self.max_len_feat = max_len_feat
        self.padding_value = padding_value
        try:
            with open(self.pkl_files[0], 'rb') as f:
                first_data_chunk = pickle.load(f)
                if not first_data_chunk: raise ValueError("首个PKL文件为空。")
                _, _, first_sample_feat_array = first_data_chunk[0]
                self.n_features_at_timestep = first_sample_feat_array.shape[1]
                print(f"自动检测到每个时间步的特征数量 (M): {self.n_features_at_timestep}")
        except Exception as e:
            print(f"错误：无法从首个PKL文件自动检测特征数量。错误: {e}"); raise
        self.outcomes_df = pd.read_parquet(outcomes_file)
        self.outcome_map = self.outcomes_df.set_index(['stay_id', 'prediction_hour'])['outcome_death_next_24h']
        self.file_index_map = self._build_file_index()
        self.file_data_cache = {}
        self.total_samples = sum(entry['count'] for entry in self.file_index_map)
        print(f"数据集初始化完成。总样本数: {self.total_samples}")

    def _build_file_index(self):
        index_map = []
        current_global_offset = 0
        for f_path in tqdm(self.pkl_files, desc="构建文件索引"):
            try:
                with open(f_path, 'rb') as file: data_in_file = pickle.load(file); num_samples_in_file = len(data_in_file)
                if num_samples_in_file > 0:
                    index_map.append({'path': f_path, 'offset': current_global_offset, 'count': num_samples_in_file})
                    current_global_offset += num_samples_in_file
            except Exception as e: print(f"警告: 处理文件 {f_path} 时出错: {e}")
        if not index_map: raise ValueError("未能从任何PKL文件构建有效的索引。")
        return index_map

    def __len__(self): return self.total_samples

    def _load_file_data_from_cache_or_disk(self, file_path):
        if file_path not in self.file_data_cache:
            with open(file_path, 'rb') as f: self.file_data_cache[file_path] = pickle.load(f)
            if len(self.file_data_cache) > 3: oldest_key = next(iter(self.file_data_cache)); del self.file_data_cache[oldest_key]
        return self.file_data_cache[file_path]

    def get_item_features_and_label(self, global_idx):
        target_file_path, local_idx_in_file = None, -1
        for file_info in self.file_index_map:
            if file_info['offset'] <= global_idx < file_info['offset'] + file_info['count']:
                target_file_path, local_idx_in_file = file_info['path'], global_idx - file_info['offset']; break
        if target_file_path is None: raise IndexError(f"全局索引 {global_idx} 超出范围。")
        data_chunk_for_file = self._load_file_data_from_cache_or_disk(target_file_path)
        stay_id, pred_hour, feat_array_original_len = data_chunk_for_file[local_idx_in_file]
        current_seq_len = feat_array_original_len.shape[0]
        if current_seq_len > self.max_len_feat: feat_array_processed = feat_array_original_len[:self.max_len_feat, :]
        elif current_seq_len < self.max_len_feat:
            padding_shape = (self.max_len_feat - current_seq_len, self.n_features_at_timestep)
            padding_block = np.full(padding_shape, self.padding_value, dtype=feat_array_original_len.dtype)
            feat_array_processed = np.vstack((feat_array_original_len, padding_block))
        else: feat_array_processed = feat_array_original_len
        features_flattened = feat_array_processed.reshape(-1)
        try: label = self.outcome_map.loc[(stay_id, pred_hour)]; return features_flattened, label, True
        except KeyError: return features_flattened, -1, False

    def generate_batches_for_xgboost(self, indices_to_load, batch_size_load):
        if indices_to_load is None: indices_to_load = range(len(self))
        num_total_indices = len(indices_to_load)
        num_batches_to_load = (num_total_indices + batch_size_load - 1) // batch_size_load
        for i in tqdm(range(num_batches_to_load), desc="分批加载数据生成DMatrix"):
            start_offset, end_offset = i * batch_size_load, min((i + 1) * batch_size_load, num_total_indices)
            current_batch_indices = indices_to_load[start_offset:end_offset]
            batch_features_list, batch_labels_list = [], []
            for global_idx in current_batch_indices:
                features, label, is_valid = self.get_item_features_and_label(global_idx)
                if is_valid: batch_features_list.append(features); batch_labels_list.append(label)
            if batch_features_list: yield np.array(batch_features_list, dtype=np.float32), np.array(batch_labels_list, dtype=np.int32)

# --- 指标计算函数 (与PyTorch脚本的calculate_test_metrics对齐) ---
def calculate_metrics_bundle(y_true, y_pred_probs, stage_name=""):
    """计算一套完整的二分类评估指标，返回字典"""
    metrics_dict = {'loss': np.nan, 'auc': 0.0, 'auprc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0}
    y_true_np, y_pred_probs_np = np.array(y_true), np.array(y_pred_probs)

    if len(y_true_np) == 0 or len(y_pred_probs_np) == 0 : # 添加对y_pred_probs_np的检查
        print(f"警告 ({stage_name}): 真实标签或预测概率为空，无法计算指标。")
        return metrics_dict

    # Log Loss (BCE Loss) - XGBoost predict输出的是概率
    epsilon = 1e-15 # 避免log(0)
    y_pred_probs_clipped = np.clip(y_pred_probs_np, epsilon, 1 - epsilon)
    try:
        metrics_dict['loss'] = float(log_loss(y_true_np, y_pred_probs_clipped))
    except Exception as e_logloss:
        print(f"警告 ({stage_name}): 计算Log Loss出错: {e_logloss}")

    if len(np.unique(y_true_np)) > 1: # 确保至少有两个类别
        try:
            metrics_dict['auc'] = float(roc_auc_score(y_true_np, y_pred_probs_np))
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_np, y_pred_probs_np)
            metrics_dict['auprc'] = float(auc(recall_curve, precision_curve))
            
            y_pred_binary = (y_pred_probs_np >= 0.5).astype(int) # 标准阈值0.5
            metrics_dict['f1'] = float(f1_score(y_true_np, y_pred_binary, zero_division=0))
            metrics_dict['precision'] = float(precision_score(y_true_np, y_pred_binary, zero_division=0))
            metrics_dict['recall'] = float(recall_score(y_true_np, y_pred_binary, zero_division=0))
            
            # 确保 labels=[0,1] 用于混淆矩阵，如果数据中确实只包含这两类
            cm_labels_to_use = [0,1] if (0 in y_true_np and 1 in y_true_np) else np.unique(y_true_np).tolist()
            if len(cm_labels_to_use) == 2: # 仅在二分类情况下计算 TN, FP, FN, TP
                tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_binary, labels=cm_labels_to_use).ravel()
                metrics_dict['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            else:
                metrics_dict['specificity'] = 0.0 # 或 np.nan
        except ValueError as e_val:
            print(f"警告 ({stage_name}): 计算核心指标时出现ValueError: {e_val}。标签数据可能存在问题。")
            # 保留已计算的loss，其他设为0
            for k_metric in ['auc', 'auprc', 'f1', 'precision', 'recall', 'specificity']:
                metrics_dict.setdefault(k_metric, 0.0)
        except Exception as e_other: # 捕获其他潜在错误
            print(f"警告 ({stage_name}): 计算核心指标时出现未知错误: {e_other}")
            for k_metric in ['auc', 'auprc', 'f1', 'precision', 'recall', 'specificity']:
                metrics_dict.setdefault(k_metric, 0.0)
    else:
        print(f"警告 ({stage_name}): 标签种类不足 ({np.unique(y_true_np)})，部分指标可能无意义或为0。")
        for k_metric in ['auc', 'auprc', 'f1', 'precision', 'recall', 'specificity']:
            metrics_dict.setdefault(k_metric, 0.0)
            
    return metrics_dict

# --- XGBoost自定义回调，用于记录详细指标并写入TSV日志 ---
class XGBCustomMetricsAndLoggingCallback(xgb.callback.TrainingCallback):
    def __init__(self, dtrain_cb, dval_cb, tsv_log_path_cb, initial_eta_cb, model_name_cb):
        super().__init__()
        self.dtrain = dtrain_cb
        self.dval = dval_cb
        self.tsv_log_path = tsv_log_path_cb
        self.initial_eta = initial_eta_cb
        self.model_name = model_name_cb # 例如 "XGBoost_Granular"
        self.history_for_plotting = [] # 存储每轮的指标字典，用于后续绘图

        # 写入TSV文件头
        header_cols = [
            "Epoch", "Train_Loss", "Train_AUC", "Train_F1", "Train_Prec", "Train_Rec", "Train_Spec",
            "Val_Loss", "Val_AUC", "Val_F1", "Val_Prec", "Val_Rec", "Val_Spec", "LR"
        ]
        try:
            with open(self.tsv_log_path, 'w') as f_log:
                f_log.write("\t".join(header_cols) + "\n")
        except IOError as e:
            print(f"错误: 无法写入TSV日志文件表头 {self.tsv_log_path}: {e}")


    def after_iteration(self, model, epoch, evals_log):
        # epoch 是当前迭代轮数 (0-indexed)
        current_boosting_round = epoch + 1 # 转为1-indexed的Epoch/Boosting Round
        
        # 注意：在回调中频繁对整个dtrain和dval进行predict开销很大。
        # XGBoost的evals_log已经包含了params['eval_metric']中指定的指标。
        # 我们主要依赖evals_log获取loss和auc，其他指标如果必须每轮都计算，则需接受这个开销。
        # 或者，可以修改为仅在特定间隔（如每10轮）计算完整指标。
        # 为与PyTorch脚本输出完全一致，这里保留每轮计算。

        # 从evals_log获取内置评估指标 (更高效)
        train_loss_xgb = evals_log['train'].get('logloss', [np.nan])[-1] # 假设logloss在eval_metric中
        train_auc_xgb = evals_log['train'].get('auc', [np.nan])[-1]     # 假设auc在eval_metric中
        val_loss_xgb = evals_log['validation'].get('logloss', [np.nan])[-1]
        val_auc_xgb = evals_log['validation'].get('auc', [np.nan])[-1]

        # ---- 如果需要计算其他指标，则进行预测 ----
        # 为减少开销，可以选择性地只在需要时（例如，最后几轮或特定间隔）计算这些额外指标
        # 或者，如果eval_metric中已包含如error等，可以利用它们
        # 以下代码计算所有自定义指标，开销较大
        train_preds_probs_cb = model.predict(self.dtrain, iteration_range=(0, current_boosting_round))
        train_labels_cb = self.dtrain.get_label()
        # 仅计算F1, Prec, Rec, Spec (AUC和Loss已从evals_log获取)
        train_metrics_custom = calculate_metrics_bundle(train_labels_cb, train_preds_probs_cb, f"Train_Round{current_boosting_round}")
        
        val_preds_probs_cb = model.predict(self.dval, iteration_range=(0, current_boosting_round))
        val_labels_cb = self.dval.get_label()
        val_metrics_custom = calculate_metrics_bundle(val_labels_cb, val_preds_probs_cb, f"Val_Round{current_boosting_round}")
        # ---- 额外指标计算结束 ----

        # 使用从evals_log获取的loss和auc，其他指标用自定义计算的
        current_train_metrics = {
            'loss': train_loss_xgb if not np.isnan(train_loss_xgb) else train_metrics_custom['loss'],
            'auc': train_auc_xgb if not np.isnan(train_auc_xgb) else train_metrics_custom['auc'],
            'f1': train_metrics_custom['f1'],
            'precision': train_metrics_custom['precision'],
            'recall': train_metrics_custom['recall'],
            'specificity': train_metrics_custom['specificity']
        }
        current_val_metrics = {
            'loss': val_loss_xgb if not np.isnan(val_loss_xgb) else val_metrics_custom['loss'],
            'auc': val_auc_xgb if not np.isnan(val_auc_xgb) else val_metrics_custom['auc'],
            'f1': val_metrics_custom['f1'],
            'precision': val_metrics_custom['precision'],
            'recall': val_metrics_custom['recall'],
            'specificity': val_metrics_custom['specificity']
        }

        current_lr_val = self.initial_eta # XGBoost的eta通常在训练开始时设定

        log_line_data_list = [
            str(current_boosting_round),
            f"{current_train_metrics['loss']:.4f}", f"{current_train_metrics['auc']:.4f}", f"{current_train_metrics['f1']:.4f}",
            f"{current_train_metrics['precision']:.4f}", f"{current_train_metrics['recall']:.4f}", f"{current_train_metrics['specificity']:.4f}",
            f"{current_val_metrics['loss']:.4f}", f"{current_val_metrics['auc']:.4f}", f"{current_val_metrics['f1']:.4f}",
            f"{current_val_metrics['precision']:.4f}", f"{current_val_metrics['recall']:.4f}", f"{current_val_metrics['specificity']:.4f}",
            f"{current_lr_val:.2e}"
        ]
        
        try:
            with open(self.tsv_log_path, 'a') as f_log_append:
                f_log_append.write("\t".join(log_line_data_list) + "\n")
        except IOError as e:
            print(f"错误: 写入TSV日志文件时发生IO错误 {self.tsv_log_path}: {e}")


        # 打印到控制台 (与PyTorch脚本风格一致)
        print(f"Epoch {current_boosting_round} | {self.model_name} | Train: Loss={current_train_metrics['loss']:.4f} AUC={current_train_metrics['auc']:.4f} F1={current_train_metrics['f1']:.4f} | "
              f"Val: Loss={current_val_metrics['loss']:.4f} AUC={current_val_metrics['auc']:.4f} F1={current_val_metrics['f1']:.4f}")
        
        # 记录用于绘图的历史
        self.history_for_plotting.append({
            'Epoch': current_boosting_round, # 使用Epoch作为键以与绘图函数兼容
            'Train_Loss': current_train_metrics['loss'], 'Train_AUC': current_train_metrics['auc'], 
            'Train_F1': current_train_metrics['f1'], 'Train_Precision': current_train_metrics['precision'],
            'Train_Recall': current_train_metrics['recall'], 'Train_Specificity': current_train_metrics['specificity'],
            'Val_Loss': current_val_metrics['loss'], 'Val_AUC': current_val_metrics['auc'],
            'Val_F1': current_val_metrics['f1'], 'Val_Precision': current_val_metrics['precision'],
            'Val_Recall': current_val_metrics['recall'], 'Val_Specificity': current_val_metrics['specificity'],
            'LR': current_lr_val # 记录学习率
        })
        
        # 早停由XGBoost内置的EarlyStopping回调处理，这里不需要额外逻辑
        # XGBoost的model对象在save_best=True时，已经是最佳状态的引用
        return False # 返回False表示继续训练

# --- 绘图函数 (与PyTorch脚本一致，可复用) ---
def plot_training_metrics_from_history(training_history_list_of_dicts, model_name_plot, output_plots_dir):
    """根据记录的字典列表历史绘制训练过程指标图"""
    if not training_history_list_of_dicts:
        print(f"警告: {model_name_plot} 的训练历史为空，无法绘制指标图。")
        return
    try:
        history_df = pd.DataFrame(training_history_list_of_dicts)
        if history_df.empty or 'Epoch' not in history_df.columns:
             print(f"警告: {model_name_plot} 的训练历史DataFrame为空或缺少'Epoch'列。")
             return

        epochs_ran = len(history_df)
        if epochs_ran == 0: return
            
        epoch_range = history_df['Epoch']
        
        # 确保键名与PyTorch脚本的绘图函数一致
        metric_plot_config = [
            ('Loss', 'Loss Value', 'Loss Function'), ('AUC', 'AUC Value', 'Area Under Curve (AUC)'), 
            ('F1', 'F1 Score', 'F1 Score'), ('Precision', 'Precision Value', 'Precision'), 
            ('Recall', 'Recall Value', 'Recall (Sensitivity)'), ('Specificity', 'Specificity Value', 'Specificity')
        ]

        num_cols_plot = 3
        num_rows_plot = (len(metric_plot_config) + num_cols_plot - 1) // num_cols_plot
        fig, axes = plt.subplots(num_rows_plot, num_cols_plot, figsize=(18, 5 * num_rows_plot))
        axes = axes.ravel() if num_rows_plot > 1 or num_cols_plot > 1 else [axes] # 确保axes可迭代
        
        fig.suptitle(f'{model_name_plot} - Training Process Metrics', fontsize=16, y=1.02 if num_rows_plot > 1 else 1.05)

        for i, (metric_key_short, ylabel_text, title_text) in enumerate(metric_plot_config):
            ax = axes[i]
            train_col = f'Train_{metric_key_short}'
            val_col = f'Val_{metric_key_short}'
            if train_col in history_df and val_col in history_df:
                ax.plot(epoch_range, history_df[train_col], marker='o', linestyle='-', label=f'Train {metric_key_short}')
                ax.plot(epoch_range, history_df[val_col], marker='x', linestyle='--', label=f'Validation {metric_key_short}')
                ax.set_title(title_text)
                ax.set_xlabel('Epoch (Boosting Round)')
                ax.set_ylabel(ylabel_text)
                ax.legend()
                ax.grid(True)
            else:
                print(f"警告: 在训练历史中未找到指标 {train_col} 或 {val_col}，无法绘制。")
                ax.text(0.5, 0.5, f'Data for {metric_key_short}\nnot available', ha='center', va='center')
                ax.set_title(title_text)


        # 隐藏多余的子图 (如果存在)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # 文件名与PyTorch脚本一致
        plot_file_path = os.path.join(output_plots_dir, f'{model_name_plot.lower().replace(" ", "_")}_training_metrics.png')
        fig.savefig(plot_file_path)
        plt.close(fig)
        print(f"{model_name_plot} 训练过程指标图已保存到: {plot_file_path}")
    except Exception as e_plot_hist:
        print(f"绘制 {model_name_plot} 训练历史图表时出错: {e_plot_hist}")
        traceback.print_exc()


def plot_roc_pr_cm_curves(true_labels, pred_probs, model_name_plot, output_plots_dir):
    """为指定模型绘制ROC, PR曲线和混淆矩阵 (与PyTorch脚本的函数类似)"""
    # ... (此函数逻辑与您之前提供的PyTorch脚本中的版本基本一致，确保文件名和标题使用model_name_plot)
    # 我将直接复制并调整该函数，确保它在此处可用
    plot_paths = {}
    full_model_name_for_file = model_name_plot.lower().replace(" ", "_") 

    if len(true_labels) == 0 or len(pred_probs) == 0 or len(np.unique(true_labels)) < 2:
        print(f"警告: {model_name_plot} 测试集数据不足以绘制ROC/PR/CM图表。")
        return plot_paths
    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        roc_auc_val = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'{model_name_plot} - ROC Curve (Test Set)'); ax_roc.legend(loc="lower right"); ax_roc.grid(True)
        roc_plot_path = os.path.join(output_plots_dir, f'{full_model_name_for_file}_roc_curve_test.png')
        fig_roc.savefig(roc_plot_path); plt.close(fig_roc); plot_paths['ROC'] = roc_plot_path
        print(f"{model_name_plot} ROC曲线图已保存到: {roc_plot_path}")

        # PR Curve
        precision_vals, recall_vals, _ = precision_recall_curve(true_labels, pred_probs)
        pr_auc_val = auc(recall_vals, precision_vals)
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        ax_pr.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR Curve (AUPRC = {pr_auc_val:.4f})')
        ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
        ax_pr.set_ylim([0.0, 1.05]); ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_title(f'{model_name_plot} - Precision-Recall Curve (Test Set)'); ax_pr.legend(loc="lower left"); ax_pr.grid(True)
        pr_plot_path = os.path.join(output_plots_dir, f'{full_model_name_for_file}_pr_curve_test.png')
        fig_pr.savefig(pr_plot_path); plt.close(fig_pr); plot_paths['PR'] = pr_plot_path
        print(f"{model_name_plot} PR曲线图已保存到: {pr_plot_path}")

        # Confusion Matrix
        pred_binary = (np.array(pred_probs) >= 0.5).astype(int)
        # 确保标签是[0, 1]，即使数据中只有一个类别，以避免confusion_matrix报错
        cm_labels_to_use = [0,1] 
        # 注意: 如果真实标签中只有一个类别，confusion_matrix的行为可能依赖于sklearn版本
        # 理想情况下，测试集应该包含所有类别。如果不是，这里的绘制可能不完美。
        cm = confusion_matrix(true_labels, pred_binary, labels=cm_labels_to_use) 
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        ax_cm.set_xlabel('Predicted Label'); ax_cm.set_ylabel('True Label')
        ax_cm.set_title(f'{model_name_plot} - Confusion Matrix (Test, Threshold=0.50)')
        cm_plot_path = os.path.join(output_plots_dir, f'{full_model_name_for_file}_confusion_matrix_test.png')
        fig_cm.savefig(cm_plot_path); plt.close(fig_cm); plot_paths['ConfusionMatrix'] = cm_plot_path
        print(f"{model_name_plot} 混淆矩阵图已保存到: {cm_plot_path}")
    except Exception as e: print(f"为 {model_name_plot} 绘制评估图表时出错: {e}"); traceback.print_exc()
    return plot_paths

# --- 主训练和评估函数 ---
def train_and_evaluate_xgboost():
    """训练并评估XGBoost模型"""
    print(f"\n===== 开始处理 {MODEL_NAME_XGB} =====")
    # ... (Dataset, 索引划分, DMatrix构建与之前版本相同) ...
    dataset = GranularPklDataset(
        pkl_dir=os.path.join(OUTPUT_DIR, 'step4b_temp_granular'),
        outcomes_file=os.path.join(OUTPUT_DIR, f'step3_dynamic_outcomes_{COHORT_KEY}.parquet'),
        cohort_key_for_glob=COHORT_KEY, max_len_feat=MAX_LEN, padding_value=PADDING_VALUE_INPUT
    )
    max_samples_debug = 50000
    indices = list(range(min(len(dataset), max_samples_debug))) if max_samples_debug and len(dataset) > max_samples_debug else list(range(len(dataset)))
    if max_samples_debug and len(dataset) > max_samples_debug: print(f"警告: 当前使用数据子集进行测试，样本数上限: {max_samples_debug}")
    print(f"将使用 {len(indices)} 个样本进行训练、验证和测试。")
    train_indices, test_indices = train_test_split(indices, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
    train_indices, val_indices = train_test_split(train_indices, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=SEED, shuffle=True)
    print(f"训练集索引数: {len(train_indices)}, 验证集索引数: {len(val_indices)}, 测试集索引数: {len(test_indices)}")
    dtrain = build_dmatrix_from_indices(train_indices, "训练(Train)", dataset)
    dval = build_dmatrix_from_indices(val_indices, "验证(Validation)", dataset)
    dtest = build_dmatrix_from_indices(test_indices, "测试(Test)", dataset)
    
    train_labels_for_pos_weight = dtrain.get_label()
    calculated_pos_weight = (len(train_labels_for_pos_weight) - np.sum(train_labels_for_pos_weight)) / np.sum(train_labels_for_pos_weight) if np.sum(train_labels_for_pos_weight) > 0 else 1.0
    print(f"为XGBoost计算得到的 scale_pos_weight: {calculated_pos_weight:.2f}")

    params = {
        'objective': 'binary:logistic', 'eval_metric': ['auc', 'logloss'],
        'max_depth': 4, 'eta': 0.1, 'subsample': 0.7, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7,
        'scale_pos_weight': calculated_pos_weight,
        'min_child_weight': 5, 'tree_method': 'hist', 'grow_policy': 'lossguide', 'max_leaves': 32, 'verbosity': 1
    }
    # ... (GPU检测逻辑与之前一致) ...
    try: 
        if xgb.config.get_config().get('use_cuda', False): params['device'] = 'cuda'
    except Exception: pass 
    if 'device' in params: print(f"XGBoost将尝试使用设备: {params['device']}")
    else: print("XGBoost将使用CPU。")
    
    print("\n开始训练XGBoost模型...")
    # 将dtrain, dval传递给回调函数
    xgb_custom_logger = XGBCustomMetricsAndLoggingCallback(dtrain, dval, XGB_TRAINING_LOG_PATH, params['eta'], MODEL_NAME_XGB)
    
    model = xgb.train(
        params, dtrain, num_boost_round=XGB_EPOCHS,
        evals=[(dtrain, 'train'), (dval, 'validation')], # evals_log仍然会被填充
        callbacks=[xgb.callback.EarlyStopping(rounds=XGB_ES_PATIENCE, metric_name='auc', maximize=True, save_best=True),
                   xgb_custom_logger], 
        verbose_eval=False 
    )
    print(f"XGBoost模型训练完成。最佳迭代轮数 (0-indexed): {model.best_iteration}")

    # 使用自定义回调记录的history进行绘图
    plot_training_metrics_from_history(xgb_custom_logger.history_for_plotting, MODEL_NAME_XGB, RESULTS_OUTPUT_DIR)
    
    # 保存最佳模型 (EarlyStopping的save_best=True会在训练结束时使model对象处于最佳状态)
    # 但为了路径统一，我们仍然保存到XGB_MODEL_PATH
    # XGBoost的save_best会保存到模型内部，我们可以通过XGB_ES_CHECKPOINT_PATH来加载，或者直接用训练结束的model（已经是最佳）
    model.save_model(XGB_MODEL_PATH) # 直接保存训练结束时（即最佳）的模型
    print(f"XGBoost最佳模型已保存到: {XGB_MODEL_PATH}")
    
    print("\n在测试集上评估最佳XGBoost模型...")
    # 可以直接用训练结束的model进行预测，因为它已经是最佳迭代的模型
    test_preds_probs = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    test_labels = dtest.get_label()
    
    test_metrics_xgb = calculate_metrics_bundle(test_labels, test_preds_probs, f"{MODEL_NAME_XGB}-Test")
    
    print(f"\n{MODEL_NAME_XGB} - 测试集评估结果:")
    for name, val in test_metrics_xgb.items(): print(f"  {name.capitalize()}: {val:.4f}")
    
    with open(XGB_TEST_RESULTS_PATH, 'w') as f: json.dump(test_metrics_xgb, f, indent=4)
    print(f"{MODEL_NAME_XGB} 测试集评估结果已保存到: {XGB_TEST_RESULTS_PATH}")
    
    plot_roc_pr_cm_curves(test_labels, test_preds_probs, MODEL_NAME_XGB, RESULTS_OUTPUT_DIR)

    del dtrain, dval, dtest; gc.collect(); print("DMatrix对象已清理。")

# --- build_dmatrix_from_indices (辅助函数，与之前版本一致) ---
def build_dmatrix_from_indices(indices_subset, name_subset, data_loader_instance):
    print(f"开始为 {name_subset} 集构建DMatrix...")
    all_features_list, all_labels_list = [], []
    for X_batch_np, y_batch_np in data_loader_instance.generate_batches_for_xgboost(indices_subset, DATA_LOADER_BATCH_SIZE):
        all_features_list.append(X_batch_np); all_labels_list.append(y_batch_np); gc.collect()
    if not all_features_list: raise ValueError(f"{name_subset} 数据集为空，无法创建DMatrix。")
    X_full_np, y_full_np = np.vstack(all_features_list), np.concatenate(all_labels_list)
    del all_features_list, all_labels_list; gc.collect()
    print(f"  {name_subset} 集数据形状: X={X_full_np.shape}, y={y_full_np.shape}")
    dmatrix_obj = xgb.DMatrix(X_full_np, label=y_full_np, missing=PADDING_VALUE_INPUT, nthread=-1)
    del X_full_np, y_full_np; gc.collect()
    print(f"  {name_subset} DMatrix 创建完成。")
    return dmatrix_obj

if __name__ == "__main__":
    script_start_time = time.time()
    train_and_evaluate_xgboost() # 调用主函数
    script_end_time = time.time()
    print(f"\n{MODEL_NAME_XGB} 脚本总运行时间: {(script_end_time - script_start_time) / 60:.2f} 分钟")