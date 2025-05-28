import os
import glob
import gc
import json
import pickle
import time
import math
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
import functools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm

# --- 抑制警告 ---
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True  # 启用CuDNN优化
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速

# ---------------------- 全局配置 -----------------------------
OUTPUT_DIR = './output_dynamic'
COHORT_KEY = 'aki'

# 文件路径
TEMP_GRANULAR_DIR = os.path.join(OUTPUT_DIR, 'step4b_temp_granular')
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'step5_models_pytorch_{COHORT_KEY}')
RESULTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'step5_results_pytorch_{COHORT_KEY}')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

BIN_EDGES_FILE = os.path.join(OUTPUT_DIR, f'step4ab_bin_edges_{COHORT_KEY}_5bins.pkl')
OUTCOMES_FILE = os.path.join(OUTPUT_DIR, f'step3_dynamic_outcomes_{COHORT_KEY}.parquet')

# --- 数据维度 ---
with open(BIN_EDGES_FILE, 'rb') as f:
    bin_edges = pickle.load(f)
N_FEATURES_GRANULAR = len(bin_edges)
N_BINS = len(next(iter(bin_edges.values()))) - 1

MAX_LEN = 240
PADDING_VALUE_INPUT = -1
PADDING_IDX_EMBEDDING = 0

# --- 训练参数 ---
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15  # 添加缺失的VAL_SPLIT参数
SEED = 42
BATCH_SIZE_MODEL = 64
EPOCHS = 20
ES_PATIENCE = 10

# --- GPU配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENABLE_AMP = True
NUM_WORKERS = 0
PIN_MEMORY = True

# --- 模型参数 ---
EMBEDDING_DIM = 32
LSTM_HIDDEN_DIM = 64
D_MODEL_TRANSFORMER = 128


# ---------------------- 模型组件定义 ----------------------

class PositionalEncoding(nn.Module):
    """Transformer的位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EarlyStoppingAndCheckpoint:
    """早停与模型保存类"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',
                 monitor='val_loss', mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.best_metric = np.Inf if mode == 'min' else -np.Inf

    def __call__(self, current_metric, model):
        score = -current_metric if self.mode == 'min' else current_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class GranularPklDataset(Dataset):
    """内存优化数据集（按需加载）"""

    def __init__(self, identifiers_df, file_mapping, padding_value_input, embedding_pad_idx, n_features, batch_size=64):
        self.samples = []
        self.embedding_pad_idx = embedding_pad_idx
        self.n_features = n_features

        # 步骤1：建立文件索引
        print("\n[1/3] 建立文件索引...")
        file_info_list = []
        global_idx = 0
        for file_info in file_mapping:
            file_path = file_info['path']
            count = file_info['count']
            file_info_list.append((file_path, global_idx, global_idx + count))
            global_idx += count
        
        # 步骤2：初始化文件数据缓存 (不预加载所有文件)
        print("[2/3] 初始化文件数据缓存...")
        self.file_data_cache = {}  # 使用缓存避免重复加载

        # 步骤3：流式处理数据
        print("[3/3] 流式加载样本...")
        self.identifiers_df = identifiers_df
        self.file_info_list = file_info_list

    def _load_file(self, path):
        """按需加载文件"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.identifiers_df)

    def __getitem__(self, idx):
        """返回单个样本"""
        file_path = None
        local_idx = -1
        # 找到文件和数据的索引
        for fp, start, end in self.file_info_list:
            if start <= idx < end:
                file_path = fp
                local_idx = idx - start
                break
        
        if file_path is None:
            raise IndexError(f"Index {idx} out of bounds or file_path not found.")

        # 获取文件数据 (按需加载并缓存)
        if file_path not in self.file_data_cache:
            self.file_data_cache[file_path] = self._load_file(file_path)
        
        data_chunk = self.file_data_cache[file_path]
        _, _, feat_array = data_chunk[local_idx]

        # 处理数据
        x_adjusted = feat_array.astype(np.int64) + 1  # 将标签从 -1 映射到 0
        y = self.identifiers_df.iloc[idx]['y']
        return torch.from_numpy(x_adjusted).long(), torch.tensor(y, dtype=torch.float32)


def vectorized_pad_collate(batch, max_len, n_features, padding_value):
    """向量化填充函数"""
    if not batch:
        return (
            torch.zeros((0, max_len, n_features), dtype=torch.long),
            torch.zeros((0, max_len), dtype=torch.bool),
            torch.zeros((0,), dtype=torch.float32)
        )

    sequences, labels = zip(*batch)

    # 计算实际长度
    lengths = [min(len(seq), max_len) for seq in sequences]

    padded_seqs = torch.full((len(batch), max_len, n_features),
                             padding_value, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, (seq, seq_len) in enumerate(zip(sequences, lengths)):
        if seq_len > 0:
            padded_seqs[i, :seq_len] = seq[:seq_len]
            mask[i, :seq_len] = True

    return padded_seqs, mask, torch.stack(labels)


# ---------------------- 模型架构 ----------------------

class LSTMClassifierGranular(nn.Module):
    """基于LSTM的分类器"""
    def __init__(self, n_features, n_bins, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_bins, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim * n_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        x = self.embedding(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output.squeeze(-1)


class TransformerClassifierGranular(nn.Module):
    """基于Transformer的分类器"""
    def __init__(self, n_features, n_bins, embedding_dim, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(n_bins, embedding_dim, padding_idx=0)
        self.feature_proj = nn.Linear(embedding_dim * n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, n_features = x.shape
        x = self.embedding(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.feature_proj(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        if mask is not None:
            mask = ~mask
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x[-1]
        output = self.fc(x)
        return output.squeeze(-1)


# ---------------------- 训练与评估函数 ----------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, is_transformer, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds_probs, all_labels = [], []
    
    with tqdm(loader, desc=f'Epoch {epoch + 1}', dynamic_ncols=True, mininterval=0.5, leave=False) as pbar:
        for batch in pbar:
            x, mask, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            with autocast(enabled=True):
                output = model(x, mask) if is_transformer else model(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            all_preds_probs.extend(torch.sigmoid(output).cpu().detach().numpy())
            all_labels.extend(y.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = total_loss / len(loader)
    
    # 计算各项指标
    all_labels_np = np.array(all_labels)
    all_preds_probs_np = np.array(all_preds_probs)
    all_preds_binary = (all_preds_probs_np > 0.5).astype(int)

    epoch_auc = epoch_f1 = epoch_precision = epoch_recall = epoch_specificity = 0.0

    if len(np.unique(all_labels_np)) > 1:  # 确保至少有两个类别
        epoch_auc = roc_auc_score(all_labels_np, all_preds_probs_np)
        epoch_f1 = f1_score(all_labels_np, all_preds_binary, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(all_labels_np, all_preds_binary, labels=[0, 1]).ravel()
        
        epoch_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        epoch_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall (Sensitivity)
        epoch_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        print(f"警告: Epoch {epoch + 1} (训练) 标签种类不足，部分指标可能无法计算或无意义。")

    return epoch_loss, epoch_auc, epoch_f1, epoch_precision, epoch_recall, epoch_specificity


def validate_one_epoch(model, loader, criterion, device, is_transformer):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_preds_probs, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            x, mask, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            output = model(x, mask) if is_transformer else model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            all_preds_probs.extend(torch.sigmoid(output).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    epoch_loss = total_loss / len(loader)

    # 计算各项指标
    all_labels_np = np.array(all_labels)
    all_preds_probs_np = np.array(all_preds_probs)
    all_preds_binary = (all_preds_probs_np > 0.5).astype(int)

    epoch_auc = epoch_f1 = epoch_precision = epoch_recall = epoch_specificity = 0.0

    if len(np.unique(all_labels_np)) > 1:  # 确保至少有两个类别
        epoch_auc = roc_auc_score(all_labels_np, all_preds_probs_np)
        epoch_f1 = f1_score(all_labels_np, all_preds_binary, zero_division=0)
        
        # 确保混淆矩阵的labels参数正确处理可能的单类情况
        unique_labels = np.unique(all_labels_np)
        if len(unique_labels) == 2:  # 只有在同时存在0和1时才计算混淆矩阵相关指标
            cm_labels = [0,1]
            tn, fp, fn, tp = confusion_matrix(all_labels_np, all_preds_binary, labels=cm_labels).ravel()
            epoch_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            epoch_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall (Sensitivity)
            epoch_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        elif len(unique_labels) == 1:
            print(f"警告: 验证轮次中仅存在单一类别标签 ({unique_labels[0]})，精确率、召回率、特异性将为0。")
    else:
        print("警告: 验证轮次标签种类不足，部分指标可能无法计算或无意义。")
        
    return epoch_loss, epoch_auc, epoch_f1, epoch_precision, epoch_recall, epoch_specificity


def train_model(model, name, optimizer, lr_scheduler, es_patience):
    """训练模型的完整流程"""
    print(f"\n开始训练 {name}...")
    best_auc = 0 
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=ENABLE_AMP)
    es_checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, f'es_checkpoint_{name}.pt')
    es = EarlyStoppingAndCheckpoint(patience=es_patience, verbose=True, path=es_checkpoint_path, monitor='val_auc', mode='max')

    history = {
        'loss': [], 'auc': [], 'f1': [], 'precision': [], 'recall': [], 'specificity': [],
        'val_loss': [], 'val_auc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'val_specificity': []
    }
    
    for epoch in range(EPOCHS):
        # 训练阶段
        train_loss, train_auc, train_f1, train_precision, train_recall, train_specificity = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, 
            isinstance(model, TransformerClassifierGranular), epoch
        )
        
        # 验证阶段
        val_loss, val_auc, val_f1, val_precision, val_recall, val_specificity = validate_one_epoch(
            model, val_loader, criterion, DEVICE, isinstance(model, TransformerClassifierGranular)
        )

        # 记录历史
        metrics = [train_loss, train_auc, train_f1, train_precision, train_recall, train_specificity,
                  val_loss, val_auc, val_f1, val_precision, val_recall, val_specificity]
        metric_names = ['loss', 'auc', 'f1', 'precision', 'recall', 'specificity',
                       'val_loss', 'val_auc', 'val_f1', 'val_precision', 'val_recall', 'val_specificity']
        
        for metric_name, metric_value in zip(metric_names, metrics):
            history[metric_name].append(metric_value)

        # 学习率调整
        lr_scheduler.step(val_auc if name == 'Transformer' else val_loss) 

        # 早停检查 (使用val_auc)
        es(val_auc, model) 
        if es.early_stop:
            print(f"{name} 早停于第 {epoch + 1} 轮，因为 {es.monitor} 未改善。")
            print(f"加载早停前最佳模型权重: {es.path}")
            model.load_state_dict(torch.load(es.path))
            break

        # 保存最佳模型 (基于val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_path = os.path.join(MODEL_OUTPUT_DIR, f'best_{name}.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"最佳模型 ({name}) 已保存到 {best_model_path} (Val AUC: {best_auc:.4f})")

        print(f"Epoch {epoch + 1}/{EPOCHS} | {name} | "
              f"Loss: {train_loss:.4f} (Val: {val_loss:.4f}) | "
              f"AUC: {train_auc:.4f} (Val: {val_auc:.4f}) | "
              f"F1: {train_f1:.4f} (Val: {val_f1:.4f}) | "
              f"Prec: {train_precision:.4f} (Val: {val_precision:.4f}) | "
              f"Rec: {train_recall:.4f} (Val: {val_recall:.4f}) | "
              f"Spec: {train_specificity:.4f} (Val: {val_specificity:.4f})")

    # 确保加载最佳模型
    if not es.early_stop and os.path.exists(os.path.join(MODEL_OUTPUT_DIR, f'best_{name}.pt')):
        model.load_state_dict(torch.load(os.path.join(MODEL_OUTPUT_DIR, f'best_{name}.pt')))
    elif not es.early_stop:
        print(f"警告: {name} 训练完成但未找到 'best_{name}.pt'。模型可能未保存最佳状态。")

    return history


def evaluate(model, loader, name):
    """模型评估函数"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"评估 {name}"):
            x, mask, y = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2]  # y is on CPU
            output = model(x, mask) if isinstance(model, TransformerClassifierGranular) else model(x)
            all_preds.extend(torch.sigmoid(output).cpu().numpy())
            all_labels.extend(y.numpy())  # y was already on CPU

    results = {}
    # 确保有至少两个类别用于指标计算
    if len(np.unique(all_labels)) > 1 and len(all_labels) > 0 and len(all_preds) > 0:
        try:
            results['auc'] = roc_auc_score(all_labels, all_preds)
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            results['auprc'] = auc(recall, precision)
            
            preds_binary = (np.array(all_preds) > 0.5).astype(int)
            results['f1_score'] = f1_score(all_labels, preds_binary, zero_division=0)
        except ValueError as e:
            print(f"计算指标时出错 ({name}): {e}. 可能标签数据有问题。")
            results['auc'] = results['auprc'] = results['f1_score'] = 0.0
    else:
        print(f"警告: {name} 评估时标签种类不足或数据为空。指标可能无法定义或无意义。")
        results['auc'] = results['auprc'] = results['f1_score'] = 0.0

    return results, all_preds, all_labels


def plot_curves(labels, preds, model_name, results_dir):
    """绘制ROC和PR曲线"""
    plot_paths = {}
    
    if len(np.unique(labels)) > 1 and len(labels) > 0 and len(preds) > 0:
        try:
            # ROC曲线
            fpr, tpr, _ = roc_curve(labels, preds)
            roc_auc_val = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} Receiver Operating Characteristic (Test Set)')
            plt.legend(loc="lower right")
            roc_plot_path = os.path.join(results_dir, f'{model_name.lower()}_roc_curve_test.png')
            plt.savefig(roc_plot_path)
            plt.close()
            plot_paths['ROC'] = roc_plot_path
            
            # PR曲线
            precision, recall, _ = precision_recall_curve(labels, preds)
            pr_auc_val = auc(recall, precision)
            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc_val:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'{model_name} Precision-Recall Curve (Test Set)')
            plt.legend(loc="lower left")
            pr_plot_path = os.path.join(results_dir, f'{model_name.lower()}_pr_curve_test.png')
            plt.savefig(pr_plot_path)
            plt.close()
            plot_paths['PR'] = pr_plot_path
        except Exception as e:
            print(f"绘制 {model_name} 图表时出错: {e}")
    else:
        print(f"{model_name} 评估的标签数据不足以绘制曲线。")
        
    return plot_paths


# ---------------------- 主程序 --------------------------
if __name__ == '__main__':
    # --- 数据加载 ---
    print("\n加载数据...")
    outcomes_df = pd.read_parquet(OUTCOMES_FILE)
    outcomes_map = outcomes_df.set_index(['stay_id', 'prediction_hour'])['outcome_death_next_24h']
    all_pkl_files = sorted(glob.glob(os.path.join(TEMP_GRANULAR_DIR, f'*{COHORT_KEY}*.pkl')))

    # 构建文件映射
    file_mapping = []
    global_idx = 0
    for f_path in all_pkl_files:
        with open(f_path, 'rb') as f:
            count = len(pickle.load(f))
        file_mapping.append({'path': f_path, 'offset': global_idx, 'count': count})
        global_idx += count

    # 构建标识符
    identifiers = []
    for file_info in file_mapping:
        with open(file_info['path'], 'rb') as f:
            data = pickle.load(f)
            identifiers.extend([(s[0], s[1]) for s in data])
    identifiers_df = pd.DataFrame(identifiers, columns=['stay_id', 'prediction_hour'])
    identifiers_df['y'] = identifiers_df.apply(lambda x: outcomes_map.get((x['stay_id'], x['prediction_hour']), 0), axis=1)

    # --- 数据集分割 ---
    indices = np.arange(len(identifiers_df))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SPLIT, random_state=SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=VAL_SPLIT / (1 - TEST_SPLIT), random_state=SEED)

    # --- 创建数据加载器 ---
    print("\n创建数据加载器...")
    collate_fn = functools.partial(vectorized_pad_collate, max_len=MAX_LEN,
                                   n_features=N_FEATURES_GRANULAR, padding_value=PADDING_IDX_EMBEDDING)

    full_dataset = GranularPklDataset(identifiers_df, file_mapping, PADDING_VALUE_INPUT,
                                      PADDING_IDX_EMBEDDING, N_FEATURES_GRANULAR)

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_MODEL,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_fn)

    val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_MODEL * 2,
                            sampler=SequentialSampler(val_idx),
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_MODEL * 2,
                             sampler=SequentialSampler(test_idx),
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)

# --- 模型初始化 ---
    print("\n初始化模型...")
    lstm = LSTMClassifierGranular(N_FEATURES_GRANULAR, N_BINS + 1, EMBEDDING_DIM, LSTM_HIDDEN_DIM).to(DEVICE)
    transformer = TransformerClassifierGranular(N_FEATURES_GRANULAR, N_BINS + 1, EMBEDDING_DIM,
                                                D_MODEL_TRANSFORMER, 8, 2, 256).to(DEVICE)

    # --- 训练循环 ---
    # train_model 函数已经在之前定义过，这里不再重复
                                                
    # --- 训练模型 ---
    print("\n开始训练模型...")
    # 训练LSTM
    lstm_optim = optim.Adam(lstm.parameters(), lr=1e-3)
    lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(lstm_optim, 'min', patience=10)
    lstm_history = train_model(lstm, 'LSTM', lstm_optim, lstm_scheduler, ES_PATIENCE)

    # 训练Transformer
    trans_optim = optim.Adam(transformer.parameters(), lr=1e-4)
    trans_scheduler = optim.lr_scheduler.ReduceLROnPlateau(trans_optim, 'max', patience=10)
    trans_history = train_model(transformer, 'Transformer', trans_optim, trans_scheduler, ES_PATIENCE)

    # --- 模型评估与结果可视化 ---
    print("\n最终评估:")
    all_results_summary = {}  # 存储评估结果
    plot_paths = {}  # 存储图表路径

    # LSTM评估
    lstm_model_final_path = os.path.join(MODEL_OUTPUT_DIR, 'best_LSTM.pt')
    print(f"加载最佳LSTM模型进行评估: {lstm_model_final_path}")
    if os.path.exists(lstm_model_final_path):
        lstm.load_state_dict(torch.load(lstm_model_final_path, map_location=DEVICE))
        lstm_test_metrics, lstm_preds, lstm_labels = evaluate(lstm, test_loader, 'LSTM')
        all_results_summary['LSTM'] = lstm_test_metrics
        # 绘制LSTM模型评估曲线
        lstm_plot_paths = plot_curves(lstm_labels, lstm_preds, 'LSTM', RESULTS_OUTPUT_DIR)
        plot_paths.update({f'LSTM_{k}': v for k, v in lstm_plot_paths.items()})
    else:
        print(f"未找到最佳LSTM模型: {lstm_model_final_path}")
        all_results_summary['LSTM'] = "Model not found"

    # Transformer评估
    transformer_model_final_path = os.path.join(MODEL_OUTPUT_DIR, 'best_Transformer.pt')
    print(f"加载最佳Transformer模型进行评估: {transformer_model_final_path}")
    if os.path.exists(transformer_model_final_path):
        transformer.load_state_dict(torch.load(transformer_model_final_path, map_location=DEVICE))
        trans_test_metrics, trans_preds, trans_labels = evaluate(transformer, test_loader, 'Transformer')
        all_results_summary['Transformer'] = trans_test_metrics
        # 绘制Transformer模型评估曲线
        transformer_plot_paths = plot_curves(trans_labels, trans_preds, 'Transformer', RESULTS_OUTPUT_DIR)
        plot_paths.update({f'Transformer_{k}': v for k, v in transformer_plot_paths.items()})
    else:
        print(f"未找到最佳Transformer模型: {transformer_model_final_path}")
        all_results_summary['Transformer'] = "Model not found"

    # 保存评估结果
    final_results_path = os.path.join(RESULTS_OUTPUT_DIR, 'final_evaluation_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(all_results_summary, f, indent=4)
    print(f"\n最终评估结果已保存到: {final_results_path}")

    if plot_paths:
        print("\n评估图表已保存到:")
        for model_key, path in plot_paths.items():
            print(f"- {model_key}: {path}")

    # --- 手动测试集评估 ---
    print("\n\n--- 手动在测试集上评估已保存的最佳模型 ---")
    criterion_for_test = nn.BCEWithLogitsLoss()
    
    # 评估Transformer模型
    if os.path.exists(transformer_model_final_path):
        try:
            transformer.load_state_dict(torch.load(transformer_model_final_path, map_location=DEVICE))
            test_loss, test_auc, test_f1, test_prec, test_rec, test_spec = validate_one_epoch(
                model=transformer,
                loader=test_loader,
                criterion=criterion_for_test,
                device=DEVICE,
                is_transformer=True
            )

            print(f"\nTransformer - 测试集评估结果:")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  AUC: {test_auc:.4f}")
            print(f"  F1: {test_f1:.4f}")
            print(f"  Precision: {test_prec:.4f}")
            print(f"  Recall: {test_rec:.4f}")
            print(f"  Specificity: {test_spec:.4f}")
        except Exception as e:
            print(f"Transformer模型评估出错: {e}")
    
    # 也可以添加LSTM模型的手动评估，如有需要
    
    print("\n训练和评估完成！所有结果已保存。")


if __name__ == '__main__':
    # 主程序入口点，之前的代码包含了初始化、数据加载等部分
    pass