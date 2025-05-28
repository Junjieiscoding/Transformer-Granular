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
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc, f1_score,
                             confusion_matrix, roc_curve, precision_score, recall_score)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg") # Use Agg backend to avoid GUI issues
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns # For confusion matrix heatmap

# --- Suppress warnings ---
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True  # Enable CuDNN benchmarking for fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs for faster matmuls

# ---------------------- Global Configurations -----------------------------
OUTPUT_DIR: str = './output_dynamic' # Output directory
COHORT_KEY: str = 'aki' # Cohort identifier

# File paths
TEMP_GRANULAR_DIR: str = os.path.join(OUTPUT_DIR, 'step4b_temp_granular')
MODEL_OUTPUT_DIR: str = os.path.join(OUTPUT_DIR, f'step5_models_pytorch_{COHORT_KEY}')
RESULTS_OUTPUT_DIR: str = os.path.join(OUTPUT_DIR, f'step5_results_pytorch_{COHORT_KEY}')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

BIN_EDGES_FILE: str = os.path.join(OUTPUT_DIR, f'step4ab_bin_edges_{COHORT_KEY}_5bins.pkl')
OUTCOMES_FILE: str = os.path.join(OUTPUT_DIR, f'step3_dynamic_outcomes_{COHORT_KEY}.parquet')

# --- Data Dimensions ---
N_FEATURES_GRANULAR: int
N_BINS: int
try:
    with open(BIN_EDGES_FILE, 'rb') as f:
        bin_edges: Dict[str, np.ndarray] = pickle.load(f)
    N_FEATURES_GRANULAR = len(bin_edges)
    N_BINS = len(next(iter(bin_edges.values()))) - 1 # Assumes all features have same num bins
except FileNotFoundError:
    print(f"Critical Error: Bin edges file {BIN_EDGES_FILE} not found. Please ensure the file exists.")
    exit(1)
except Exception as e:
    print(f"Critical Error: Failed to load or parse bin edges file {BIN_EDGES_FILE}: {e}")
    exit(1)

MAX_LEN: int = 240 # Maximum sequence length
PADDING_VALUE_INPUT: int = -1 # Original padding value in input data (mapped to PADDING_IDX_EMBEDDING)
PADDING_IDX_EMBEDDING: int = 0 # Padding index used by nn.Embedding

# --- Training Parameters ---
TEST_SPLIT: float = 0.15
VAL_SPLIT: float = 0.15 # Relative to (train+val) set
SEED: int = 42
BATCH_SIZE_MODEL: int = 64
EPOCHS: int = 50
ES_PATIENCE: int = 10 # Early stopping patience

# --- GPU Configuration ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENABLE_AMP: bool = DEVICE.type == 'cuda' # Automatic Mixed Precision
NUM_WORKERS: int = 0 # DataLoader num_workers (0 often more stable, esp. on Windows)
PIN_MEMORY: bool = DEVICE.type == 'cuda' # DataLoader pin_memory

# --- Model Parameters ---
EMBEDDING_DIM: int = 32
# LSTM_HIDDEN_DIM: int = 64 # Kept for consistency if any shared part might use it (not by Transformer)
D_MODEL_TRANSFORMER: int = 128
NHEAD_TRANSFORMER: int = 4
NUM_ENCODER_LAYERS_TRANSFORMER: int = 3
DIM_FEEDFORWARD_TRANSFORMER: int = 256 # Typically 2-4x D_MODEL_TRANSFORMER
DROPOUT_TRANSFORMER: float = 0.1


# ---------------------- Model Components ----------------------

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = MAX_LEN):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe is [max_len, d_model]. Reshape for broadcasting with batch_first=True inputs.
        # PyTorch TransformerEncoderLayer expects [seq_len, batch_size, embedding_dim] by default,
        # or [batch_size, seq_len, embedding_dim] if batch_first=True.
        # Here, we prepare pe to be added to x of shape [batch_size, seq_len, d_model].
        # So, pe should be effectively [seq_len, d_model] after slicing.
        # The original `pe = pe.unsqueeze(0).transpose(0, 1)` resulted in `[max_len, 1, d_model]`.
        # Let's keep it compatible with the forward pass.
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # self.pe is [max_len, 1, d_model].
        # self.pe[:x.size(1), :] slices to [seq_len, 1, d_model].
        # .squeeze(1) makes it [seq_len, d_model]. This broadcasts with x.
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)


class EarlyStoppingAndCheckpoint:
    """Implements early stopping and saves the best model checkpoint."""
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0,
                 path: str = 'checkpoint.pt', monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.delta = delta # Minimum change to qualify as an improvement
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.val_metric_best: float = np.Inf if mode == 'min' else -np.Inf

    def __call__(self, current_metric_val: float, model: nn.Module):
        score = -current_metric_val if self.mode == 'min' else current_metric_val

        if self.best_score is None:
            self.best_score = score
            self.val_metric_best = current_metric_val
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta: # Not improved enough
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience} ({self.monitor}: {current_metric_val:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # Improved
            self.best_score = score
            self.val_metric_best = current_metric_val
            self.save_checkpoint(model)
            self.counter = 0
            if self.verbose:
                print(f'Validation metric improved ({self.monitor}: {current_metric_val:.4f}). Counter reset.')

    def save_checkpoint(self, model: nn.Module):
        if self.verbose:
            print(f'Saving model to {self.path} ({self.monitor}: {self.val_metric_best:.4f})')
        try:
            torch.save(model.state_dict(), self.path)
        except Exception as e:
            print(f"Error saving checkpoint {self.path}: {e}")


class GranularPklDataset(Dataset):
    """Memory-optimized dataset, loads data from PKL files on demand."""
    def __init__(self, identifiers_df: pd.DataFrame, file_mapping: List[Dict[str, Any]], n_features: int):
        self.identifiers_df = identifiers_df
        self.file_mapping = file_mapping # List of dicts: {'path': str, 'offset': int, 'count': int}
        self.n_features = n_features
        self.file_data_cache: Dict[str, List[Tuple[Any, Any, np.ndarray]]] = {} # Cache for loaded file contents

        # Precompute index ranges for faster __getitem__ lookups
        self.file_index_ranges: List[Dict[str, Any]] = []
        # current_offset = 0 # This was in original, but file_mapping already has global offsets
        for fm_entry in self.file_mapping:
            self.file_index_ranges.append({
                'path': fm_entry['path'],
                'start_idx': fm_entry['offset'], # Global start index for this file
                'end_idx': fm_entry['offset'] + fm_entry['count'] # Global end index (exclusive)
            })
            # current_offset += fm_entry['count'] # Not needed if 'offset' is already global start

    def _load_file_data(self, file_path: str) -> List[Tuple[Any, Any, np.ndarray]]:
        if file_path not in self.file_data_cache:
            try:
                with open(file_path, 'rb') as f:
                    self.file_data_cache[file_path] = pickle.load(f)
            except Exception as e:
                print(f"Error loading PKL file {file_path}: {e}")
                # Potentially return empty or raise, depending on desired handling
                return [] # Or raise an error to stop if a file is critical
        return self.file_data_cache[file_path]

    def __len__(self) -> int:
        return len(self.identifiers_df)

    def __getitem__(self, global_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_file_path: Optional[str] = None
        local_idx_in_file: int = -1

        for fr_entry in self.file_index_ranges:
            if fr_entry['start_idx'] <= global_idx < fr_entry['end_idx']:
                target_file_path = fr_entry['path']
                local_idx_in_file = global_idx - fr_entry['start_idx']
                break
        
        if target_file_path is None:
            raise IndexError(f"Global index {global_idx} out of bounds or file mapping error.")

        data_chunk = self._load_file_data(target_file_path)
        if not data_chunk or local_idx_in_file >= len(data_chunk):
             raise IndexError(f"Local index {local_idx_in_file} out of bounds for file {target_file_path} (chunk len {len(data_chunk)}). Global idx: {global_idx}")
        
        # Assuming data_chunk[local_idx_in_file] is (stay_id, pred_hour, feat_array)
        _, _, feat_array = data_chunk[local_idx_in_file]
        
        # Adjust features: map original padding (-1) to PADDING_IDX_EMBEDDING (0),
        # and shift actual bin indices (0 to N_BINS-1) to (1 to N_BINS).
        # This is crucial for nn.Embedding(padding_idx=0).
        x_adjusted = feat_array.astype(np.int64) + 1 
        y_label = self.identifiers_df.iloc[global_idx]['y']
        
        return torch.from_numpy(x_adjusted).long(), torch.tensor(y_label, dtype=torch.float32)


def vectorized_pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]],
                           max_len: int, n_features: int,
                           padding_idx_embedding: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads sequences in a batch and creates a mask."""
    if not batch:
        return (
            torch.zeros((0, max_len, n_features), dtype=torch.long), # padded_seqs
            torch.zeros((0, max_len), dtype=torch.bool),             # mask
            torch.zeros((0,), dtype=torch.float32)                   # labels
        )

    sequences, labels_list = zip(*batch)
    actual_lengths = [min(seq.shape[0], max_len) for seq in sequences] # seq.shape[0] is original length

    padded_seqs = torch.full((len(batch), max_len, n_features),
                             padding_idx_embedding, dtype=torch.long)
    # Mask: True for valid (non-padded) tokens, False for padded tokens
    mask_valid_tokens = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, (seq, length) in enumerate(zip(sequences, actual_lengths)):
        if length > 0: # Ensure there's something to copy
            padded_seqs[i, :length, :] = seq[:length, :] # Correctly slice features
            mask_valid_tokens[i, :length] = True

    return padded_seqs, mask_valid_tokens, torch.stack(labels_list)


# ---------------------- Model Architectures ----------------------

class TransformerClassifierGranular(nn.Module):
    """Transformer-based classifier for granular, binned features."""
    def __init__(self, n_features: int, n_bins_plus_padding: int, embedding_dim: int,
                 d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float = 0.1, max_len: int = MAX_LEN):
        super().__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        # Embedding layer for binned features (input values are bin indices)
        # n_bins_plus_padding = N_BINS + 1 (since 0 is padding, 1 to N_BINS are actual bins)
        self.embedding = nn.Embedding(n_bins_plus_padding, embedding_dim, padding_idx=PADDING_IDX_EMBEDDING)
        
        # Project concatenated feature embeddings to d_model
        self.feature_projection = nn.Linear(embedding_dim * n_features, d_model)
        
        self.norm_input = nn.LayerNorm(d_model) # LayerNorm after projection and scaling
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Attention pooling over transformer_encoder's output sequence
        self.attention_pooling = nn.Sequential(
            nn.Linear(d_model, d_model // 2), # Reduce dimensionality
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.fc_out = nn.Linear(d_model, 1) # Final classification layer
        
    def forward(self, x: torch.Tensor, src_padding_mask_for_transformer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features], containing bin indices.
            src_padding_mask_for_transformer: Boolean tensor, shape [batch_size, seq_len].
                                               True for tokens that SHOULD BE IGNORED (padded).
        """
        batch_size, seq_len, _ = x.shape # x is [B, S, F]
        
        x_embedded = self.embedding(x)  # [B, S, F, E_dim]
        # Concatenate embeddings of all features for each time step
        x_reshaped = x_embedded.view(batch_size, seq_len, self.n_features * self.embedding_dim) # [B, S, F*E_dim]
        
        x_projected = self.feature_projection(x_reshaped) # [B, S, D_model]
        
        # Scaling factor, common in Transformers
        x_projected_scaled = x_projected * math.sqrt(float(self.d_model))
        x_with_pos = self.pos_encoder(x_projected_scaled) # Add positional encoding
        x_normed = self.norm_input(x_with_pos) # Apply LayerNorm

        # Transformer encoder
        # src_key_padding_mask: if a BoolTensor is provided, positions with True are ignored
        transformer_output = self.transformer_encoder(x_normed, src_key_padding_mask=src_padding_mask_for_transformer) # [B, S, D_model]
        
        # Attention Pooling
        # attn_weights: [B, S, 1]
        attn_weights = self.attention_pooling(transformer_output)
        
        # Mask attention weights for padded tokens before softmax
        # src_padding_mask_for_transformer is [B, S], True for padded. Unsqueeze for broadcasting.
        if src_padding_mask_for_transformer is not None:
            attn_weights = attn_weights.masked_fill(src_padding_mask_for_transformer.unsqueeze(-1), -float('inf'))
        
        attn_weights_softmax = F.softmax(attn_weights, dim=1) # Softmax over sequence dimension
        
        # Weighted sum (context vector)
        # transformer_output: [B, S, D_model], attn_weights_softmax: [B, S, 1]
        context_vector = torch.sum(transformer_output * attn_weights_softmax, dim=1) # [B, D_model]
        
        output_logits = self.fc_out(context_vector) # [B, 1]
        return output_logits.squeeze(-1) # [B]

# ---------------------- Helper Function: Get Predictions and Labels ----------------------
def get_predictions_and_labels(model: nn.Module, loader: DataLoader,
                               device: torch.device, is_transformer: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true_labels: List[float] = []
    all_pred_probs: List[float] = []
    
    print(f"  Collecting predictions for model {model.__class__.__name__}...")
    with torch.no_grad():
        for x_batch, mask_batch_valid_tokens, y_true_batch in tqdm(loader, desc="  Batch Progress", leave=False, dynamic_ncols=True):
            x_batch_dev = x_batch.to(device)
            mask_batch_valid_tokens_dev = mask_batch_valid_tokens.to(device) # True for valid tokens
            
            with autocast(enabled=ENABLE_AMP):
                if is_transformer:
                    # Transformer expects True for PADDED tokens
                    padding_mask_for_transformer = ~mask_batch_valid_tokens_dev
                    outputs = model(x_batch_dev, padding_mask_for_transformer)
                else: # Placeholder for other model types if any
                    outputs = model(x_batch_dev, mask_batch_valid_tokens_dev) # Assume other models use valid_mask
            
            pred_probs = torch.sigmoid(outputs).cpu().numpy()
            all_true_labels.extend(y_true_batch.numpy())
            all_pred_probs.extend(pred_probs)
            
    return np.array(all_true_labels), np.array(all_pred_probs)


# ---------------------- Training and Validation Functions ----------------------

def _calculate_epoch_metrics(labels_np: np.ndarray, preds_probs_np: np.ndarray,
                             preds_binary_np: np.ndarray, epoch_loss: float,
                             epoch_num: int, stage: str) -> Dict[str, float]:
    metrics = {'loss': epoch_loss, 'auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0}
    num_unique_labels = len(np.unique(labels_np))

    if num_unique_labels > 1:
        try:
            metrics['auc'] = roc_auc_score(labels_np, preds_probs_np)
            metrics['f1'] = f1_score(labels_np, preds_binary_np, zero_division=0)
            metrics['precision'] = precision_score(labels_np, preds_binary_np, zero_division=0)
            metrics['recall'] = recall_score(labels_np, preds_binary_np, zero_division=0)
            
            # For specificity, we need TN, FP from confusion matrix
            cm_labels_to_use = [0, 1] if (0 in labels_np and 1 in labels_np) else np.unique(labels_np).tolist()
            if len(cm_labels_to_use) == 2 : # Ensure it's binary for ravel()
                tn, fp, fn, tp = confusion_matrix(labels_np, preds_binary_np, labels=cm_labels_to_use).ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else: # Should not happen if num_unique_labels > 1 and handled above, but as a safeguard
                print(f"Warning: Epoch {epoch_num + 1} ({stage}) - Confusion matrix not 2x2 for specificity. Labels: {cm_labels_to_use}")
        except ValueError as e:
            print(f"Warning: Epoch {epoch_num + 1} ({stage}) - Error calculating metrics: {e}. Label data: {np.unique(labels_np)}")
    else:
        print(f"Warning: Epoch {epoch_num + 1} ({stage}) - Insufficient label variety ({np.unique(labels_np)}), some metrics may be skipped or 0.")
    return metrics

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: optim.Optimizer, device: torch.device, scaler: GradScaler,
                    is_transformer: bool, epoch_num: int) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_epoch_labels: List[float] = []
    all_epoch_pred_probs: List[float] = []
    
    progress_bar = tqdm(loader, desc=f'Training Epoch {epoch_num + 1}', dynamic_ncols=True, leave=False)
    for x_batch, mask_batch_valid_tokens, y_batch in progress_bar:
        x_batch = x_batch.to(device)
        mask_batch_valid_tokens = mask_batch_valid_tokens.to(device) # True for valid tokens
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=ENABLE_AMP):
            if is_transformer:
                # Transformer expects True for PADDED tokens
                padding_mask_for_transformer = ~mask_batch_valid_tokens
                outputs = model(x_batch, padding_mask_for_transformer)
            else: # Placeholder for other model types
                outputs = model(x_batch, mask_batch_valid_tokens)
            loss = criterion(outputs, y_batch)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        all_epoch_labels.extend(y_batch.cpu().numpy())
        all_epoch_pred_probs.extend(torch.sigmoid(outputs).cpu().detach().numpy())
        
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
    avg_epoch_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    labels_np = np.array(all_epoch_labels)
    preds_probs_np = np.array(all_epoch_pred_probs)
    preds_binary_np = (preds_probs_np > 0.5).astype(int)

    return _calculate_epoch_metrics(labels_np, preds_probs_np, preds_binary_np, avg_epoch_loss, epoch_num, "Train")


def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device, is_transformer: bool, epoch_num: int) -> Dict[str, float]:
    model.eval()
    total_val_loss = 0.0
    all_val_labels: List[float] = []
    all_val_pred_probs: List[float] = []

    progress_bar = tqdm(loader, desc=f'Validating Epoch {epoch_num + 1}', dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for x_batch, mask_batch_valid_tokens, y_batch in progress_bar:
            x_batch = x_batch.to(device)
            mask_batch_valid_tokens = mask_batch_valid_tokens.to(device) # True for valid tokens
            y_batch = y_batch.to(device)
            
            with autocast(enabled=ENABLE_AMP):
                if is_transformer:
                    # Transformer expects True for PADDED tokens
                    padding_mask_for_transformer = ~mask_batch_valid_tokens
                    outputs = model(x_batch, padding_mask_for_transformer)
                else: # Placeholder for other model types
                    outputs = model(x_batch, mask_batch_valid_tokens)
                loss = criterion(outputs, y_batch)
            
            total_val_loss += loss.item()
            all_val_labels.extend(y_batch.cpu().numpy())
            all_val_pred_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    avg_val_loss = total_val_loss / len(loader) if len(loader) > 0 else 0.0
    labels_np = np.array(all_val_labels)
    preds_probs_np = np.array(all_val_pred_probs)
    preds_binary_np = (preds_probs_np > 0.5).astype(int)

    return _calculate_epoch_metrics(labels_np, preds_probs_np, preds_binary_np, avg_val_loss, epoch_num, "Validation")


def train_model(model: nn.Module, model_name: str, train_loader: DataLoader, val_loader: DataLoader,
                optimizer: optim.Optimizer, lr_scheduler: Optional[optim.lr_scheduler._LRScheduler],
                criterion: nn.Module, es_patience: int, device: torch.device,
                epochs_to_run: int) -> Dict[str, List[float]]:
    print(f"\n===== Starting Training for {model_name} on {device} =====")
    
    is_transformer_model = isinstance(model, TransformerClassifierGranular)
    scaler = GradScaler(enabled=ENABLE_AMP)
    
    es_checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, f'es_checkpoint_{model_name}.pt')
    early_stopper = EarlyStoppingAndCheckpoint(patience=es_patience, verbose=True, path=es_checkpoint_path,
                                               monitor='val_auc', mode='max', delta=0.0001) # Small delta for AUC
    
    best_model_overall_path = os.path.join(MODEL_OUTPUT_DIR, f'best_{model_name}.pt')
    print(f"Training {model_name} for up to {epochs_to_run} epochs.")
    print(f"Early stopping checkpoint will be saved to: {es_checkpoint_path}")
    print(f"Best overall model will be saved to: {best_model_overall_path}")

    history_keys = ['loss', 'auc', 'f1', 'precision', 'recall', 'specificity']
    training_history: Dict[str, List[float]] = {f'{stage}_{key}': [] for stage in ['train', 'val'] for key in history_keys}
    
    history_log_path = os.path.join(RESULTS_OUTPUT_DIR, f'{model_name.lower()}_training_log.tsv')
    print(f"Training log will be saved to: {history_log_path}")

    with open(history_log_path, 'w') as f_log:
        header_cols = ["Epoch"] + [f"Train_{key.capitalize()}" for key in history_keys] + \
                      [f"Val_{key.capitalize()}" for key in history_keys] + ["LR"]
        f_log.write("\t".join(header_cols) + "\n")

        for epoch in range(epochs_to_run):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n--- Epoch {epoch + 1}/{epochs_to_run} --- LR: {current_lr:.2e} ---")
            
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, is_transformer_model, epoch)
            val_metrics = validate_one_epoch(model, val_loader, criterion, device, is_transformer_model, epoch)
            
            for key in history_keys: training_history[f'train_{key}'].append(train_metrics.get(key, np.nan))
            for key in history_keys: training_history[f'val_{key}'].append(val_metrics.get(key, np.nan))

            print(f"Epoch {epoch + 1} | {model_name} | "
                  f"Train: Loss={train_metrics['loss']:.4f} AUC={train_metrics['auc']:.4f} F1={train_metrics['f1']:.4f} | "
                  f"Val: Loss={val_metrics['loss']:.4f} AUC={val_metrics['auc']:.4f} F1={val_metrics['f1']:.4f}")
            
            log_line_data = [str(epoch + 1)] + \
                            [f"{train_metrics.get(key, np.nan):.4f}" for key in history_keys] + \
                            [f"{val_metrics.get(key, np.nan):.4f}" for key in history_keys] + \
                            [f"{current_lr:.2e}"]
            f_log.write("\t".join(log_line_data) + "\n")
            f_log.flush()

            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_metrics['auc']) # Monitor val_auc for ReduceLROnPlateau
                else:
                    lr_scheduler.step() # For other schedulers like StepLR

            early_stopper(val_metrics['auc'], model) # Monitor val_auc
            
            # Save to best_model_overall_path if current val_auc is the best seen so far by early_stopper
            if val_metrics['auc'] >= early_stopper.val_metric_best: # Using >= to catch the first epoch too
                 if abs(val_metrics['auc'] - early_stopper.val_metric_best) < 1e-5 or val_metrics['auc'] > early_stopper.val_metric_best : # effectively if it's the new best
                    try:
                        torch.save(model.state_dict(), best_model_overall_path)
                        print(f"Best overall model ({model_name}) updated and saved to {best_model_overall_path} (Val AUC: {val_metrics['auc']:.4f})")
                    except Exception as e:
                        print(f"Error saving best overall model {best_model_overall_path}: {e}")


            if early_stopper.early_stop:
                print(f"{model_name} early stopping at epoch {epoch + 1} as {early_stopper.monitor} "
                      f"did not improve from {early_stopper.val_metric_best:.4f} for {early_stopper.patience} epochs.")
                print(f"Loading best model from early stopping checkpoint: {early_stopper.path}")
                try:
                    model.load_state_dict(torch.load(early_stopper.path, map_location=device))
                except Exception as e:
                    print(f"Error loading model from early stopping checkpoint {early_stopper.path}: {e}")
                break
            
    # After loop, if not early stopped, ensure best model is loaded
    if not early_stopper.early_stop:
        if os.path.exists(best_model_overall_path):
            print(f"Training completed. Loading final best model from {best_model_overall_path} (Val AUC: {early_stopper.val_metric_best:.4f}).")
            try:
                model.load_state_dict(torch.load(best_model_overall_path, map_location=device))
            except Exception as e:
                 print(f"Error loading final best model {best_model_overall_path}: {e}")
        elif epochs_to_run > 0 : # If training ran but no best_model_overall_path (e.g. all val_auc were NaN)
            last_model_path = os.path.join(MODEL_OUTPUT_DIR, f'last_epoch_{model_name}.pt')
            print(f"Warning: Training completed, but no 'best_{model_name}.pt' found (perhaps val_auc was always NaN or saving failed).")
            print(f"The model state is from the last completed epoch. Saving to {last_model_path}")
            try:
                torch.save(model.state_dict(), last_model_path)
            except Exception as e:
                print(f"Error saving last epoch model {last_model_path}: {e}")


    print(f"===== {model_name} Training Finished =====")
    return training_history


def calculate_test_metrics(true_labels: np.ndarray, pred_probs: np.ndarray, model_name: str) -> Dict[str, Union[float, None]]:
    results: Dict[str, Union[float, None]] = {
        'loss': None, 'auc': 0.0, 'auprc': 0.0, 'f1': 0.0,
        'precision': 0.0, 'recall': 0.0, 'specificity': 0.0
    }
    
    if len(true_labels) == 0 or len(pred_probs) == 0:
        print(f"Warning: {model_name} test evaluation - labels or predictions are empty. Cannot calculate metrics.")
        return results

    # Calculate BCE Loss (log loss)
    epsilon = 1e-7 # To prevent log(0)
    pred_probs_clipped = np.clip(pred_probs, epsilon, 1 - epsilon)
    try:
        bce_loss = -np.mean(true_labels * np.log(pred_probs_clipped) + (1 - true_labels) * np.log(1 - pred_probs_clipped))
        results['loss'] = float(bce_loss)
    except Exception as e:
        print(f"Error calculating {model_name} test BCE loss: {e}")

    num_unique_labels = len(np.unique(true_labels))
    if num_unique_labels > 1: 
        try:
            results['auc'] = float(roc_auc_score(true_labels, pred_probs))
            
            precision_curve, recall_curve, _ = precision_recall_curve(true_labels, pred_probs)
            results['auprc'] = float(auc(recall_curve, precision_curve))
            
            pred_binary = (pred_probs > 0.5).astype(int)
            results['f1'] = float(f1_score(true_labels, pred_binary, zero_division=0))
            results['precision'] = float(precision_score(true_labels, pred_binary, zero_division=0))
            results['recall'] = float(recall_score(true_labels, pred_binary, zero_division=0))
            
            cm_labels_to_use = [0, 1] if (0 in true_labels and 1 in true_labels) else np.unique(true_labels).tolist()
            if len(cm_labels_to_use) == 2:
                tn, fp, fn, tp = confusion_matrix(true_labels, pred_binary, labels=cm_labels_to_use).ravel()
                results['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            else:
                results['specificity'] = 0.0 # Or None, if preferred for non-binary cases
        except ValueError as e:
            print(f"Warning: Error calculating {model_name} test metrics (ValueError): {e}. Unique labels: {np.unique(true_labels)}.")
        except Exception as e: # Catch any other unexpected errors
            print(f"Unexpected error calculating {model_name} test metrics: {e}")
            traceback.print_exc()
    else:
        print(f"Warning: {model_name} test evaluation - insufficient label variety ({np.unique(true_labels)}). Some metrics may be undefined or 0.")
    return results


def print_test_metrics(model_name: str, metrics: Dict[str, Union[float, None]]):
    print(f"\n--- {model_name} - Test Set Evaluation Results ---")
    print(f"  Loss (BCE) : {metrics.get('loss', np.nan):.4f}")
    print(f"  AUC        : {metrics.get('auc', 0.0):.4f}")
    print(f"  AUPRC      : {metrics.get('auprc', 0.0):.4f}")
    print(f"  F1 Score   : {metrics.get('f1', 0.0):.4f}")
    print(f"  Precision  : {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall (Sens): {metrics.get('recall', 0.0):.4f}")
    print(f"  Specificity: {metrics.get('specificity', 0.0):.4f}")


# ---------------------- Plotting Functions (All text in English) ----------------------

def plot_training_metrics(history: Dict[str, List[float]], model_name: str, output_dir_plots: str) -> Optional[str]:
    if not history or "train_loss" not in history or not history["train_loss"]:
        print(f"Warning: Insufficient training history for model {model_name}. Cannot plot training metrics.")
        return None

    epochs_ran = len(history['train_loss'])
    if epochs_ran == 0:
        print(f"Warning: Model {model_name} was not trained (0 epochs). Cannot plot training metrics.")
        return None
        
    epoch_range = range(1, epochs_ran + 1)
    metric_details = [
        ('loss', 'Loss Function', 'Loss Value'),
        ('auc', 'Area Under Curve (AUC)', 'AUC Value'),
        ('f1', 'F1 Score', 'F1 Score'),
        ('precision', 'Precision', 'Precision'),
        ('recall', 'Recall (Sensitivity)', 'Recall'),
        ('specificity', 'Specificity', 'Specificity')
    ]

    plt.figure(figsize=(18, 10 if len(metric_details) > 3 else 5)) # Adjust height based on num subplots
    plt.suptitle(f'{model_name} - Training Process Metrics', fontsize=16)

    num_plots = len(metric_details)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    for i, (key, title, ylabel) in enumerate(metric_details):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(epoch_range, history.get(f'train_{key}', [np.nan]*epochs_ran), label=f'Train {key.capitalize()}', marker='o', linestyle='-')
        plt.plot(epoch_range, history.get(f'val_{key}', [np.nan]*epochs_ran), label=f'Validation {key.capitalize()}', marker='x', linestyle='--')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect for suptitle
    plot_path = os.path.join(output_dir_plots, f'{model_name.lower()}_training_metrics.png')
    try:
        plt.savefig(plot_path)
        print(f"{model_name} training metrics plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving training metrics plot for {model_name} to {plot_path}: {e}")
        plot_path = None # Indicate failure
    plt.close()
    return plot_path


def plot_roc_pr_curves(true_labels: np.ndarray, pred_probs: np.ndarray,
                       model_name: str, output_dir_plots: str) -> Dict[str, Optional[str]]:
    plot_paths: Dict[str, Optional[str]] = {'ROC': None, 'PR': None}
    
    if len(true_labels) == 0 or len(pred_probs) == 0 or len(np.unique(true_labels)) < 2:
        print(f"Warning: Insufficient or single-class data for {model_name} on test set. Cannot plot ROC/PR curves.")
        return plot_paths

    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve (Test Set)')
        plt.legend(loc="lower right")
        plt.grid(True)
        roc_plot_path = os.path.join(output_dir_plots, f'{model_name.lower()}_roc_curve_test.png')
        plt.savefig(roc_plot_path)
        plt.close()
        plot_paths['ROC'] = roc_plot_path
        print(f"{model_name} ROC curve plot saved to: {roc_plot_path}")
    except Exception as e:
        print(f"Error plotting ROC curve for {model_name}: {e}")
        traceback.print_exc()

    # Precision-Recall Curve
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(true_labels, pred_probs)
        # Note: AUC for PR curve can be calculated with auc(recall_vals, precision_vals)
        # but for average precision, scikit-learn has average_precision_score
        pr_auc_val = auc(recall_vals, precision_vals) # Using auc(recall, precision) as per original
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR Curve (AUPRC = {pr_auc_val:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'{model_name} - Precision-Recall Curve (Test Set)')
        plt.legend(loc="lower left") # Or 'best'
        plt.grid(True)
        pr_plot_path = os.path.join(output_dir_plots, f'{model_name.lower()}_pr_curve_test.png')
        plt.savefig(pr_plot_path)
        plt.close()
        plot_paths['PR'] = pr_plot_path
        print(f"{model_name} PR curve plot saved to: {pr_plot_path}")
    except Exception as e:
        print(f"Error plotting PR curve for {model_name}: {e}")
        traceback.print_exc()
        
    return plot_paths


def plot_confusion_matrix_heatmap(true_labels: np.ndarray, pred_probs: np.ndarray,
                                  model_name: str, output_dir_plots: str,
                                  threshold: float = 0.5) -> Optional[str]:
    if len(true_labels) == 0 or len(pred_probs) == 0:
        print(f"Warning: Insufficient data for {model_name} on test set. Cannot plot confusion matrix.")
        return None

    pred_binary = (np.array(pred_probs) > threshold).astype(int)
    
    try:
        cm_labels_to_use = [0,1] if (0 in true_labels and 1 in true_labels) else sorted(list(np.unique(true_labels)))
        if len(cm_labels_to_use) < 2 :
            print(f"Warning: Only one class present in true labels for {model_name} (labels: {cm_labels_to_use}). Cannot plot meaningful confusion matrix.")
            return None
            
        cm = confusion_matrix(true_labels, pred_binary, labels=cm_labels_to_use)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=[f'Predicted {l}' for l in cm_labels_to_use], 
                    yticklabels=[f'Actual {l}' for l in cm_labels_to_use])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{model_name} - Confusion Matrix (Test Set, Threshold={threshold:.2f})')
        cm_plot_path = os.path.join(output_dir_plots, f'{model_name.lower()}_confusion_matrix_test.png')
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"{model_name} confusion matrix plot saved to: {cm_plot_path}")
        return cm_plot_path
    except Exception as e:
        print(f"Error plotting confusion matrix for {model_name}: {e}")
        traceback.print_exc()
        return None

# ---------------------- Main Program --------------------------
def main():
    """Main execution function for Transformer model training and evaluation."""
    start_time_main = time.time()
    MODEL_NAME_CURRENT = 'TransformerGranular' # More descriptive name

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    if ENABLE_AMP:
        print("Automatic Mixed Precision (AMP) is ENABLED.")
    else:
        print("Automatic Mixed Precision (AMP) is DISABLED.")

    print("\n[Phase 1/5] Loading outcomes and mapping PKL files...")
    try:
        outcomes_df = pd.read_parquet(OUTCOMES_FILE)
        # Create a unique key for mapping (stay_id, prediction_hour) to outcome
        outcomes_map = outcomes_df.set_index(['stay_id', 'prediction_hour'])['outcome_death_next_24h'].to_dict()
        print(f"  Loaded {len(outcomes_df)} outcome entries from {OUTCOMES_FILE}")
    except FileNotFoundError:
        print(f"Critical Error: Outcomes file {OUTCOMES_FILE} not found.")
        return
    except Exception as e:
        print(f"Critical Error: Failed to load or parse outcomes file {OUTCOMES_FILE}: {e}")
        return

    all_pkl_files = sorted(glob.glob(os.path.join(TEMP_GRANULAR_DIR, f'*{COHORT_KEY}*.pkl')))
    if not all_pkl_files:
        print(f"Critical Error: No PKL files found in {TEMP_GRANULAR_DIR} matching pattern '*{COHORT_KEY}*.pkl'.")
        print(f"  Please check path and filename pattern. Searched in: {os.path.join(TEMP_GRANULAR_DIR, f'*{COHORT_KEY}*.pkl')}")
        return
    print(f"  Found {len(all_pkl_files)} PKL files to process.")

    file_mapping_info: List[Dict[str, Any]] = []
    current_global_idx = 0
    print("  Building file map by scanning PKL files for sample counts...")
    for pkl_file_path in tqdm(all_pkl_files, desc="  Scanning PKLs", unit="file"):
        try:
            with open(pkl_file_path, 'rb') as f_pkl:
                # Efficiently get length without loading full data if possible,
                # but pickle requires loading to know the length of the list.
                data_in_file = pickle.load(f_pkl)
                num_samples_in_file = len(data_in_file)
                del data_in_file # Free memory
            if num_samples_in_file > 0:
                file_mapping_info.append({'path': pkl_file_path, 'offset': current_global_idx, 'count': num_samples_in_file})
                current_global_idx += num_samples_in_file
            else:
                print(f"Warning: PKL file {pkl_file_path} is empty or contains no samples.")
        except Exception as e:
            print(f"Warning: Failed to process/count samples in file {pkl_file_path}: {e}")
            continue 
    
    if not file_mapping_info:
        print("Critical Error: Failed to build file map. No usable data files found or all files were empty/corrupt.")
        return
    print(f"  File map built. Total potential samples from file map: {current_global_idx}")

    print("  Extracting identifiers (stay_id, prediction_hour) from PKL files...")
    all_identifiers_tuples: List[Tuple[Any, Any]] = []
    
    def load_ids_from_file(file_info_dict: Dict[str, Any]) -> List[Tuple[Any, Any]]:
        ids_in_file: List[Tuple[Any, Any]] = []
        try:
            with open(file_info_dict['path'], 'rb') as f:
                # Data structure: List of (stay_id, prediction_hour, feature_array)
                data_chunk = pickle.load(f) 
            for sample_tuple in data_chunk:
                ids_in_file.append((sample_tuple[0], sample_tuple[1])) # (stay_id, prediction_hour)
        except Exception as e:
            print(f"Warning: ThreadPool: Failed to load identifiers from {file_info_dict['path']}: {e}")
        return ids_in_file

    # Use ThreadPoolExecutor for I/O-bound task of reading PKL file headers/metadata
    # Adjust max_workers based on your system's I/O capacity and number of files
    num_io_workers = min(max(1, NUM_WORKERS if NUM_WORKERS > 0 else os.cpu_count() // 2 or 1), 8) # Cap at 8
    print(f"  Using {num_io_workers} workers for identifier extraction.")
    with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
        # executor.map preserves order, which is good if file_mapping_info is sorted
        results_from_threads = list(tqdm(executor.map(load_ids_from_file, file_mapping_info),
                                         total=len(file_mapping_info), desc="  Extracting IDs", unit="file"))
    
    for id_list_from_file in results_from_threads:
        all_identifiers_tuples.extend(id_list_from_file)

    if not all_identifiers_tuples:
        print("Critical Error: Failed to extract any sample identifiers from PKL files. Dataset will be empty.")
        return
    
    print(f"  Successfully extracted {len(all_identifiers_tuples)} identifiers.")
    identifiers_df = pd.DataFrame(all_identifiers_tuples, columns=['stay_id', 'prediction_hour'])
    
    # Merge with outcomes
    print("  Mapping identifiers to outcomes...")
    identifiers_df['y'] = identifiers_df.apply(
        lambda row: outcomes_map.get((row['stay_id'], row['prediction_hour']), np.nan), # Use np.nan for missing
        axis=1
    )
    
    # Handle cases where outcome might be missing
    initial_len = len(identifiers_df)
    identifiers_df.dropna(subset=['y'], inplace=True)
    if len(identifiers_df) < initial_len:
        print(f"  Warning: Dropped {initial_len - len(identifiers_df)} samples due to missing outcomes.")
    
    identifiers_df['y'] = identifiers_df['y'].astype(int) # Ensure 'y' is integer type for classification

    print(f"  Total samples with valid outcomes: {len(identifiers_df)}")
    if len(identifiers_df) == 0:
        print("Critical Error: No samples remaining after outcome mapping. Cannot proceed.")
        return

    pos_weight_value: Optional[float] = None
    if 'y' in identifiers_df.columns:
        print("  Data label distribution:")
        label_counts = identifiers_df['y'].value_counts(normalize=True)
        print(label_counts)
        if len(label_counts) == 2 and 0 in label_counts.index and 1 in label_counts.index:
            class_counts_raw = identifiers_df['y'].value_counts()
            pos_weight_value = class_counts_raw[0] / class_counts_raw[1]
            print(f"  Calculated pos_weight for BCEWithLogitsLoss: {pos_weight_value:.3f} (count_neg / count_pos)")
        else:
            print("  Warning: Could not calculate pos_weight. Dataset is not binary or one class is missing.")
            print(f"  Unique labels found: {identifiers_df['y'].unique()}")
    else: # Should not happen given prior checks
        print("Critical Error: 'y' column not in identifiers_df. This should not happen.")
        return

    print("\n[Phase 2/5] Creating datasets and DataLoaders...")
    indices = np.arange(len(identifiers_df))
    
    # Stratification only works if there are at least 2 samples per class
    # and at least 2 classes.
    can_stratify = 'y' in identifiers_df.columns and identifiers_df['y'].nunique() > 1
    stratify_labels = identifiers_df['y'].iloc[indices] if can_stratify else None
    if not can_stratify:
        print("  Warning: Cannot stratify splits (single class or no 'y' column). Using random split.")

    train_val_indices, test_indices = train_test_split(
        indices, test_size=TEST_SPLIT, random_state=SEED, stratify=stratify_labels
    )
    
    # Adjust validation split size for the remaining (train+val) data
    # Stratify for train/val split as well
    stratify_labels_train_val = identifiers_df['y'].iloc[train_val_indices] if can_stratify else None
    # Ensure val_split_adjusted is valid and doesn't cause issues if 1-TEST_SPLIT is very small or zero
    denominator_val_split = 1.0 - TEST_SPLIT
    if denominator_val_split <= 1e-6: # effectively zero or too small
        print(f"Warning: TEST_SPLIT ({TEST_SPLIT}) is too close to 1.0. Validation set might be empty or very small.")
        val_split_adjusted = 0.0 # No validation set if train+val pool is empty
    else:
        val_split_adjusted = VAL_SPLIT / denominator_val_split

    if val_split_adjusted >= 1.0 or val_split_adjusted <= 0.0: # If VAL_SPLIT is too large or TEST_SPLIT makes it invalid
        print(f"Warning: Adjusted validation split ({val_split_adjusted:.3f}) is not valid (must be 0 < split < 1).")
        # Decide on fallback: e.g., fixed small validation set or error
        # For now, if train_val_indices is small, split might fail.
        # train_test_split handles small sample sizes by potentially giving empty sets.
        if len(train_val_indices) < 2: # Need at least 2 samples to split
             train_indices = train_val_indices
             val_indices = np.array([], dtype=int)
             print("  Train+Val set too small for further splitting. Assigning all to train, validation set will be empty.")
        elif val_split_adjusted <=0:
            train_indices = train_val_indices
            val_indices = np.array([], dtype=int)
            print("  Adjusted validation split is <=0. Validation set will be empty.")
        elif val_split_adjusted >=1:
            train_indices = np.array([], dtype=int)
            val_indices = train_val_indices
            print("  Adjusted validation split is >=1. Training set will be empty (all to validation).")
        else:
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=val_split_adjusted, random_state=SEED, stratify=stratify_labels_train_val
            )
    else: # Normal case
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_split_adjusted, random_state=SEED, stratify=stratify_labels_train_val
        )


    print(f"  Training set samples: {len(train_indices)}")
    print(f"  Validation set samples: {len(val_indices)}")
    print(f"  Test set samples: {len(test_indices)}")

    if len(train_indices) == 0:
        print("Critical Error: Training set is empty. Cannot proceed.")
        return
    if len(val_indices) == 0:
        print("Warning: Validation set is empty. Early stopping and best model selection based on validation will not work.")
        # Consider implications: if ES_PATIENCE > 0, it might run for all epochs or stop if val_auc is NaN

    full_dataset = GranularPklDataset(identifiers_df, file_mapping_info, N_FEATURES_GRANULAR)

    # Partial function for collate_fn
    collate_fn_custom = functools.partial(vectorized_pad_collate, 
                                          max_len=MAX_LEN,
                                          n_features=N_FEATURES_GRANULAR, 
                                          padding_idx_embedding=PADDING_IDX_EMBEDDING)

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_MODEL,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_fn_custom, drop_last=False) # drop_last=False is usually fine for training
    
    # Use larger batch size for validation/testing if memory allows, as no gradients are computed
    eval_batch_size = BATCH_SIZE_MODEL * 2
    
    val_loader = DataLoader(full_dataset, batch_size=eval_batch_size, 
                            sampler=SequentialSampler(val_indices) if len(val_indices) > 0 else None, # Sampler must not be None if dataset not empty
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            collate_fn=collate_fn_custom) if len(val_indices) > 0 else None
    
    test_loader = DataLoader(full_dataset, batch_size=eval_batch_size,
                             sampler=SequentialSampler(test_indices) if len(test_indices) > 0 else None,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             collate_fn=collate_fn_custom) if len(test_indices) > 0 else None
    
    # Free up memory
    del outcomes_df, outcomes_map, identifiers_df, all_identifiers_tuples, file_mapping_info, results_from_threads
    gc.collect()
    print("  DataLoaders created. Cleared intermediate data structures from memory.")


    print(f"\n[Phase 3/5] Initializing {MODEL_NAME_CURRENT} model, optimizer, and loss function...")
    # Number of embeddings = Number of bins + 1 (for padding_idx=0)
    num_embeddings_for_layer = N_BINS + 1 

    model = TransformerClassifierGranular(
        n_features=N_FEATURES_GRANULAR,
        n_bins_plus_padding=num_embeddings_for_layer,
        embedding_dim=EMBEDDING_DIM,
        d_model=D_MODEL_TRANSFORMER,
        nhead=NHEAD_TRANSFORMER,
        num_encoder_layers=NUM_ENCODER_LAYERS_TRANSFORMER,
        dim_feedforward=DIM_FEEDFORWARD_TRANSFORMER,
        dropout=DROPOUT_TRANSFORMER,
        max_len=MAX_LEN
    ).to(DEVICE)
    print(f"  Model: {MODEL_NAME_CURRENT}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer (AdamW is often good for Transformers)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) # Typical LR and WD
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=ES_PATIENCE // 2, verbose=True)

    # Loss function
    # If pos_weight_value is calculated and valid, use it. Otherwise, no pos_weight.
    criterion_pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE) if pos_weight_value is not None and pos_weight_value > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=criterion_pos_weight_tensor)
    if criterion_pos_weight_tensor is not None:
        print(f"  Using BCEWithLogitsLoss with pos_weight: {criterion_pos_weight_tensor.item():.3f}")
    else:
        print("  Using BCEWithLogitsLoss without pos_weight.")


    print(f"\n[Phase 4/5] Training {MODEL_NAME_CURRENT} model...")
    if val_loader is None and ES_PATIENCE > 0:
        print("Warning: Validation loader is not available, but ES_PATIENCE is set. Early stopping will not function correctly.")
        print("Training will run for the full number of epochs or until manually stopped.")
        # Potentially disable early stopping or adjust logic if no val_loader
        effective_es_patience = EPOCHS + 1 # Effectively disable early stopping if no val_loader
    else:
        effective_es_patience = ES_PATIENCE

    training_history = train_model(model, MODEL_NAME_CURRENT, train_loader, val_loader,
                                   optimizer, lr_scheduler, criterion,
                                   effective_es_patience, DEVICE, EPOCHS)
    
    plot_training_metrics(training_history, MODEL_NAME_CURRENT, RESULTS_OUTPUT_DIR)
    
    # Save final model explicitly (might be redundant if best model logic covers it)
    final_model_path = os.path.join(MODEL_OUTPUT_DIR, f'final_{MODEL_NAME_CURRENT}.pt')
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model state saved to: {final_model_path}")
    except Exception as e:
        print(f"Error saving final model state to {final_model_path}: {e}")


    print(f"\n[Phase 5/5] Evaluating {MODEL_NAME_CURRENT} on the test set...")
    if test_loader:
        true_labels_test, pred_probs_test = get_predictions_and_labels(
            model, test_loader, DEVICE, isinstance(model, TransformerClassifierGranular)
        )
        
        test_metrics = calculate_test_metrics(true_labels_test, pred_probs_test, MODEL_NAME_CURRENT)
        print_test_metrics(MODEL_NAME_CURRENT, test_metrics)
        
        # Save test metrics to a JSON file
        test_metrics_path = os.path.join(RESULTS_OUTPUT_DIR, f'{MODEL_NAME_CURRENT.lower()}_test_metrics.json')
        try:
            with open(test_metrics_path, 'w') as f_json:
                json.dump({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in test_metrics.items()}, f_json, indent=4)
            print(f"Test metrics saved to: {test_metrics_path}")
        except Exception as e:
            print(f"Error saving test metrics to {test_metrics_path}: {e}")

        # Plot ROC, PR, Confusion Matrix for test set
        plot_roc_pr_curves(true_labels_test, pred_probs_test, MODEL_NAME_CURRENT, RESULTS_OUTPUT_DIR)
        plot_confusion_matrix_heatmap(true_labels_test, pred_probs_test, MODEL_NAME_CURRENT, RESULTS_OUTPUT_DIR)
    else:
        print("  Test loader not available. Skipping test set evaluation.")

    # Clean up CUDA cache if applicable
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        print("  Cleared CUDA cache.")
    
    total_time_main = time.time() - start_time_main
    print(f"\n--- Script finished in {total_time_main // 60:.0f}m {total_time_main % 60:.0f}s ---")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An unhandled critical error occurred in main execution: {e}")
        traceback.print_exc()
        # Consider more robust error reporting here for production systems
    finally:
        print("Execution attempt finished.")