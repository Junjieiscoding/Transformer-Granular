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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, roc_curve, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") # Use Agg backend to avoid GUI issues
from tqdm import tqdm
import seaborn as sns # For confusion matrix heatmap

# --- Suppress warnings ---
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True  # Enable CuDNN benchmarking for fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs

# ---------------------- Global Configurations -----------------------------
OUTPUT_DIR = './output_dynamic' # Output directory
COHORT_KEY = 'aki' # Cohort identifier

# File paths
TEMP_GRANULAR_DIR = os.path.join(OUTPUT_DIR, 'step4b_temp_granular') # Temporary granular data directory
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'step5_models_pytorch_{COHORT_KEY}') # Model output directory
RESULTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'step5_results_pytorch_{COHORT_KEY}') # Results output directory
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True) # Create directories if they don't exist
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

BIN_EDGES_FILE = os.path.join(OUTPUT_DIR, f'step4ab_bin_edges_{COHORT_KEY}_5bins.pkl') # Bin edges file
OUTCOMES_FILE = os.path.join(OUTPUT_DIR, f'step3_dynamic_outcomes_{COHORT_KEY}.parquet') # Outcomes file

# --- Data Dimensions ---
try:
    with open(BIN_EDGES_FILE, 'rb') as f:
        bin_edges = pickle.load(f)
    N_FEATURES_GRANULAR = len(bin_edges) # Number of features
    N_BINS = len(next(iter(bin_edges.values()))) - 1 # Number of bins per feature (assuming all features have the same number of bins)
except FileNotFoundError:
    print(f"Error: Bin edges file {BIN_EDGES_FILE} not found. Please ensure the file exists.")
    exit(1)
except Exception as e:
    print(f"Error: Failed to load bin edges file {BIN_EDGES_FILE}: {e}")
    exit(1)


MAX_LEN = 240 # Maximum sequence length
PADDING_VALUE_INPUT = -1 # Original padding value in input data (will be mapped to PADDING_IDX_EMBEDDING)
PADDING_IDX_EMBEDDING = 0 # Padding index used by the Embedding layer

# --- Training Parameters ---
TEST_SPLIT = 0.15 # Test set proportion
VAL_SPLIT = 0.15  # Validation set proportion (relative to train+val set)
SEED = 42 # Random seed
BATCH_SIZE_MODEL = 64 # Model training batch size
EPOCHS = 50 # Number of training epochs (changed from 1 to 50 as requested)
ES_PATIENCE = 10 # Early stopping patience

# --- GPU Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device configuration
ENABLE_AMP = True if DEVICE.type == 'cuda' else False # Automatic Mixed Precision (enabled only on CUDA)
NUM_WORKERS = 0 # DataLoader number of workers (0 is often more stable on Windows)
PIN_MEMORY = True if DEVICE.type == 'cuda' else False # DataLoader pin_memory

# --- Model Parameters ---
EMBEDDING_DIM = 32 # Embedding dimension
LSTM_HIDDEN_DIM = 64 # LSTM hidden layer dimension
D_MODEL_TRANSFORMER = 128 # Transformer d_model


# ---------------------- Model Components ----------------------

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=MAX_LEN): # Changed max_len to use global MAX_LEN
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # Positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Position indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # Frequency decay term
        pe[:, 0::2] = torch.sin(position * div_term) # Use sin for even dimensions
        pe[:, 1::2] = torch.cos(position * div_term) # Use cos for odd dimensions
        pe = pe.unsqueeze(0).transpose(0, 1) # Adjust shape to match input (seq_len, batch, dim)
        self.register_buffer('pe', pe) # Register as buffer, not updated by gradients

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] if not batch_first else [batch_size, seq_len, embedding_dim]
        """
        # If batch_first is True for the main module, x is [batch, seq_len, dim]
        # PE is [max_len, 1, dim] or similar, needs to be broadcastable
        # Assuming x is [batch_size, seq_len, embedding_dim] from TransformerEncoderLayer with batch_first=True
        x = x + self.pe[:x.size(1), :].squeeze(1) # self.pe is [max_len, 1, dim], take [:seq_len, :], result [seq_len, dim]
        return self.dropout(x)


class EarlyStoppingAndCheckpoint:
    """Early stopping and model checkpointing"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',
                 monitor='val_loss', mode='min'):
        self.patience = patience # How many epochs to wait for improvement
        self.verbose = verbose # Whether to print messages
        self.counter = 0 # Counter
        self.best_score = None # Best score
        self.early_stop = False # Early stop flag
        self.delta = delta # Minimum change to qualify as an improvement
        self.path = path # Model save path
        self.monitor = monitor # Metric to monitor
        self.mode = mode # 'min' for loss, 'max' for AUC/F1
        self.val_metric_best = np.Inf if mode == 'min' else -np.Inf # Best value of the monitored metric

    def __call__(self, current_metric_val, model):
        score = -current_metric_val if self.mode == 'min' else current_metric_val # Adjust score comparison based on mode

        if self.best_score is None: # First epoch
            self.best_score = score
            self.val_metric_best = current_metric_val
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta: # No improvement beyond delta
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.monitor}: {current_metric_val:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else: # Improvement
            self.best_score = score
            self.val_metric_best = current_metric_val
            self.save_checkpoint(model)
            self.counter = 0
            if self.verbose:
                print(f'Validation metric improved ({self.monitor}: {current_metric_val:.4f}). Counter reset.')


    def save_checkpoint(self, model):
        """Save model state_dict"""
        if self.verbose:
            print(f'Saving model to {self.path} ({self.monitor}: {self.val_metric_best:.4f})')
        torch.save(model.state_dict(), self.path)


class GranularPklDataset(Dataset):
    """Memory-optimized granular dataset (loads from PKL files on demand)"""
    def __init__(self, identifiers_df, file_mapping, n_features):
        self.identifiers_df = identifiers_df # DataFrame with stay_id, prediction_hour, y_label
        self.file_mapping = file_mapping # File path and sample index start/end info
        self.n_features = n_features # Number of features
        self.file_data_cache = {} # File data cache to avoid redundant loads

        self.file_index_ranges = []
        current_offset = 0
        for fm_entry in self.file_mapping:
            self.file_index_ranges.append({
                'path': fm_entry['path'],
                'start_idx': fm_entry['offset'], # Global start index
                'end_idx': fm_entry['offset'] + fm_entry['count'] # Global end index (exclusive)
            })
            current_offset += fm_entry['count']

    def _load_file_data(self, file_path):
        """Load single PKL file on demand and cache it"""
        if file_path not in self.file_data_cache:
            with open(file_path, 'rb') as f:
                self.file_data_cache[file_path] = pickle.load(f)
        return self.file_data_cache[file_path]

    def __len__(self):
        return len(self.identifiers_df)

    def __getitem__(self, global_idx):
        """Get a single sample by global index"""
        target_file_path = None
        local_idx_in_file = -1

        for fr_entry in self.file_index_ranges:
            if fr_entry['start_idx'] <= global_idx < fr_entry['end_idx']:
                target_file_path = fr_entry['path']
                local_idx_in_file = global_idx - fr_entry['start_idx']
                break
        
        if target_file_path is None:
            raise IndexError(f"Global index {global_idx} out of bounds or corresponding file not found.")

        data_chunk = self._load_file_data(target_file_path)
        
        _, _, feat_array = data_chunk[local_idx_in_file]

        x_adjusted = feat_array.astype(np.int64) + 1 
        
        y_label = self.identifiers_df.iloc[global_idx]['y']
        
        return torch.from_numpy(x_adjusted).long(), torch.tensor(y_label, dtype=torch.float32)


def vectorized_pad_collate(batch, max_len, n_features, padding_idx_embedding):
    """
    Custom collate function to pad sequences to max_len and create attention masks.
    Args:
        batch: List of samples, each sample is (feature_tensor, label_tensor)
        max_len: Maximum sequence length
        n_features: Number of features
        padding_idx_embedding: Index value for padding (e.g., 0)
    Returns:
        padded_seqs: Padded sequence tensor (batch_size, max_len, n_features)
        mask: Attention mask tensor (batch_size, max_len), True for valid data, False for padding
        labels: Label tensor (batch_size,)
    """
    if not batch: # Handle empty batch
        return (
            torch.zeros((0, max_len, n_features), dtype=torch.long),
            torch.zeros((0, max_len), dtype=torch.bool),
            torch.zeros((0,), dtype=torch.float32)
        )

    sequences, labels_list = zip(*batch) # Unpack features and labels

    actual_lengths = [min(len(seq), max_len) for seq in sequences] # Calculate actual valid length for each sequence

    padded_seqs = torch.full((len(batch), max_len, n_features),
                             padding_idx_embedding, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool) # False represents padding

    for i, (seq, length) in enumerate(zip(sequences, actual_lengths)):
        if length > 0: # Ensure sequence length is greater than 0
            padded_seqs[i, :length] = seq[:length] # Fill sequence data
            mask[i, :length] = True # Set valid part of mask to True

    return padded_seqs, mask, torch.stack(labels_list)


# ---------------------- Model Architectures ----------------------

class LSTMClassifierGranular(nn.Module):
    """LSTM-based classifier for granular features"""
    def __init__(self, n_features, n_bins_plus_padding, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_bins_plus_padding, embedding_dim, padding_idx=PADDING_IDX_EMBEDDING)
        self.lstm = nn.LSTM(embedding_dim * n_features, hidden_dim, batch_first=True, num_layers=1) 
        self.fc = nn.Linear(hidden_dim, 1) 
        
    def forward(self, x, mask=None): # mask is for API consistency with Transformer
        batch_size, seq_len, num_feat = x.shape
        x_embedded = self.embedding(x) 
        x_reshaped = x_embedded.view(batch_size, seq_len, -1) 
        
        lstm_out, (h_n, c_n) = self.lstm(x_reshaped)
        last_hidden_state = lstm_out[:, -1, :] 
        
        output = self.fc(last_hidden_state)
        return output.squeeze(-1)


class TransformerClassifierGranular(nn.Module):
    """Transformer-based classifier for granular features"""
    def __init__(self, n_features, n_bins_plus_padding, embedding_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_bins_plus_padding, embedding_dim, padding_idx=PADDING_IDX_EMBEDDING)
        # Linear layer to project concatenated feature embeddings to Transformer's d_model
        self.feature_projection = nn.Linear(embedding_dim * n_features, d_model) # Keeping this name consistent
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=MAX_LEN) # Positional encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(d_model, 1) # Classification head
        self.d_model = d_model # Store d_model for scaling input to pos_encoder
        
    def forward(self, x, src_padding_mask):
        """
        Args:
            x: Input features (batch_size, seq_len, n_features)
            src_padding_mask: Source sequence padding mask (batch_size, seq_len), True indicates a padded position
        """
        batch_size, seq_len, num_feat = x.shape
        x_embedded = self.embedding(x) 
        x_reshaped = x_embedded.view(batch_size, seq_len, -1)
        
        x_projected = self.feature_projection(x_reshaped) # (batch_size, seq_len, d_model)
        
        # Scale input to positional encoder, a common practice
        x_projected_scaled = x_projected * math.sqrt(self.d_model)
        x_with_pos = self.pos_encoder(x_projected_scaled) # Input is (batch_size, seq_len, d_model)

        # TransformerEncoderLayer expects src_key_padding_mask where True means padded.
        # Our collate_fn's mask is True for valid data, so we need to invert it.
        # This is handled in train/validate/get_predictions_and_labels functions before calling model.
        transformer_output = self.transformer_encoder(x_with_pos, src_key_padding_mask=src_padding_mask) # (batch_size, seq_len, d_model)
        
        pooled_output = transformer_output[:, -1, :] # Take the output of the last time step

        output = self.fc(pooled_output)
        return output.squeeze(-1)

# ---------------------- Helper Function: Get Predictions and Labels ----------------------
def get_predictions_and_labels(model, loader, device, is_transformer):
    """Run model on data loader and collect true labels and predicted probabilities"""
    model.eval() # Set to evaluation mode
    all_true_labels = []
    all_pred_probs = []
    
    print(f"  Collecting predictions for model {model.__class__.__name__}...")
    with torch.no_grad(): # Disable gradient calculations
        for x_batch, mask_batch_valid, y_true_batch in tqdm(loader, desc="  Batch Progress", leave=False, dynamic_ncols=True):
            x_batch_dev, mask_batch_valid_dev = x_batch.to(device), mask_batch_valid.to(device)
            
            with autocast(enabled=ENABLE_AMP): # Automatic mixed precision
                if is_transformer:
                    # TransformerEncoderLayer's src_key_padding_mask needs True for padded positions.
                    # collate_fn's mask (mask_batch_valid) is True for valid data. So, invert it.
                    padding_mask_for_transformer = ~mask_batch_valid_dev
                    outputs = model(x_batch_dev, padding_mask_for_transformer)
                else:
                    outputs = model(x_batch_dev, mask_batch_valid_dev) # LSTM forward also receives mask
            
            pred_probs = torch.sigmoid(outputs).cpu().numpy() # Get probabilities and move to CPU
            all_true_labels.extend(y_true_batch.numpy())
            all_pred_probs.extend(pred_probs)
            
    return np.array(all_true_labels), np.array(all_pred_probs)


# ---------------------- Training and Validation Functions ----------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, is_transformer, epoch_num):
    """Train for one epoch"""
    model.train() # Set to training mode
    total_loss = 0.0
    all_epoch_labels = []
    all_epoch_pred_probs = []
    
    progress_bar = tqdm(loader, desc=f'Training Epoch {epoch_num + 1}', dynamic_ncols=True, leave=False)
    for x_batch, mask_batch_valid, y_batch in progress_bar:
        x_batch, mask_batch_valid, y_batch = x_batch.to(device), mask_batch_valid.to(device), y_batch.to(device)
        
        optimizer.zero_grad() # Clear gradients
        
        with autocast(enabled=ENABLE_AMP): # AMP context
            if is_transformer:
                padding_mask_for_transformer = ~mask_batch_valid # True means padded
                outputs = model(x_batch, padding_mask_for_transformer)
            else:
                outputs = model(x_batch, mask_batch_valid)
            loss = criterion(outputs, y_batch) # Calculate loss
            
        scaler.scale(loss).backward() # Scale loss and backpropagate
        scaler.step(optimizer) # Update optimizer
        scaler.update() # Update scaler
        
        total_loss += loss.item()
        all_epoch_labels.extend(y_batch.cpu().numpy())
        all_epoch_pred_probs.extend(torch.sigmoid(outputs).cpu().detach().numpy())
        
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
    avg_epoch_loss = total_loss / len(loader)
    
    labels_np = np.array(all_epoch_labels)
    preds_probs_np = np.array(all_epoch_pred_probs)
    preds_binary = (preds_probs_np > 0.5).astype(int) # Use 0.5 threshold

    metrics = {'loss': avg_epoch_loss, 'auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0}
    if len(np.unique(labels_np)) > 1: # Ensure at least two classes
        try:
            metrics['auc'] = roc_auc_score(labels_np, preds_probs_np)
            metrics['f1'] = f1_score(labels_np, preds_binary, zero_division=0)
            
            # Ensure labels=[0,1] for confusion_matrix if not all classes are present
            cm_labels = [0, 1] if (0 in labels_np and 1 in labels_np) else np.unique(labels_np)
            if len(cm_labels) == 2 : # only calculate if binary
                tn, fp, fn, tp = confusion_matrix(labels_np, preds_binary, labels=cm_labels).ravel()
                metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else: # if only one class present in batch labels, can't calculate these
                print(f"Warning: Epoch {epoch_num + 1} (Train) - Not enough classes in batch labels to compute detailed confusion matrix metrics.")

        except ValueError as e:
            print(f"Warning: Epoch {epoch_num + 1} (Train) - Error calculating metrics: {e}")
    else:
        print(f"Warning: Epoch {epoch_num + 1} (Train) - Insufficient label variety ({np.unique(labels_np)}), some metrics may be skipped.")
        
    return metrics


def validate_one_epoch(model, loader, criterion, device, is_transformer, epoch_num):
    """Validate model for one epoch and calculate metrics"""
    model.eval() # Set to evaluation mode
    total_val_loss = 0.0
    all_val_labels = []
    all_val_pred_probs = []

    progress_bar = tqdm(loader, desc=f'Validating Epoch {epoch_num + 1}', dynamic_ncols=True, leave=False)
    with torch.no_grad(): # Disable gradient calculations
        for x_batch, mask_batch_valid, y_batch in progress_bar:
            x_batch, mask_batch_valid, y_batch = x_batch.to(device), mask_batch_valid.to(device), y_batch.to(device)
            
            with autocast(enabled=ENABLE_AMP):
                if is_transformer:
                    padding_mask_for_transformer = ~mask_batch_valid
                    outputs = model(x_batch, padding_mask_for_transformer)
                else:
                    outputs = model(x_batch, mask_batch_valid)
                loss = criterion(outputs, y_batch)
            
            total_val_loss += loss.item()
            all_val_labels.extend(y_batch.cpu().numpy())
            all_val_pred_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    avg_val_loss = total_val_loss / len(loader)
    
    labels_np = np.array(all_val_labels)
    preds_probs_np = np.array(all_val_pred_probs)
    preds_binary = (preds_probs_np > 0.5).astype(int)

    metrics = {'loss': avg_val_loss, 'auc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0}
    if len(np.unique(labels_np)) > 1:
        try:
            metrics['auc'] = roc_auc_score(labels_np, preds_probs_np)
            metrics['f1'] = f1_score(labels_np, preds_binary, zero_division=0)
            
            metrics['precision'] = precision_score(labels_np, preds_binary, zero_division=0)
            metrics['recall'] = recall_score(labels_np, preds_binary, zero_division=0) # Sensitivity
            
            cm_labels = [0, 1] if (0 in labels_np and 1 in labels_np) else np.unique(labels_np)
            if len(cm_labels) == 2:
                tn, fp, fn, tp = confusion_matrix(labels_np, preds_binary, labels=cm_labels).ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                 print(f"Warning: Epoch {epoch_num + 1} (Validation) - Not enough classes in batch labels to compute detailed confusion matrix metrics.")
        except ValueError as e:
            print(f"Warning: Epoch {epoch_num + 1} (Validation) - Error calculating metrics: {e}")
    else:
        print(f"Warning: Epoch {epoch_num + 1} (Validation) - Insufficient label variety ({np.unique(labels_np)}), some metrics may be skipped.")
            
    return metrics


def train_model(model, model_name, train_loader, val_loader, optimizer, lr_scheduler, criterion, es_patience, device, epochs_to_run):
    """Complete model training loop"""
    print(f"\n===== Starting Training for {model_name} =====")
    
    is_transformer_model = isinstance(model, TransformerClassifierGranular)
    scaler = GradScaler(enabled=ENABLE_AMP) # AMP GradScaler
    
    es_checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, f'es_checkpoint_{model_name}.pt')
    early_stopper = EarlyStoppingAndCheckpoint(patience=es_patience, verbose=True, path=es_checkpoint_path,
                                             monitor='val_auc', mode='max') # Monitor validation AUC
    
    best_model_overall_path = os.path.join(MODEL_OUTPUT_DIR, f'best_{model_name}.pt') # Path for the best model over all epochs
    
    # REMOVED: Logic for loading pre-existing best model to force retraining from scratch.
    print(f"Training {model_name} from scratch for {epochs_to_run} epochs.")

    training_history = {
        'train_loss': [], 'train_auc': [], 'train_f1': [], 'train_precision': [], 'train_recall': [], 'train_specificity': [],
        'val_loss': [], 'val_auc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'val_specificity': []
    }
    
    history_log_path = os.path.join(RESULTS_OUTPUT_DIR, f'{model_name.lower()}_training_log.tsv')
    with open(history_log_path, 'w') as f_log:
        header_cols = [
            "Epoch", "Train_Loss", "Train_AUC", "Train_F1", "Train_Prec", "Train_Rec", "Train_Spec",
            "Val_Loss", "Val_AUC", "Val_F1", "Val_Prec", "Val_Rec", "Val_Spec", "LR"
        ]
        f_log.write("\t".join(header_cols) + "\n")
        print(f"Training log will be saved to: {history_log_path}")

        for epoch in range(epochs_to_run): # Use epochs_to_run
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n--- Epoch {epoch + 1}/{epochs_to_run} --- LR: {current_lr:.2e} ---")
            
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, is_transformer_model, epoch)
            val_metrics = validate_one_epoch(model, val_loader, criterion, device, is_transformer_model, epoch)
            
            for metric_key in train_metrics: training_history[f'train_{metric_key}'].append(train_metrics[metric_key])
            for metric_key in val_metrics: training_history[f'val_{metric_key}'].append(val_metrics[metric_key])

            print(f"Epoch {epoch + 1} | {model_name} | Train: Loss={train_metrics['loss']:.4f} AUC={train_metrics['auc']:.4f} F1={train_metrics['f1']:.4f} | "
                  f"Val: Loss={val_metrics['loss']:.4f} AUC={val_metrics['auc']:.4f} F1={val_metrics['f1']:.4f}")
            
            log_line_data = [
                str(epoch + 1),
                f"{train_metrics['loss']:.4f}", f"{train_metrics['auc']:.4f}", f"{train_metrics['f1']:.4f}",
                f"{train_metrics['precision']:.4f}", f"{train_metrics['recall']:.4f}", f"{train_metrics['specificity']:.4f}",
                f"{val_metrics['loss']:.4f}", f"{val_metrics['auc']:.4f}", f"{val_metrics['f1']:.4f}",
                f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}", f"{val_metrics['specificity']:.4f}",
                f"{current_lr:.2e}"
            ]
            f_log.write("\t".join(log_line_data) + "\n")
            f_log.flush()

            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_metrics['auc'])
            # else: lr_scheduler.step() 

            early_stopper(val_metrics['auc'], model) # Check early stopping based on val_auc
            if early_stopper.early_stop:
                print(f"{model_name} early stopping at epoch {epoch + 1} as {early_stopper.monitor} did not improve for {early_stopper.patience} epochs.")
                print(f"Loading best model from early stopping (Val AUC: {early_stopper.val_metric_best:.4f}) from: {early_stopper.path}")
                model.load_state_dict(torch.load(early_stopper.path)) 
                break
            
            # Save the overall best model based on val_auc
            if val_metrics['auc'] >= early_stopper.val_metric_best : # Use >= to save even if it's the same (e.g. first epoch)
                 torch.save(model.state_dict(), best_model_overall_path)
                 print(f"Best overall model ({model_name}) updated and saved to {best_model_overall_path} (Val AUC: {val_metrics['auc']:.4f})")

    if not early_stopper.early_stop and os.path.exists(best_model_overall_path):
        print(f"Training completed. Loading final best model from {best_model_overall_path}.")
        model.load_state_dict(torch.load(best_model_overall_path))
    elif not early_stopper.early_stop and not os.path.exists(best_model_overall_path) and epochs_to_run > 0 :
        last_model_path = os.path.join(MODEL_OUTPUT_DIR, f'last_epoch_{model_name}.pt')
        torch.save(model.state_dict(), last_model_path)
        print(f"Warning: Training completed but no explicit 'best_{model_name}.pt' found. Last epoch model saved to {last_model_path}.")

    print(f"===== {model_name} Training Finished =====")
    return training_history


def calculate_test_metrics(true_labels, pred_probs, model_name):
    """Calculate various evaluation metrics for the test set"""
    results = {'auc': 0.0, 'auprc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 'loss': np.nan}
    
    if len(true_labels) == 0 or len(pred_probs) == 0:
        print(f"Warning: {model_name} test evaluation - labels or predictions are empty. Cannot calculate metrics.")
        return results

    epsilon = 1e-7 
    pred_probs_clipped = np.clip(pred_probs, epsilon, 1 - epsilon)
    true_labels_np = np.array(true_labels) # Ensure numpy array
    try:
        # Calculate BCE loss from probabilities
        bce_loss = -np.mean(true_labels_np * np.log(pred_probs_clipped) + (1 - true_labels_np) * np.log(1 - pred_probs_clipped))
        results['loss'] = float(bce_loss) # Convert to Python float for JSON
    except Exception as e:
        print(f"Error calculating {model_name} test loss: {e}")

    if len(np.unique(true_labels_np)) > 1: 
        try:
            results['auc'] = float(roc_auc_score(true_labels_np, pred_probs))
            
            precision_curve, recall_curve, _ = precision_recall_curve(true_labels_np, pred_probs)
            results['auprc'] = float(auc(recall_curve, precision_curve))
            
            pred_binary = (pred_probs > 0.5).astype(int)
            results['f1'] = float(f1_score(true_labels_np, pred_binary, zero_division=0))
            results['precision'] = float(precision_score(true_labels_np, pred_binary, zero_division=0))
            results['recall'] = float(recall_score(true_labels_np, pred_binary, zero_division=0))
            
            cm_labels = [0, 1] if (0 in true_labels_np and 1 in true_labels_np) else np.unique(true_labels_np)
            if len(cm_labels) == 2:
                tn, fp, fn, tp = confusion_matrix(true_labels_np, pred_binary, labels=cm_labels).ravel()
                results['specificity'] = float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            else:
                results['specificity'] = 0.0 # Default if not binary case for CM
        except ValueError as e:
            print(f"Warning: Error calculating {model_name} test metrics: {e}. Label data might be problematic.")
        except Exception as e:
            print(f"Unknown error calculating {model_name} test metrics: {e}")
    else:
        print(f"Warning: {model_name} test evaluation - insufficient label variety ({np.unique(true_labels_np)}). Some metrics may be undefined or meaningless.")
        
    return results


def print_test_metrics(model_name, metrics):
    """Print test set evaluation results"""
    print(f"\n--- {model_name} - Test Set Evaluation Results ---")
    print(f"  Loss       : {metrics.get('loss', np.nan):.4f}")
    print(f"  AUC        : {metrics.get('auc', 0.0):.4f}")
    print(f"  AUPRC      : {metrics.get('auprc', 0.0):.4f}")
    print(f"  F1 Score   : {metrics.get('f1', 0.0):.4f}")
    print(f"  Precision  : {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall (Sens): {metrics.get('recall', 0.0):.4f}")
    print(f"  Specificity: {metrics.get('specificity', 0.0):.4f}")


# ---------------------- Plotting Functions (All text in English) ----------------------

def plot_training_metrics(history, model_name, output_dir_plots):
    """Plot loss and key metrics during training"""
    if not history or "train_loss" not in history or not history["train_loss"]:
        print(f"Warning: Insufficient training history for model {model_name}. Cannot plot training metrics.")
        return None

    epochs_ran = len(history['train_loss'])
    if epochs_ran == 0:
        print(f"Warning: Model {model_name} was not trained (epochs_ran=0). Cannot plot training metrics.")
        return None
        
    epoch_range = range(1, epochs_ran + 1)

    plt.figure(figsize=(18, 10))
    plt.suptitle(f'{model_name} - Training Process Metrics', fontsize=16)

    # Loss
    plt.subplot(2, 3, 1)
    plt.plot(epoch_range, history.get('train_loss', [np.nan]*epochs_ran), label='Train Loss', marker='o', linestyle='-')
    plt.plot(epoch_range, history.get('val_loss', [np.nan]*epochs_ran), label='Validation Loss', marker='x', linestyle='--')
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)

    # AUC
    plt.subplot(2, 3, 2)
    plt.plot(epoch_range, history.get('train_auc', [np.nan]*epochs_ran), label='Train AUC', marker='o', linestyle='-')
    plt.plot(epoch_range, history.get('val_auc', [np.nan]*epochs_ran), label='Validation AUC', marker='x', linestyle='--')
    plt.title('Area Under Curve (AUC)')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Value')
    plt.legend()
    plt.grid(True)

    # F1 Score
    plt.subplot(2, 3, 3)
    plt.plot(epoch_range, history.get('train_f1', [np.nan]*epochs_ran), label='Train F1 Score', marker='o', linestyle='-')
    plt.plot(epoch_range, history.get('val_f1', [np.nan]*epochs_ran), label='Validation F1 Score', marker='x', linestyle='--')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Precision
    plt.subplot(2, 3, 4)
    plt.plot(epoch_range, history.get('train_precision', [np.nan]*epochs_ran), label='Train Precision', marker='o', linestyle='-')
    plt.plot(epoch_range, history.get('val_precision', [np.nan]*epochs_ran), label='Validation Precision', marker='x', linestyle='--')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    # Recall
    plt.subplot(2, 3, 5)
    plt.plot(epoch_range, history.get('train_recall', [np.nan]*epochs_ran), label='Train Recall', marker='o', linestyle='-')
    plt.plot(epoch_range, history.get('val_recall', [np.nan]*epochs_ran), label='Validation Recall', marker='x', linestyle='--')
    plt.title('Recall (Sensitivity)')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    # Specificity
    plt.subplot(2, 3, 6)
    plt.plot(epoch_range, history.get('train_specificity', [np.nan]*epochs_ran), label='Train Specificity', marker='o', linestyle='-')
    plt.plot(epoch_range, history.get('val_specificity', [np.nan]*epochs_ran), label='Validation Specificity', marker='x', linestyle='--')
    plt.title('Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    plot_path = os.path.join(output_dir_plots, f'{model_name.lower()}_training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"{model_name} training metrics plot saved to: {plot_path}")
    return plot_path


def plot_roc_pr_curves(true_labels, pred_probs, model_name, output_dir_plots):
    """Plot ROC and PR curves"""
    plot_paths = {}
    
    if len(true_labels) == 0 or len(pred_probs) == 0 or len(np.unique(true_labels)) < 2:
        print(f"Warning: Insufficient or single-class data for {model_name} on test set. Cannot plot ROC/PR curves.")
        return plot_paths

    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc_val:.4f})')
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

        # PR Curve
        precision_vals, recall_vals, _ = precision_recall_curve(true_labels, pred_probs)
        pr_auc_val = auc(recall_vals, precision_vals) 
        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR Curve (Area = {pr_auc_val:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'{model_name} - Precision-Recall Curve (Test Set)')
        plt.legend(loc="lower left")
        plt.grid(True)
        pr_plot_path = os.path.join(output_dir_plots, f'{model_name.lower()}_pr_curve_test.png')
        plt.savefig(pr_plot_path)
        plt.close()
        plot_paths['PR'] = pr_plot_path
        print(f"{model_name} PR curve plot saved to: {pr_plot_path}")
        
    except Exception as e:
        print(f"Error plotting ROC/PR curves for {model_name}: {e}")
        traceback.print_exc()
        
    return plot_paths


def plot_confusion_matrix_heatmap(true_labels, pred_probs, model_name, output_dir_plots, threshold=0.5):
    """Plot confusion matrix heatmap"""
    if len(true_labels) == 0 or len(pred_probs) == 0:
        print(f"Warning: Insufficient data for {model_name} on test set. Cannot plot confusion matrix.")
        return None

    pred_binary = (np.array(pred_probs) > threshold).astype(int)
    
    try:
        cm_labels = [0,1] if (0 in true_labels and 1 in true_labels) else np.unique(true_labels)
        if len(cm_labels) < 2 :
            print(f"Warning: Only one class present in true labels for {model_name}. Cannot plot meaningful confusion matrix.")
            return None
        cm = confusion_matrix(true_labels, pred_binary, labels=cm_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=[f'Predicted {l}' for l in cm_labels], 
                    yticklabels=[f'Actual {l}' for l in cm_labels])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{model_name} - Confusion Matrix (Test Set, Threshold={threshold})')
        cm_plot_path = os.path.join(output_dir_plots, f'{model_name.lower()}_confusion_matrix_test.png')
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"{model_name} confusion matrix plot saved to: {cm_plot_path}")
        return cm_plot_path
    except Exception as e:
        print(f"Error plotting confusion matrix for {model_name}: {e}")
        traceback.print_exc()
        return None


def plot_models_comparison(all_results_dict, output_dir_plots):
    """Plot comparison of multiple models on key metrics"""
    if not all_results_dict:
        print("No model results available for comparison.")
        return None

    model_names = list(all_results_dict.keys())
    metrics_to_plot = ['auc', 'auprc', 'f1', 'precision', 'recall', 'specificity']
    
    num_metrics = len(metrics_to_plot)
    plt.figure(figsize=(max(15, num_metrics * 2.5), 5)) 
    
    for i, metric_key in enumerate(metrics_to_plot):
        metric_values = [all_results_dict[model].get(metric_key, 0.0) for model in model_names]
        
        ax = plt.subplot(1, num_metrics, i + 1)
        bars = plt.bar(model_names, metric_values, color=sns.color_palette("viridis", len(model_names)))
        plt.title(metric_key.upper(), fontsize=10)
        plt.ylabel('Score', fontsize=8)
        plt.xticks(rotation=45, ha="right", fontsize=8) 
        plt.ylim(0, 1.05) 
        plt.grid(axis='y', linestyle='--')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Model Performance Comparison (Test Set)', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    comparison_plot_path = os.path.join(output_dir_plots, 'models_comparison_test.png')
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Model performance comparison plot saved to: {comparison_plot_path}")
    return comparison_plot_path


# ---------------------- Main Program --------------------------
def main():
    """Main execution function"""
    start_time_main = time.time()
    print(f"Using device: {DEVICE}")
    if ENABLE_AMP:
        print("Automatic Mixed Precision (AMP) enabled.")

    lstm_test_true_labels, lstm_test_pred_probs = np.array([]), np.array([])
    transformer_test_true_labels, transformer_test_pred_probs = np.array([]), np.array([])
    
    all_models_test_results = {}

    print("\n[Phase 1/5] Loading data...")
    try:
        outcomes_df = pd.read_parquet(OUTCOMES_FILE)
        outcomes_map = outcomes_df.set_index(['stay_id', 'prediction_hour'])['outcome_death_next_24h'].to_dict()
    except Exception as e:
        print(f"Error: Failed to load outcomes file {OUTCOMES_FILE}: {e}")
        return

    all_pkl_files = sorted(glob.glob(os.path.join(TEMP_GRANULAR_DIR, f'*{COHORT_KEY}*.pkl')))
    if not all_pkl_files:
        print(f"Error: No matching PKL files found in {TEMP_GRANULAR_DIR}. Check path and filename pattern.")
        return

    file_mapping_info = []
    current_global_idx = 0
    print("  Building file map...")
    for pkl_file_path in tqdm(all_pkl_files, desc="  Scanning PKL files"):
        try:
            with open(pkl_file_path, 'rb') as f_pkl:
                num_samples_in_file = len(pickle.load(f_pkl))
            file_mapping_info.append({'path': pkl_file_path, 'offset': current_global_idx, 'count': num_samples_in_file})
            current_global_idx += num_samples_in_file
        except Exception as e:
            print(f"Warning: Failed to process file {pkl_file_path}: {e}")
            continue 
    
    if not file_mapping_info:
        print("Error: Failed to build file map, no usable data files.")
        return

    print("  Building identifiers and labels...")
    all_identifiers = []
    def load_ids_from_file(file_path_info_dict): # Renamed to avoid conflict
        ids_in_file = []
        try:
            with open(file_path_info_dict['path'], 'rb') as f:
                data_chunk = pickle.load(f) 
            for sample_tuple in data_chunk:
                ids_in_file.append((sample_tuple[0], sample_tuple[1]))
        except Exception as e:
            print(f"Warning: Failed to load identifiers from {file_path_info_dict['path']}: {e}")
        return ids_in_file

    # Use a modest number of workers for I/O bound tasks to avoid overwhelming disk
    # max_workers_io = min(4, (os.cpu_count() or 1) + 2) # Heuristic
    max_workers_io = 4 # Simpler fixed value, adjust based on system
    with ThreadPoolExecutor(max_workers=max_workers_io) as executor:
        results = list(tqdm(executor.map(load_ids_from_file, file_mapping_info), total=len(file_mapping_info), desc="  Extracting identifiers"))
    
    for id_list in results:
        all_identifiers.extend(id_list)

    if not all_identifiers:
        print("Error: Failed to extract any sample identifiers from PKL files.")
        return

    identifiers_df = pd.DataFrame(all_identifiers, columns=['stay_id', 'prediction_hour'])
    identifiers_df['y'] = identifiers_df.apply(
        lambda row: outcomes_map.get((row['stay_id'], row['prediction_hour']), 0), 
        axis=1
    )
    print(f"  Total samples loaded: {len(identifiers_df)}")

    if 'y' in identifiers_df.columns:
        print("  Data label distribution:")
        print(identifiers_df['y'].value_counts(normalize=True))
        # Check for class imbalance issue for pos_weight later
        class_counts = identifiers_df['y'].value_counts()
        if 1 in class_counts and 0 in class_counts:
            pos_weight_value = class_counts[0] / class_counts[1]
            print(f"  Calculated pos_weight for BCEWithLogitsLoss (if needed): {pos_weight_value:.2f}")
        else:
            pos_weight_value = 1.0 # Default if one class is missing (should not happen with good data)
            print("  Warning: Could not calculate pos_weight, one class might be missing or y column error.")
    else:
        print("Warning: 'y' column not in identifiers_df, cannot check label distribution or calculate pos_weight.")
        pos_weight_value = 1.0

    print("\n[Phase 2/5] Creating datasets and DataLoaders...")
    indices = np.arange(len(identifiers_df))
    # Stratify if 'y' column is available and has multiple classes
    stratify_labels = identifiers_df['y'].iloc[indices] if 'y' in identifiers_df.columns and len(identifiers_df['y'].unique()) > 1 else None

    train_val_indices, test_indices = train_test_split(indices, test_size=TEST_SPLIT, random_state=SEED, stratify=stratify_labels)
    
    stratify_labels_train_val = identifiers_df['y'].iloc[train_val_indices] if 'y' in identifiers_df.columns and len(identifiers_df['y'].iloc[train_val_indices].unique()) > 1 else None
    val_split_adjusted = VAL_SPLIT / (1 - TEST_SPLIT)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_split_adjusted, random_state=SEED, stratify=stratify_labels_train_val)

    print(f"  Training set samples: {len(train_indices)}")
    print(f"  Validation set samples: {len(val_indices)}")
    print(f"  Test set samples: {len(test_indices)}")

    full_dataset = GranularPklDataset(identifiers_df, file_mapping_info, N_FEATURES_GRANULAR)

    collate_fn_custom = functools.partial(vectorized_pad_collate, 
                                          max_len=MAX_LEN,
                                          n_features=N_FEATURES_GRANULAR, 
                                          padding_idx_embedding=PADDING_IDX_EMBEDDING)

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_MODEL,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_fn_custom)
    val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_MODEL * 2, 
                            sampler=SequentialSampler(val_indices), 
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            collate_fn=collate_fn_custom)
    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE_MODEL * 2,
                             sampler=SequentialSampler(test_indices),
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             collate_fn=collate_fn_custom)
    
    del outcomes_df, outcomes_map, identifiers_df, all_identifiers, file_mapping_info
    gc.collect()

    print("\n[Phase 3/5] Initializing models, optimizers, and loss function...")
    num_embeddings = N_BINS + 1 

    lstm_model = LSTMClassifierGranular(
        n_features=N_FEATURES_GRANULAR,
        n_bins_plus_padding=num_embeddings,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=LSTM_HIDDEN_DIM
    ).to(DEVICE)
    lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=1e-5)
    lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(lstm_optimizer, mode='max', factor=0.2, patience=5, verbose=True) # Patience 5 from your original

    transformer_model = TransformerClassifierGranular(
        n_features=N_FEATURES_GRANULAR,
        n_bins_plus_padding=num_embeddings,
        embedding_dim=EMBEDDING_DIM,
        d_model=D_MODEL_TRANSFORMER,
        nhead=8,
        num_encoder_layers=3, 
        dim_feedforward=D_MODEL_TRANSFORMER * 4, 
        dropout=0.1
    ).to(DEVICE)
    transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=1e-4, weight_decay=1e-5)
    transformer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(transformer_optimizer, mode='max', factor=0.2, patience=5, verbose=True) # Patience 5
    
    # Consider using pos_weight for imbalanced data
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=DEVICE))
    criterion = nn.BCEWithLogitsLoss().to(DEVICE) # Original criterion
    print(f"Using loss criterion: {criterion}")


    print("\n[Phase 4/5] Starting model training and evaluation...")
    
    try:
        lstm_training_history = train_model(lstm_model, 'LSTM', train_loader, val_loader, 
                                            lstm_optimizer, lstm_scheduler, criterion, 
                                            ES_PATIENCE, DEVICE, EPOCHS) # Pass EPOCHS
        if lstm_training_history and "loaded_val_metrics" not in lstm_training_history: 
            plot_training_metrics(lstm_training_history, 'LSTM', RESULTS_OUTPUT_DIR)
        
        print(f"Evaluating LSTM model on the test set...")
        lstm_test_true_labels, lstm_test_pred_probs = get_predictions_and_labels(lstm_model, test_loader, DEVICE, is_transformer=False)
        
        if len(lstm_test_true_labels) > 0:
            lstm_test_metrics = calculate_test_metrics(lstm_test_true_labels, lstm_test_pred_probs, 'LSTM')
            all_models_test_results['LSTM'] = lstm_test_metrics
            print_test_metrics('LSTM', lstm_test_metrics)
            plot_roc_pr_curves(lstm_test_true_labels, lstm_test_pred_probs, 'LSTM', RESULTS_OUTPUT_DIR)
            plot_confusion_matrix_heatmap(lstm_test_true_labels, lstm_test_pred_probs, 'LSTM', RESULTS_OUTPUT_DIR)
        else:
            print("LSTM model did not produce predictions on the test set. Skipping evaluation plots.")

    except Exception as e:
        print(f"Error: Critical error during LSTM model training or evaluation: {e}")
        traceback.print_exc()

    try:
        transformer_training_history = train_model(transformer_model, 'Transformer', train_loader, val_loader,
                                                   transformer_optimizer, transformer_scheduler, criterion,
                                                   ES_PATIENCE, DEVICE, EPOCHS) # Pass EPOCHS
        if transformer_training_history and "loaded_val_metrics" not in transformer_training_history:
            plot_training_metrics(transformer_training_history, 'Transformer', RESULTS_OUTPUT_DIR)

        print(f"Evaluating Transformer model on the test set...")
        transformer_test_true_labels, transformer_test_pred_probs = get_predictions_and_labels(transformer_model, test_loader, DEVICE, is_transformer=True)
        
        if len(transformer_test_true_labels) > 0:
            transformer_test_metrics = calculate_test_metrics(transformer_test_true_labels, transformer_test_pred_probs, 'Transformer')
            all_models_test_results['Transformer'] = transformer_test_metrics
            print_test_metrics('Transformer', transformer_test_metrics)
            plot_roc_pr_curves(transformer_test_true_labels, transformer_test_pred_probs, 'Transformer', RESULTS_OUTPUT_DIR)
            plot_confusion_matrix_heatmap(transformer_test_true_labels, transformer_test_pred_probs, 'Transformer', RESULTS_OUTPUT_DIR)
        else:
            print("Transformer model did not produce predictions on the test set. Skipping evaluation plots.")

    except Exception as e:
        print(f"Error: Critical error during Transformer model training or evaluation: {e}")
        traceback.print_exc()

    print("\n[Phase 5/5] Results summary and comparison...")
    if all_models_test_results:
        summary_file_path = os.path.join(RESULTS_OUTPUT_DIR, 'all_models_test_summary.json')
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(i) for i in obj]
                return obj

            serializable_results = convert_numpy_types(all_models_test_results)
            with open(summary_file_path, 'w') as f_summary:
                json.dump(serializable_results, f_summary, indent=4)
            print(f"All models test results summary saved to: {summary_file_path}")
        except Exception as e:
            print(f"Failed to save model test results summary: {e}")
            traceback.print_exc()

        plot_models_comparison(all_models_test_results, RESULTS_OUTPUT_DIR)
    else:
        print("No models completed evaluation successfully. Cannot generate comparison plot or summary.")

    if DEVICE.type == 'cuda':
        print(f"Current CUDA memory allocated: {torch.cuda.memory_allocated(DEVICE) / 1024**2:.2f} MB")
        print(f"Peak CUDA memory cached: {torch.cuda.max_memory_reserved(DEVICE) / 1024**2:.2f} MB") # max_memory_reserved is better

    end_time_main = time.time()
    print(f"\nTotal script runtime: {(end_time_main - start_time_main) / 60:.2f} minutes")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Unhandled top-level error during script execution: {e}")
        traceback.print_exc()
    finally:
        print("Script execution finished.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()