import os, sys, glob, pickle, traceback, functools
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm
import shap
import xgboost as xgb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------- 0. 配置 ------------------------------------
OUTPUT_DIR = "./output_dynamic"
COHORT_KEY = "aki"   # 'aki' or 'hf'
SHAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f"shap_analysis_{COHORT_KEY}")
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

MODEL_DIR_XGB = os.path.join(OUTPUT_DIR, f"step5_xgboost_{COHORT_KEY}.model")
MODEL_DIR_PT  = os.path.join(OUTPUT_DIR, f"step5_models_pytorch_{COHORT_KEY}")

MODEL_PATHS = {
    "xgboost": MODEL_DIR_XGB,
    "lstm": os.path.join(MODEL_DIR_PT, "best_LSTM.pt"),
    "transformer": os.path.join(MODEL_DIR_PT, "best_Transformer.pt"),
}

BIN_EDGES_FILE = os.path.join(OUTPUT_DIR,
                              f"step4ab_bin_edges_{COHORT_KEY}_5bins.pkl")

# --------- 通用超参 (需与训练保持一致) ------------------------------
N_BINS            = 5
MAX_SEQ_LEN       = 240
PADDING_IDX_EMB   = 0

# LSTM
LSTM_EMB_DIM      = 32
LSTM_HID_DIM      = 64

# Transformer
TRANS_EMB_DIM     = 32
TRANS_D_MODEL     = 128
TRANS_NHEAD       = 4
TRANS_NLAYERS     = 3
TRANS_FF          = 256
TRANS_DROPOUT     = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------------- 1. SHAP 参数 --------------------------------
# 优化 SHAP 分析的样本数和效率
N_SHAP_SAMPLES       = 100000    # 前景解释样本数量
N_BACKGROUND_SAMPLES = 30000   # 背景样本总数
N_KMEANS_CLUSTERS    = 10000    # K-Means压缩背景样本数（仅用于PyTorch模型）
KERNEL_BATCH_SIZE    = 10    # KernelExplainer的批处理大小
KERNEL_NSAMPLES      = 50000  # KernelExplainer采样次数，平衡速度和精度 

# ---------------------- 2. 全局占位 --------------------------------
N_FEATURES_GRANULAR = -1
model_type_to_analyze_for_data_conv = ""  # 保留原全局

# ---------------------- 3. 工具函数 --------------------------------
def load_feature_info(bin_edges_path):
    global N_FEATURES_GRANULAR
    try:
        with open(bin_edges_path, "rb") as f:
            bin_edges_map = pickle.load(f)
        feature_names = sorted(list(map(str, bin_edges_map.keys())))
        N_FEATURES_GRANULAR = len(feature_names)
        print(f"  成功加载 {N_FEATURES_GRANULAR} 个特征的名称")
        return feature_names, bin_edges_map
    except Exception as e:
        print("  读取特征失败:", e)
        traceback.print_exc()
        return None, None


# ---------- 4. Transformer 旧版兼容 --------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifierGranularLegacy(nn.Module):
    """仅 embedding+encoder+fc 兼容旧 checkpoint"""
    def __init__(self, n_feat, n_bins_plus_pad, emb_dim, d_model,
                 nhead, nlayer, d_ff, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(n_bins_plus_pad, emb_dim,
                                      padding_idx=PADDING_IDX_EMB)
        self.proj = nn.Linear(emb_dim * n_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, nlayer)
        self.posenc = PositionalEncoding(d_model, dropout, max_len)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x_idx, mask=None):
        b, l, m = x_idx.shape
        x = self.embedding(x_idx).view(b, l, -1)
        x = self.proj(x).transpose(0, 1)  # (l,b,d)
        x = self.posenc(x)
        mask = ~mask if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=mask)
        out = self.fc(x[-1]).squeeze(-1)
        return out


# ---------- 5. PytorchModelWrapper ---------------------------------
class PytorchModelWrapper(nn.Module):
    """确保 SHAP 输入/输出格式正确"""
    def __init__(self, model, model_type):
        super().__init__()
        self.model, self.type = model, model_type

    def forward(self, x_f32):
        # x_f32: B,L,M float → long idx
        x_long = torch.clamp(x_f32.long(), 0, N_BINS)
        if self.type == "transformer":
            # 修复：确保掩码是2D的 [B, L]，而不是3D [B, L, M]
            # 对于多特征情况，如果任何特征是填充值，则认为该时间步是填充
            if x_long.ndim == 3 and x_long.shape[2] > 1:
                # 多特征情况：如果任一特征是填充值，则该时间步为填充
                pad_mask = (x_long == PADDING_IDX_EMB).any(dim=2)
            else:
                # 单特征情况或已经是2D
                pad_mask = x_long.squeeze(-1).eq(PADDING_IDX_EMB) if x_long.ndim == 3 else x_long.eq(PADDING_IDX_EMB)
            
            preds = self.model(x_long, pad_mask)
        else:
            preds = self.model(x_long)

        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        return preds.float()   # ### FIX-PT-1  logits→float


# ---------- 6. 模型加载 --------------------------------------------
def load_model(model_type, path):
    if not os.path.exists(path):
        print("  模型文件缺失:", path)
        return None

    if model_type == "xgboost":
        booster = xgb.Booster(); booster.load_model(path); return booster

    elif model_type == "lstm":
        from step5_train_granular_models_pytorch import LSTMClassifierGranular
        mdl = LSTMClassifierGranular(N_FEATURES_GRANULAR, N_BINS + 1,
                                     LSTM_EMB_DIM, LSTM_HID_DIM)
        sd = torch.load(path, map_location=DEVICE)
        sd = sd.get("model_state_dict", sd)
        new_sd = OrderedDict((k[7:] if k.startswith("module.") else k, v)
                           for k, v in sd.items())
        mdl.load_state_dict(new_sd, strict=False)
        return mdl.to(DEVICE).eval()

    elif model_type == "transformer":
        mdl = TransformerClassifierGranularLegacy(
            N_FEATURES_GRANULAR, N_BINS + 1,
            TRANS_EMB_DIM, TRANS_D_MODEL,
            TRANS_NHEAD, TRANS_NLAYERS,
            TRANS_FF, TRANS_DROPOUT, MAX_SEQ_LEN)
        try:
            sd = torch.load(path, map_location=DEVICE)
            sd = sd.get("model_state_dict", sd)
            mdl.load_state_dict(sd, strict=False)      # ### FIX-TR-1
            print("  Transformer legacy 模式 strict=False 加载成功")
            return mdl.to(DEVICE).eval()
        except Exception as e:
            print("  legacy 加载失败:", e)
            traceback.print_exc()
            return None
    else:
        return None


# ---------- 7. 数据加载 -----------------------------------
def load_test_data():
    proc_dir = os.path.join(OUTPUT_DIR,
                            f"step4c_processed_granular_batches_{COHORT_KEY}")
    files = sorted(glob.glob(os.path.join(proc_dir, "processed_batch_*.npz")))
    if not files:
        raise FileNotFoundError(f"未找到 NPZ 批次文件于 {proc_dir}")

    X_col, y_col = [], []
    need = N_SHAP_SAMPLES + N_BACKGROUND_SAMPLES
    for fp in files:
        dat = np.load(fp)
        X, y = dat["X"].astype(np.int16), dat["y"]

        # 修复-1值，替换为填充值0
        X[X == -1] = 0
        np.clip(X, 0, N_BINS, out=X)

        X_col.append(X); y_col.append(y)
        if sum(x.shape[0] for x in X_col) >= need: break

    X_all = np.concatenate(X_col, 0)
    y_all = np.concatenate(y_col, 0)
    print(f"  加载数据: X {X_all.shape}, y {y_all.shape}")
    return X_all, y_all


# ---------- 8. SHAP 辅助绘图/保存 -----------------------------------
def plot_shap_summary(shap_values, X_df, fnames, mtype,
                      plot_type="bar", out_dir=None):
    plt.figure(figsize=(16, 10))  # 调整图像尺寸
    if plot_type == "bar":
        shap.summary_plot(shap_values, X_df, feature_names=fnames,
                          plot_type="bar", show=False, max_display=8, alpha=0.5)  # 增加透明度
        fname = f"{mtype}_{COHORT_KEY}_shap_bar.png"
    else:
        shap.summary_plot(shap_values, X_df, feature_names=fnames,
                          show=False, max_display=8, alpha=0.5)  # 增加透明度
        fname = f"{mtype}_{COHORT_KEY}_shap_dot.png"
    if out_dir:
        p = os.path.join(out_dir, fname)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        print("  SHAP 图已保存:", p)
    plt.close()


def save_feature_importance(shap_vals, fnames, mtype, out_dir):
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]  # 二分类正类
    imp = np.abs(shap_vals).mean(0)
    df = pd.DataFrame({"feature": fnames, "importance": imp})
    df.sort_values("importance", ascending=False, inplace=True)
    csv = os.path.join(out_dir,
                       f"{mtype}_{COHORT_KEY}_feature_importance.csv")
    df.to_csv(csv, index=False)
    print("  Feature importance CSV:", csv)
    return csv


# ---------- 9. 核心分析函数 ----------------------------------------
def analyze_model(model_type):
    global model_type_to_analyze_for_data_conv
    model_type_to_analyze_for_data_conv = model_type

    print(f"\n===== 分析 {model_type.upper()} ({COHORT_KEY.upper()}) =====")
    feat_names, _ = load_feature_info(BIN_EDGES_FILE)
    if not feat_names:
        model_type_to_analyze_for_data_conv = ""; return

    model = load_model(model_type, MODEL_PATHS[model_type])
    if model is None:
        model_type_to_analyze_for_data_conv = ""; return

    X_all, _ = load_test_data()
    bg_num = min(N_BACKGROUND_SAMPLES, X_all.shape[0])
    ex_num = min(N_SHAP_SAMPLES, max(1, X_all.shape[0] - bg_num))

    bg_np = X_all[:bg_num]
    ex_np = X_all[bg_num:bg_num+ex_num]

    print("  背景/解释 样本:", bg_np.shape, ex_np.shape)
    shap_vals = None

    if model_type == "xgboost":
        # XGBoost模型使用TreeExplainer - 速度快且可靠
        bg_flat = bg_np.reshape(bg_np.shape[0], -1)
        ex_flat = ex_np.reshape(ex_np.shape[0], -1)
        fn_full = [f"{fn}_t{i}"
                   for i in range(MAX_SEQ_LEN) for fn in feat_names]
        explainer = shap.TreeExplainer(model, bg_flat,
                                       feature_perturbation="interventional")
        shap_vals = explainer.shap_values(ex_flat)
        X_plot = pd.DataFrame(ex_flat, columns=fn_full)

    else:  # LSTM 或 Transformer - 使用优化的KernelExplainer
        wrapper = PytorchModelWrapper(model, model_type).to(DEVICE).eval()
        
        try:
            print("  为离散嵌入模型使用KernelExplainer + K-Means背景压缩...")
            # 转换数据为平坦格式
            bg_flat = bg_np.reshape(bg_np.shape[0], -1)
            ex_flat = ex_np.reshape(ex_np.shape[0], -1)
            
            # 创建扩展特征名
            fn_full = [f"{fn}_t{i}" for i in range(MAX_SEQ_LEN) for fn in feat_names]
            
            # 使用K-Means压缩背景样本以加速计算
            print(f"  执行K-Means压缩: {bg_flat.shape[0]}样本 → {N_KMEANS_CLUSTERS}聚类")
            bg_summary = shap.kmeans(bg_flat, N_KMEANS_CLUSTERS)
            
            # 优化预测函数，使用批处理和GPU加速
            def batch_predict(x):
                results = []
                with torch.no_grad():  # 关闭梯度计算加速推理
                    for i in range(0, x.shape[0], KERNEL_BATCH_SIZE):
                        end_idx = min(i + KERNEL_BATCH_SIZE, x.shape[0])
                        batch_x = x[i:end_idx]
                        x_tensor = torch.tensor(
                            batch_x.reshape(-1, MAX_SEQ_LEN, N_FEATURES_GRANULAR), 
                            dtype=torch.float32, device=DEVICE)
                        result = wrapper(x_tensor).cpu().numpy()
                        results.append(result)
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return np.concatenate(results, axis=0)
            
            # 创建KernelExplainer并计算SHAP值
            print(f"  创建KernelExplainer (nsamples={KERNEL_NSAMPLES})...")
            explainer = shap.KernelExplainer(batch_predict, bg_summary)
            
            # 分批处理以减少内存使用
            all_kernel_shap = []
            
            for i in range(0, ex_flat.shape[0], KERNEL_BATCH_SIZE):
                print(f"  计算SHAP批次 {i//KERNEL_BATCH_SIZE + 1}/{(ex_flat.shape[0]-1)//KERNEL_BATCH_SIZE + 1}")
                end_idx = min(i + KERNEL_BATCH_SIZE, ex_flat.shape[0])
                batch_ex = ex_flat[i:end_idx]
                
                # 使用适当的nsamples平衡精度和速度
                batch_shap_vals = explainer.shap_values(batch_ex, nsamples=KERNEL_NSAMPLES)
                
                if isinstance(batch_shap_vals, list):
                    batch_shap_vals = batch_shap_vals[0]  # 获取正类的SHAP值
                
                all_kernel_shap.append(batch_shap_vals)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 合并所有批次的SHAP值
            if all_kernel_shap:
                shap_vals = np.concatenate(all_kernel_shap, axis=0)
                print(f"  KernelExplainer成功计算SHAP值，形状: {shap_vals.shape}")
                
                # 创建特征DataFrame
                X_plot = pd.DataFrame(ex_flat[:shap_vals.shape[0]], 
                                      columns=fn_full[:shap_vals.shape[1]])
            
        except Exception as e:
            print(f"  KernelExplainer计算失败: {e}")
            traceback.print_exc()
            model_type_to_analyze_for_data_conv = ""
            return None

    # 最后检查是否成功生成SHAP值
    if shap_vals is None or np.isnan(shap_vals).any() or shap_vals.shape[0] < 10:
        print(f"  未能计算{model_type}的有效SHAP值或样本数过少")
        model_type_to_analyze_for_data_conv = ""
        return None

    print(f"  SHAP 计算完成，shape: {np.array(shap_vals).shape}")
    
    # === 保存 shap_vals 与 X_plot，供 step7 画 summary plot =========
    # 保存前打印SHAP值统计信息，帮助调试
    print(f"  SHAP值统计: min={np.min(shap_vals):.4f}, max={np.max(shap_vals):.4f}, " 
          f"mean={np.mean(shap_vals):.4f}, std={np.std(shap_vals):.4f}")
    print(f"  有效非零SHAP值比例: {np.sum(np.abs(shap_vals) > 1e-6) / shap_vals.size:.4f}")
    
    np.save(os.path.join(SHAP_OUTPUT_DIR,
            f"{model_type}_{COHORT_KEY}_shap_vals.npy"), shap_vals)
    X_plot.to_parquet(os.path.join(SHAP_OUTPUT_DIR,
            f"{model_type}_{COHORT_KEY}_X_plot.parquet"), index=False)
    print("  已保存 shap_vals.npy 与 X_plot.parquet 供可视化")
    
    # 保存特征重要性
    save_feature_importance(shap_vals, fn_full[:shap_vals.shape[1]], model_type, SHAP_OUTPUT_DIR)
    
    # 生成bar和dot图
    plot_shap_summary(shap_vals, X_plot, fn_full[:shap_vals.shape[1]], model_type, "bar",
                      SHAP_OUTPUT_DIR)
    plot_shap_summary(shap_vals, X_plot, fn_full[:shap_vals.shape[1]], model_type, "dot",
                      SHAP_OUTPUT_DIR)

    model_type_to_analyze_for_data_conv = ""  # reset
    return shap_vals


# ---------- 10. main -----------------------------------------------
def main():
    mtype_arg = sys.argv[1].lower() if len(sys.argv) > 1 else None
    targets = ([mtype_arg] if mtype_arg in MODEL_PATHS.keys()
               else ["xgboost", "lstm", "transformer"])
    for mt in targets:
        analyze_model(mt)
    print("\nSHAP 分析完成！结果保存在:", SHAP_OUTPUT_DIR)


if __name__ == "__main__":
    main()