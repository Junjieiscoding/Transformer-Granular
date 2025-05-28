# Transformer-Granular
---------------📖 Overview---------------

This project presents a dynamic risk prediction framework for ICU patients based on granular computing, hourly-level temporal modeling, and interpretable machine learning. It is developed as part of my undergraduate thesis and focuses on predicting adverse outcomes (e.g., in-hospital mortality) for acute kidney injury (AKI) patients using the MIMIC-IV database.

Key components include:
	•	Structured data processing pipeline from raw ICU logs
	•	Quantile-based feature granulation (discretization)
	•	Dynamic time-window slicing for sequential modeling
	•	Model training with XGBoost, LSTM, and Transformer
	•	SHAP-based interpretability analysis

---------------📊 Data---------------

This project uses the MIMIC-IV v3.1 dataset. Due to privacy restrictions, the raw data is not included here.

To reproduce the pipeline:
	1.	Obtain access to MIMIC-IV from PhysioNet
	2.	Follow the preprocessing steps in src/preprocess/ to build the target cohort (AKI) and construct hourly-level features.

---------------🚀 How to Run---------------

Follow the steps (e.g.1-->2-->3, a-->b-->c) and run it. 

---------------📈 Results---------------

Performance (on AKI patient mortality prediction):
Model         AUC        F1 Score        Accuracy
XGBoost      0.801        0.742          0.772
LSTM         0.829        0.761          0.785
Transformer  0.851        0.784          0.801

---------------🧩 Highlights---------------

•	✔️ Granular Computing: Formal granulation function using quantile binning to discretize physiological signals
•	✔️ Temporal Modeling: Dynamic prediction task constructed per-hour with variable-length sequence padding & masking
•	✔️ Interpretability: SHAP applied to both granular and numeric models for local and global explanation

---------------🧑‍💻 Author---------------

Sungu Junjie (孙谷俊杰)
Undergraduate Thesis, Xidian University (2025)
Email: sungujunjie@outlook.com

---------------📄 License---------------

This repository is licensed under the MIT License.

---------------📎 Citation---------------

If you find this project helpful, please cite:

@thesis{sungu2025Transformer-Granular,
  author  = {Sungu, Junjie},
  title   = {Transformer-Granular},
  school  = {Xidian University},
  year    = {2025},
  type    = {Undergraduate Thesis}
}
