import pickle
import numpy as np

ARTIFACTS_DIR = "fraud_detection_artifacts"  # sesuaikan dengan path folder Anda

# 1. Load model Gradient Boosting Skenario 3 (tanpa retraining)
with open(f"{ARTIFACTS_DIR}/model_s3_GradientBoosting_emb.pkl", "rb") as f:
    gb_sc3 = pickle.load(f)

# 2. Susun nama fitur SESUAI URUTAN saat training (26 tabular + 64 embedding)
tabular_cols = [
    "account_age_days", "total_transactions_user", "avg_amount_user", "amount",
    "promo_used", "avs_match", "cvv_result", "three_ds_flag", "shipping_distance_km",
    "hour", "hour_sin", "hour_cos", "day_of_week", "month", "log_amount", "dist_log",
    "acct_age_log", "tx_count_log", "card_testing", "new_account_flag",
    "young_account_flag", "new_account_promo", "high_distance_flag",
    "late_night_flag", "channel_app", "channel_web",
]
embedding_cols = [f"hgt_emb_{i}" for i in range(64)]
feat_names = tabular_cols + embedding_cols  # total 90

assert len(feat_names) == gb_sc3.n_features_in_, \
    f"Jumlah nama fitur ({len(feat_names)}) != n_features_in_ model ({gb_sc3.n_features_in_})"

# 3. Ambil importance
importances = gb_sc3.feature_importances_

# 4. Ranking top-N
N = 15
order = np.argsort(importances)[::-1]
print(f"{'Peringkat':<10}{'Fitur':<26}{'Kelompok':<12}{'Importance':>10}")
for rank, idx in enumerate(order[:N], start=1):
    kelompok = "Tabular" if idx < 26 else "Embedding"
    print(f"{rank:<10}{feat_names[idx]:<26}{kelompok:<12}{importances[idx]:>10.4f}")

# 5. Agregat per kelompok (untuk Tabel 4.22)
tab_total = importances[:26].sum()
emb_total = importances[26:].sum()
print("\n=== Agregat ===")
print(f"Total importance 26 fitur tabular   : {tab_total:.4f}  ({tab_total*100:.2f}%)")
print(f"Total importance 64 fitur embedding : {emb_total:.4f}  ({emb_total*100:.2f}%)")
print(f"(Cek total ≈ 1.0000: {tab_total+emb_total:.4f})")