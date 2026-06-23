import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DROP_COLS = [
    "transaction_id", "transaction_time",
    "user_id", "country", "bin_country", "merchant_category", "_card_proxy",
    "is_fraud",
]

REQUIRED_COLS = {"amount", "merchant_category"}

# Imputation defaults — applied BEFORE feature engineering.
# total_transactions_user and avg_amount_user are NOT here — they are always
# computed from the batch itself by infer_user_stats().
IMPUTATION_DEFAULTS = {
    "account_age_days":     973,
    "shipping_distance_km": 356.9,
    "avs_match":            1,
    "cvv_result":           1,
    "three_ds_flag":        1,
    "promo_used":           0,
    "channel":              "web",
    "country":              "US",
    "bin_country":          "US",
}


def validate_batch(df: pd.DataFrame) -> str | None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return f"File harus mengandung kolom: {', '.join(sorted(missing))}"
    if df["amount"].isna().all():
        return "Kolom 'amount' tidak boleh kosong seluruhnya"
    if df["merchant_category"].isna().all():
        return "Kolom 'merchant_category' tidak boleh kosong seluruhnya"
    return None


def infer_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_row_order"] = np.arange(len(df))

    totals = []
    avgs   = []
    # Track per-user cumulative state
    user_count: dict = {}
    user_sum:   dict = {}

    for _, row in df.iterrows():
        uid = row["user_id"]
        amt = float(row["amount"])
        cnt = user_count.get(uid, 0) + 1
        s   = user_sum.get(uid, 0.0) + amt
        user_count[uid] = cnt
        user_sum[uid]   = s
        totals.append(cnt)
        avgs.append(s / cnt)

    df["total_transactions_user"] = totals
    df["avg_amount_user"]         = avgs
    df.drop(columns=["_row_order"], inplace=True)
    return df


def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    imputed: set[str] = set()

    for col, default in IMPUTATION_DEFAULTS.items():
        if col not in df.columns:
            imputed.add(col)
            if col == "avg_amount_user":
                df[col] = df["amount"]
            else:
                df[col] = default
        else:
            mask = df[col].isna()
            if mask.any():
                imputed.add(col)
                if col == "avg_amount_user":
                    df.loc[mask, col] = df.loc[mask, "amount"]
                else:
                    df[col] = df[col].fillna(default)

    # bin_country: copy from country if still missing
    if df["bin_country"].isna().any():
        df["bin_country"] = df["bin_country"].fillna(df["country"])

    # transaction_time
    if "transaction_time" not in df.columns:
        df["transaction_time"] = pd.Timestamp.now()
        imputed.add("transaction_time")
    else:
        mask = df["transaction_time"].isna()
        if mask.any():
            df.loc[mask, "transaction_time"] = pd.Timestamp.now()
            imputed.add("transaction_time")

    return df, sorted(imputed)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], utc=True) \
                               .dt.tz_convert(None)
    hour = df["transaction_time"].dt.hour
    df["hour"]        = hour
    df["hour_sin"]    = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * hour / 24)
    df["day_of_week"] = df["transaction_time"].dt.dayofweek
    df["month"]       = df["transaction_time"].dt.month
    df["log_amount"]   = np.log1p(df["amount"])
    df["dist_log"]     = np.log1p(df["shipping_distance_km"])
    df["acct_age_log"] = np.log1p(df["account_age_days"])
    df["tx_count_log"] = np.log1p(df["total_transactions_user"])
    df["card_testing"]      = (df["amount"] < 5.0).astype(int)
    df["new_account_flag"]   = (df["account_age_days"] < 30).astype(int)
    df["young_account_flag"] = (
        (df["account_age_days"] >= 30) & (df["account_age_days"] < 90)
    ).astype(int)
    df["new_account_promo"]  = (
        (df["account_age_days"] < 30) & (df["promo_used"] == 1)
    ).astype(int)
    df["high_distance_flag"] = (df["shipping_distance_km"] > 1000).astype(int)
    df["late_night_flag"]    = ((hour >= 23) | (hour < 2)).astype(int)
    return df


class TabularPreprocessor:
    def __init__(self, artifacts_dir: str = "fraud_detection_artifacts"):
        ad = artifacts_dir
        with open(os.path.join(ad, "encoders_s1.pkl"), "rb") as f:
            self.enc_s1 = pickle.load(f)
        with open(os.path.join(ad, "scaler_s1.pkl"), "rb") as f:
            self.scaler_s1 = pickle.load(f)
        with open(os.path.join(ad, "feature_cols_gnn.pkl"), "rb") as f:
            self.feature_cols_gnn = pickle.load(f)
        with open(os.path.join(ad, "tabular_meta.pkl"), "rb") as f:
            self.tabular_meta = pickle.load(f)
        print(f"✅ TabularPreprocessor ready — {len(self.feature_cols_gnn)} feature cols")

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = {}
        for col, enc in self.enc_s1.items():
            if col not in df.columns:
                if isinstance(enc, OneHotEncoder):
                    for cat in enc.categories_[0]:
                        parts[f"{col}_{cat}"] = np.zeros(len(df))
                else:
                    parts[col] = np.zeros(len(df), dtype=int)
                continue

            if isinstance(enc, OneHotEncoder):
                arr = enc.transform(df[[col]].astype(str))
                for i, cat in enumerate(enc.categories_[0]):
                    parts[f"{col}_{cat}"] = arr[:, i]
            else:
                known = set(enc.classes_)
                fallback = enc.classes_[0]
                mapped = df[col].astype(str).apply(
                    lambda x: x if x in known else fallback
                )
                parts[col] = enc.transform(mapped)

        return pd.DataFrame(parts, index=df.index)

    def transform(self, df_raw: pd.DataFrame):
        # Always infer these two from the batch — never trust file-supplied values
        df_raw = infer_user_stats(df_raw)
        df, imputed_cols = impute_missing(df_raw)

        # Coerce dtypes
        df["amount"]                  = df["amount"].astype(float)
        df["account_age_days"]        = df["account_age_days"].astype(float)
        df["total_transactions_user"] = df["total_transactions_user"].astype(float)
        df["avg_amount_user"]         = df["avg_amount_user"].astype(float)
        df["shipping_distance_km"]    = df["shipping_distance_km"].astype(float)
        df["promo_used"]              = df["promo_used"].astype(int)
        df["avs_match"]               = df["avs_match"].astype(int)
        df["cvv_result"]              = df["cvv_result"].astype(int)
        df["three_ds_flag"]           = df["three_ds_flag"].astype(int)

        # Compute _card_proxy server-side (never from client)
        df["_card_proxy"] = df["user_id"].astype(str) + "_" + df["bin_country"].astype(str)

        # Stash entity id columns
        df_entity = df[["user_id", "_card_proxy", "merchant_category"]].copy()

        # Feature engineering
        df_fe = engineer_features(df)

        # Encode categoricals
        cols_to_encode = [
            c for c in df_fe.columns
            if c not in DROP_COLS
            and df_fe[c].dtype == object
            and c in self.enc_s1
        ]
        df_dropped = df_fe.drop(
            columns=[c for c in DROP_COLS if c in df_fe.columns],
            errors="ignore",
        )
        # Drop object and all datetime variants regardless of resolution/tz
        dt_cols = [c for c in df_dropped.columns
                   if pd.api.types.is_datetime64_any_dtype(df_dropped[c])]
        df_num = df_dropped.drop(columns=dt_cols, errors="ignore") \
                            .select_dtypes(exclude=["object"])

        df_cat_encoded = self._encode_categorical(
            df_fe[[c for c in cols_to_encode if c in df_fe.columns]]
        )

        df_assembled = pd.concat(
            [df_num.reset_index(drop=True), df_cat_encoded.reset_index(drop=True)],
            axis=1,
        )

        # Align to exact training column order
        df_feat_raw = df_assembled.reindex(
            columns=self.feature_cols_gnn, fill_value=0
        ).fillna(0)

        # Scale
        X_tab_scaled = self.scaler_s1.transform(df_feat_raw.values.astype(np.float32))

        return X_tab_scaled, df_entity, df_feat_raw, imputed_cols
