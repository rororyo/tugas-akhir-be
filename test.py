1.   CLASS SIGAPFraudDetectionPipeline EXTENDS FraudDetectionPipeline:
2.
3.       FUNCTION __init__(artifacts_dir):
4.           # Memuat seluruh artefak pipeline dari hasil training
5.           meta              = load_metadata(artifacts_dir)
6.           SELF.threshold    = meta.best_model.threshold
7.           SELF.classifier   = load_pickle(meta.best_model.file)
8.           SELF.preprocessor = TabularPreprocessor(artifacts_dir)
9.           SELF.embedder     = HGTEmbedder(artifacts_dir)
10.      END FUNCTION
11.
12.      FUNCTION preprocess(df_raw):
13.          # 1. Validasi kolom wajib (mis. amount, merchant_category)
14.          # 2. Tangani nilai hilang (imputasi nilai default)
15.          # 3. Encoding fitur kategorikal & scaling fitur numerik
16.          (X_tabular_scaled, df_entity, df_features_raw, imputed_cols)
17.              = SELF.preprocessor.transform(df_raw)
18.          RETURN (X_tabular_scaled, df_entity, df_features_raw, imputed_cols)
19.      END FUNCTION
20.
21.      FUNCTION embed(df_features_raw, df_entity):
22.          # Membentuk graf heterogen antar entitas (user, merchant, device, dst.)
23.          # lalu menghasilkan vektor embedding berdimensi tetap (mis. 64 dimensi)
24.          X_embedding = SELF.embedder.get_embeddings(df_features_raw, df_entity)
25.          RETURN X_embedding
26.      END FUNCTION
27.
28.      FUNCTION predict_proba(df_raw):
29.          # 1. Praproses fitur tabular
30.          (X_tabular_scaled, df_entity, df_features_raw, imputed_cols) = SELF.preprocess(df_raw)
31.
32.          # 2. Hasilkan embedding relasi antar entitas
33.          X_embedding = SELF.embed(df_features_raw, df_entity)
34.
35.          # 3. Gabungkan fitur tabular & embedding sebagai input akhir
36.          X_final = CONCAT(X_tabular_scaled, X_embedding)
37.
38.          # 4. Prediksi probabilitas kelas fraud
39.          probabilities = SELF.classifier.predict_proba(X_final)
40.
41.          RETURN (probabilities[:, FRAUD_CLASS], imputed_cols)
42.      END FUNCTION
43.
44.      FUNCTION predict_label(df_raw):
45.          # Klasifikasi akhir berdasarkan ambang batas (threshold) hasil tuning
46.          (probabilities, imputed_cols) = SELF.predict_proba(df_raw)
47.          labels = [prob >= SELF.threshold FOR prob IN probabilities]
48.          RETURN (labels, imputed_cols)
49.      END FUNCTION
50.
51.      FUNCTION risk_level(probability):
52.          # Mengelompokkan probabilitas ke kategori risiko
53.          IF probability >= 0.7:  RETURN "high"
54.          IF probability >= 0.4:  RETURN "medium"
55.          RETURN "low"
56.      END FUNCTION
57.
58.  END CLASS
