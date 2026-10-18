"""
=============================================================================
MODUL 04: PELATIHAN MULTI-MODEL, STACKING & EVALUASI BERTINGKAT (TIERED)
Bali Smart Villa & Homestay Pricing Engine
=============================================================================
Fungsi:
1. Membaca data latih (dataset_train.csv) dan data uji (dataset_test.csv).
2. Melatih dan membandingkan 5 algoritma regresi dengan 5-Fold CV:
   - Random Forest Regressor (Baseline Ensemble Bagging)
   - XGBoost Regressor (Extreme Gradient Boosting)
   - HistGradientBoosting / LightGBM Architecture (Histogram Gradient Boosting)
   - CatBoost Regressor (Champion Single Model dengan Native Categorical)
   - Stacked Super-Ensemble (Out-of-Fold Ridge Meta-Learner)
3. Menghitung metrik ilmiah lengkap: R2 Score, MAE (Rupiah), MedAE (Rupiah),
   RMSE (Rupiah), dan MdAPE (Persentase).
4. Menghasilkan evaluasi akurasi per kelas harga (Tiered Evaluation) untuk
   membuktikan bebasnya model dari halusinasi pada kelas Homestay Budget
   maupun Villa Reguler.
5. Menyimpan model terbaik ke models/ dan laporan evaluasi ke reports/.
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(BASE_DIR, "data", "processed", "dataset_train.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "processed", "dataset_test.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def evaluate_metrics(y_true_log, y_pred_log):
    r2 = r2_score(y_true_log, y_pred_log)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    med_ae = np.median(np.abs(y_true - y_pred))
    med_ape = np.median(np.abs(y_true - y_pred) / y_true) * 100
    return r2, mae, rmse, med_ae, med_ape


def main():
    print("=" * 75)
    print(" [MODUL 04] PELATIHAN DAN KOMPARASI 5 ALGORITMA MACHINE LEARNING")
    print("=" * 75)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    cat_cols = ['regency', 'sub_district', 'property_type']
    for c in cat_cols:
        train_df[c] = train_df[c].fillna('Unknown').astype(str)
        test_df[c] = test_df[c].fillna('Unknown').astype(str)

    non_feature_cols = ['room_id', 'name', 'price_idr', 'log_price']
    num_cols = [c for c in train_df.columns if c not in cat_cols + non_feature_cols]

    print(f"[*] Jumlah Data Latih : {len(train_df)} baris (80%)")
    print(f"[*] Jumlah Data Uji   : {len(test_df)} baris (20%)")
    print(f"[*] Fitur Numerik     : {len(num_cols)} kolom (termasuk Spatial Hedonic PPB IDW)")
    print(f"[*] Fitur Kategori    : {cat_cols}")

    # One-Hot Encoding untuk Random Forest, XGBoost, dan HistGradientBoosting
    df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    df_encoded = pd.get_dummies(df_all, columns=cat_cols, drop_first=True)

    encoded_train = df_encoded.iloc[:len(train_df)].copy()
    encoded_test = df_encoded.iloc[len(train_df):].copy()

    feature_cols_encoded = [c for c in encoded_train.columns if c not in non_feature_cols]

    X_train_enc = encoded_train[feature_cols_encoded].astype(float)
    y_train_log = encoded_train['log_price'].values
    X_test_enc = encoded_test[feature_cols_encoded].astype(float)
    y_test_log = encoded_test['log_price'].values

    X_train_raw = train_df[num_cols + cat_cols]
    X_test_raw = test_df[num_cols + cat_cols]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    oof_rf = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_lgb = np.zeros(len(train_df))
    oof_cb = np.zeros(len(train_df))

    rf_cv_r2, rf_cv_mae = [], []
    xgb_cv_r2, xgb_cv_mae = [], []
    lgb_cv_r2, lgb_cv_mae = [], []
    cb_cv_r2, cb_cv_mae = [], []

    print("\n[1/5] Menjalankan 5-Fold Out-of-Fold Cross Validation untuk 4 Base Learners...")
    for train_idx, val_idx in kf.split(X_train_enc):
        # 1. Random Forest Fold
        rf_fold = RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1)
        rf_fold.fit(X_train_enc.iloc[train_idx], y_train_log[train_idx])
        oof_rf[val_idx] = rf_fold.predict(X_train_enc.iloc[val_idx])
        r2, mae, _, _, _ = evaluate_metrics(y_train_log[val_idx], oof_rf[val_idx])
        rf_cv_r2.append(r2)
        rf_cv_mae.append(mae)

        # 2. XGBoost Fold
        xgb_fold = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.035, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2
        )
        xgb_fold.fit(X_train_enc.iloc[train_idx], y_train_log[train_idx])
        oof_xgb[val_idx] = xgb_fold.predict(X_train_enc.iloc[val_idx])
        r2, mae, _, _, _ = evaluate_metrics(y_train_log[val_idx], oof_xgb[val_idx])
        xgb_cv_r2.append(r2)
        xgb_cv_mae.append(mae)

        # 3. LightGBM / HistGradientBoosting Fold
        lgb_fold = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.04, max_depth=6,
            min_samples_leaf=8, l2_regularization=1.0, random_state=42
        )
        lgb_fold.fit(X_train_enc.iloc[train_idx], y_train_log[train_idx])
        oof_lgb[val_idx] = lgb_fold.predict(X_train_enc.iloc[val_idx])
        r2, mae, _, _, _ = evaluate_metrics(y_train_log[val_idx], oof_lgb[val_idx])
        lgb_cv_r2.append(r2)
        lgb_cv_mae.append(mae)

        # 4. CatBoost Fold
        cb_fold = CatBoostRegressor(
            iterations=500, learning_rate=0.035, depth=6, l2_leaf_reg=3.0, loss_function='RMSE',
            cat_features=cat_cols, thread_count=2, random_seed=42, verbose=0
        )
        cb_fold.fit(X_train_raw.iloc[train_idx], y_train_log[train_idx])
        oof_cb[val_idx] = cb_fold.predict(X_train_raw.iloc[val_idx])
        r2, mae, _, _, _ = evaluate_metrics(y_train_log[val_idx], oof_cb[val_idx])
        cb_cv_r2.append(r2)
        cb_cv_mae.append(mae)

    rf_oof_r2, rf_oof_mae, _, _, _ = evaluate_metrics(y_train_log, oof_rf)
    xgb_oof_r2, xgb_oof_mae, _, _, _ = evaluate_metrics(y_train_log, oof_xgb)
    lgb_oof_r2, lgb_oof_mae, _, _, _ = evaluate_metrics(y_train_log, oof_lgb)
    cb_oof_r2, cb_oof_mae, _, _, _ = evaluate_metrics(y_train_log, oof_cb)

    # -------------------------------------------------------------
    # PELATIHAN MODEL PENUH PADA DATA LATIH & EVALUASI DATA UJI
    # -------------------------------------------------------------
    print("\n[2/5] Melatih Random Forest Regressor (Baseline)...")
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_enc, y_train_log)
    rf_test_pred = rf_model.predict(X_test_enc)
    rf_r2, rf_mae, rf_rmse, rf_medae, rf_medape = evaluate_metrics(y_test_log, rf_test_pred)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest_model.joblib"))
    results.append({
        "Model": "Random Forest (Baseline)",
        "CV_R2_Mean": rf_oof_r2,
        "CV_MAE_IDR": rf_oof_mae,
        "Test_R2": rf_r2,
        "Test_MAE_IDR": rf_mae,
        "Test_MedAE_IDR": rf_medae,
        "Test_RMSE_IDR": rf_rmse,
        "Test_MdAPE_Pct": rf_medape
    })
    print(f" -> RF Test R2: {rf_r2:.4f} | MAE: Rp {rf_mae:,.0f} | MedAE: Rp {rf_medae:,.0f} | MdAPE: {rf_medape:.2f}%")

    print("\n[3/5] Melatih XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=350, learning_rate=0.035, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2
    )
    xgb_model.fit(X_train_enc, y_train_log)
    xgb_test_pred = xgb_model.predict(X_test_enc)
    xgb_r2, xgb_mae, xgb_rmse, xgb_medae, xgb_medape = evaluate_metrics(y_test_log, xgb_test_pred)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    results.append({
        "Model": "XGBoost Regressor",
        "CV_R2_Mean": xgb_oof_r2,
        "CV_MAE_IDR": xgb_oof_mae,
        "Test_R2": xgb_r2,
        "Test_MAE_IDR": xgb_mae,
        "Test_MedAE_IDR": xgb_medae,
        "Test_RMSE_IDR": xgb_rmse,
        "Test_MdAPE_Pct": xgb_medape
    })
    print(f" -> XGBoost Test R2: {xgb_r2:.4f} | MAE: Rp {xgb_mae:,.0f} | MedAE: Rp {xgb_medae:,.0f} | MdAPE: {xgb_medape:.2f}%")

    print("\n[4/5] Melatih LightGBM / HistGradientBoosting Regressor...")
    hgb_model = HistGradientBoostingRegressor(
        max_iter=350, learning_rate=0.04, max_depth=6,
        min_samples_leaf=8, l2_regularization=1.0, random_state=42
    )
    hgb_model.fit(X_train_enc, y_train_log)
    hgb_test_pred = hgb_model.predict(X_test_enc)
    hgb_r2, hgb_mae, hgb_rmse, hgb_medae, hgb_medape = evaluate_metrics(y_test_log, hgb_test_pred)
    joblib.dump(hgb_model, os.path.join(MODELS_DIR, "lightgbm_hist_model.joblib"))
    results.append({
        "Model": "LightGBM (HistGradientBoosting)",
        "CV_R2_Mean": lgb_oof_r2,
        "CV_MAE_IDR": lgb_oof_mae,
        "Test_R2": hgb_r2,
        "Test_MAE_IDR": hgb_mae,
        "Test_MedAE_IDR": hgb_medae,
        "Test_RMSE_IDR": hgb_rmse,
        "Test_MdAPE_Pct": hgb_medape
    })
    print(f" -> LightGBM Test R2: {hgb_r2:.4f} | MAE: Rp {hgb_mae:,.0f} | MedAE: Rp {hgb_medae:,.0f} | MdAPE: {hgb_medape:.2f}%")

    print("\n[5/5] Melatih CatBoost Regressor (Champion Model) & Stacked Super-Ensemble...")
    cb_model = CatBoostRegressor(
        iterations=600, learning_rate=0.035, depth=6, l2_leaf_reg=3.0, loss_function='RMSE',
        cat_features=cat_cols, thread_count=2, random_seed=42, verbose=0
    )
    cb_model.fit(X_train_raw, y_train_log)
    cb_test_pred = cb_model.predict(X_test_raw)
    cb_r2, cb_mae, cb_rmse, cb_medae, cb_medape = evaluate_metrics(y_test_log, cb_test_pred)

    cb_model.save_model(os.path.join(MODELS_DIR, "catboost_pricing_model.cbm"))
    joblib.dump(cb_model, os.path.join(MODELS_DIR, "catboost_model.joblib"))

    results.append({
        "Model": "CatBoost Regressor",
        "CV_R2_Mean": cb_oof_r2,
        "CV_MAE_IDR": cb_oof_mae,
        "Test_R2": cb_r2,
        "Test_MAE_IDR": cb_mae,
        "Test_MedAE_IDR": cb_medae,
        "Test_RMSE_IDR": cb_rmse,
        "Test_MdAPE_Pct": cb_medape
    })
    print(f" -> CatBoost Test R2: {cb_r2:.4f} | MAE: Rp {cb_mae:,.0f} | MedAE: Rp {cb_medae:,.0f} | MdAPE: {cb_medape:.2f}%")

    # 5. Stacked Super-Ensemble (RidgeCV Meta-Learner pada Out-of-Fold Predictions)
    oof_stack = np.column_stack([oof_rf, oof_xgb, oof_lgb, oof_cb])
    test_stack = np.column_stack([rf_test_pred, xgb_test_pred, hgb_test_pred, cb_test_pred])
    meta_ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
    meta_ridge.fit(oof_stack, y_train_log)
    stack_oof_pred = meta_ridge.predict(oof_stack)
    stack_test_pred = meta_ridge.predict(test_stack)

    st_cv_r2, st_cv_mae, _, _, _ = evaluate_metrics(y_train_log, stack_oof_pred)
    st_r2, st_mae, st_rmse, st_medae, st_medape = evaluate_metrics(y_test_log, stack_test_pred)
    joblib.dump(meta_ridge, os.path.join(MODELS_DIR, "stacked_ridge_meta.joblib"))

    results.append({
        "Model": "Stacked Super-Ensemble",
        "CV_R2_Mean": st_cv_r2,
        "CV_MAE_IDR": st_cv_mae,
        "Test_R2": st_r2,
        "Test_MAE_IDR": st_mae,
        "Test_MedAE_IDR": st_medae,
        "Test_RMSE_IDR": st_rmse,
        "Test_MdAPE_Pct": st_medape
    })
    print(f" -> Stacked Ensemble Test R2: {st_r2:.4f} | CV R2: {st_cv_r2:.4f} | MAE: Rp {st_mae:,.0f} | MedAE: Rp {st_medae:,.0f}")

    joblib.dump({
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "all_features": num_cols + cat_cols,
        "feature_cols_encoded": feature_cols_encoded
    }, os.path.join(MODELS_DIR, "feature_metadata.joblib"))

    # -------------------------------------------------------------
    # REKAPITULASI KOMPARASI MODEL & EVALUASI PER KELAS HARGA (TIER)
    # -------------------------------------------------------------
    bench_df = pd.DataFrame(results)
    bench_df.to_csv(os.path.join(REPORTS_DIR, "model_benchmark_results.csv"), index=False)

    actual_idr = np.expm1(y_test_log)
    pred_idr = np.expm1(cb_test_pred)

    tier_defs = [
        ("Homestay / Budget (< Rp 750 Ribu)", 0, 750000, "Kamar Pribadi, Homestay & Guesthouse"),
        ("Villa Reguler (Rp 750 Ribu - Rp 2,5 Juta)", 750000, 2500000, "Villa 1-2 Kamar Populer (Sanur/Ubud/Kuta)"),
        ("Villa Menengah (Rp 2,5 Juta - Rp 6 Juta)", 2500000, 6000000, "Villa 2-4 Kamar Kolam Renang Pribadi"),
        ("Villa Premium (Rp 6 Juta - Rp 10 Juta)", 6000000, 10500000, "Villa Mewah 3-6 Kamar Pantai/Tebing")
    ]
    tier_rows = []
    for t_name, low_val, high_val, t_desc in tier_defs:
        mask = (actual_idr >= low_val) & (actual_idr < high_val)
        cnt = int(mask.sum())
        if cnt > 0:
            t_actual_mean = float(np.mean(actual_idr[mask]))
            t_pred_mean = float(np.mean(pred_idr[mask]))
            t_mae = float(mean_absolute_error(actual_idr[mask], pred_idr[mask]))
            t_medae = float(np.median(np.abs(actual_idr[mask] - pred_idr[mask])))
            t_mdape = float(np.median(np.abs(actual_idr[mask] - pred_idr[mask]) / actual_idr[mask]) * 100)
            tier_rows.append({
                "Kelas_Penginapan": t_name,
                "Jumlah_Sampel_Uji": cnt,
                "Rata_Rata_Aktual_IDR": t_actual_mean,
                "Rata_Rata_Prediksi_IDR": t_pred_mean,
                "MAE_IDR": t_mae,
                "MedAE_IDR": t_medae,
                "MdAPE_Pct": t_mdape,
                "Karakteristik_Properti": t_desc
            })

    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(os.path.join(REPORTS_DIR, "tier_accuracy_breakdown.csv"), index=False)

    print("\n" + "=" * 75)
    print(" [TABEL KOMPARASI AKURASI 5 ALGORITMA MACHINE LEARNING]")
    print("=" * 75)
    print(bench_df.to_string(index=False))
    print("\n" + "=" * 75)
    print(" [RINCIAN AKURASI PER KELAS PASAR PENGINAPAN (CATBOOST)]")
    print("=" * 75)
    print(tier_df.to_string(index=False))
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
