import os
import sys
import gc
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

# Tự động định cấu hình sys.path để hỗ trợ chạy độc lập
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Import trực tiếp các hàm từ pipeline gốc để đảm bảo 100% nhất quán đặc trưng!
from step1_dual_model import (
    load_and_aggregate_data,
    build_dense_grid,
    create_shared_features,
    create_lag_features_public,
    create_lag_features_private,
    FastWRMSSEEvaluator,
    _get_lag_cols,
    Config,
    find_best_multiplier
)

warnings.filterwarnings('ignore')

CLUSTERS_PATH = os.path.join(Config.DATA_DIR, 'sku_clustering_3_groups.csv')
CKPT_PUB  = os.path.join(Config.DATA_DIR, 'v22_step2_ckpt_pub.pkl')
CKPT_PRIV = os.path.join(Config.DATA_DIR, 'v22_step2_ckpt_priv.pkl')

# Tham số Nhóm B (Siêu thưa) - dùng chung cho cả Public và Private
PARAMS_B = {
    'objective': 'tweedie', 'tweedie_variance_power': 1.35, 'metric': 'rmse',
    'learning_rate': 0.03, 'num_leaves': 63, 'min_data_in_leaf': 300,
    'feature_fraction': 0.60, 'reg_alpha': 5.0, 'reg_lambda': 10.0,
    'random_state': Config.SEED, 'n_jobs': -1, 'n_estimators': 800
}


def train_decentralized_models(df_base, profit_df, cluster_map):
    print("\n" + "="*80)
    print("      HUẤN LUYỆN PHI TẬP TRUNG: NHÓM HỖN HỢP (NHÓM A vs NHÓM B)")
    print("="*80)

    exclude = ['Date', 'Quantity', 'day_int', 'profit_weight', 'ItemCode', 'cluster', 'ReturnQty', 'GiftQty']
    val_start    = 1727
    skus_order   = profit_df['ItemCode'].values
    weights_arr  = profit_df['weight'].values
    weight_map   = profit_df.set_index('ItemCode')['weight'].to_dict()

    train_wide = np.clip(df_base[df_base['day_int'] < val_start].pivot(
        index='ItemCode', columns='Date', values='Quantity'
    ).reindex(skus_order).values, 0.0, None)
    train_wide = np.nan_to_num(train_wide, nan=0.0)

    valid_wide = np.clip(df_base[(df_base['day_int'] >= val_start) & (df_base['day_int'] <= 1754)].pivot(
        index='ItemCode', columns='Date', values='Quantity'
    ).reindex(skus_order).values, 0.0, None)
    valid_wide = np.nan_to_num(valid_wide, nan=0.0)

    # =========================================================
    # PHASE 1: MÔ HÌNH CÔNG KHAI (F1-F28, lag >= 28)
    # =========================================================
    print("\n>>> TẠO ĐẶC TRƯNG LAG CÔNG KHAI (F1-F28)...")
    df_pub = df_base.copy()
    df_pub = create_lag_features_public(df_pub)
    df_pub['profit_weight'] = df_pub['ItemCode'].map(weight_map).fillna(0).astype(np.float32)
    df_pub['cluster']       = df_pub['ItemCode'].map(cluster_map).fillna(0).astype(int)
    feats_pub = _get_lag_cols(df_pub, exclude)

    if os.path.exists(CKPT_PUB):
        print(f"[CHECKPOINT] Phát hiện checkpoint Công khai! Đang tải, bỏ qua huấn luyện...")
        c = joblib.load(CKPT_PUB)
        models_pub, best_mult_pub = c['models'], c['mult']
        print(f"[CHECKPOINT] ✓ WRMSSE={c['score']:.5f} | Multiplier={best_mult_pub:.3f}")
    else:
        print("\n>>> HUẤN LUYỆN MÔ HÌNH CÔNG KHAI (F1-F28) <<<")
        train_start = max(60, val_start - Config.TRAIN_WINDOW_DAYS)
        train_mask  = (df_pub['day_int'] < val_start) & (df_pub['day_int'] >= train_start) & (df_pub['Quantity'].notna())
        val_mask    = (df_pub['day_int'] >= val_start) & (df_pub['day_int'] <= 1754)

        val_preds_pub = np.zeros(len(df_pub[val_mask]))
        df_pub_val = df_pub[val_mask].copy().reset_index(drop=True)
        df_pub_val['idx_val'] = df_pub_val.index
        models_pub   = {}
        cat_features = ['cat_proxy'] if 'cat_proxy' in feats_pub else []

        # --- Nhóm A: Standard & Superstars (Cụm 0 và 1) ---
        mask_tr_a    = train_mask & (df_pub['cluster'].isin([0, 1]))
        mask_va_a    = val_mask   & (df_pub['cluster'].isin([0, 1]))
        indices_va_a = df_pub_val[df_pub_val['cluster'].isin([0, 1])]['idx_val'].values
        print(f"\n[+] Nhóm A (Công khai: Standard & Superstars) - Train: {mask_tr_a.sum()}, Val: {mask_va_a.sum()}")

        X_tr_a = df_pub[mask_tr_a][feats_pub]
        y_tr_a = np.clip(df_pub[mask_tr_a]['Quantity'], 0, None)
        w_tr_a = df_pub[mask_tr_a]['profit_weight']
        X_va_a = df_pub[mask_va_a][feats_pub]
        y_va_a = pd.Series(np.clip(df_pub[mask_va_a]['Quantity'], 0, None)).fillna(0)
        w_va_a = df_pub[mask_va_a]['profit_weight']

        td_a = lgb.Dataset(X_tr_a, label=y_tr_a, weight=w_tr_a)
        vd_a = lgb.Dataset(X_va_a, label=y_va_a, weight=w_va_a, reference=td_a)
        model_lgb_a = lgb.train(Config.LGBM_PARAMS_PUBLIC, td_a, valid_sets=[td_a, vd_a],
                                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])

        cat_params_a = {'loss_function': 'Tweedie:variance_power=1.15', 'iterations': 800,
                        'learning_rate': 0.05, 'depth': 6, 'random_seed': Config.SEED,
                        'thread_count': -1, 'verbose': False}
        model_cat_a = CatBoostRegressor(**cat_params_a)
        model_cat_a.fit(X_tr_a, y_tr_a, cat_features=cat_features, sample_weight=w_tr_a,
                        eval_set=(X_va_a, y_va_a), early_stopping_rounds=50, verbose=False)

        val_preds_pub[indices_va_a] = 0.6 * model_lgb_a.predict(X_va_a) + 0.4 * model_cat_a.predict(X_va_a)
        models_pub['GroupA'] = (model_lgb_a, model_cat_a)

        # --- Nhóm B: Siêu thưa & Hoàn trả nhiều (Cụm 2) ---
        mask_tr_b    = train_mask & (df_pub['cluster'] == 2)
        mask_va_b    = val_mask   & (df_pub['cluster'] == 2)
        indices_va_b = df_pub_val[df_pub_val['cluster'] == 2]['idx_val'].values
        print(f"\n[+] Nhóm B (Công khai: Siêu thưa) - Train: {mask_tr_b.sum()}, Val: {mask_va_b.sum()}")

        X_tr_b = df_pub[mask_tr_b][feats_pub]
        y_tr_b = np.clip(df_pub[mask_tr_b]['Quantity'], 0, None)
        w_tr_b = df_pub[mask_tr_b]['profit_weight']
        X_va_b = df_pub[mask_va_b][feats_pub]
        y_va_b = pd.Series(np.clip(df_pub[mask_va_b]['Quantity'], 0, None)).fillna(0)
        w_va_b = df_pub[mask_va_b]['profit_weight']

        td_b = lgb.Dataset(X_tr_b, label=y_tr_b, weight=w_tr_b)
        vd_b = lgb.Dataset(X_va_b, label=y_va_b, weight=w_va_b, reference=td_b)
        model_lgb_b = lgb.train(PARAMS_B, td_b, valid_sets=[td_b, vd_b],
                                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        val_preds_pub[indices_va_b] = model_lgb_b.predict(X_va_b)
        models_pub['GroupB'] = model_lgb_b

        # Đánh giá Công khai
        vp_df = df_pub[val_mask][['ItemCode', 'Date']].copy()
        vp_df['preds']   = val_preds_pub
        vp_df['cluster'] = vp_df['ItemCode'].map(cluster_map).fillna(0).astype(int)
        vp_df.loc[(vp_df['cluster'] == 2) & (vp_df['preds'] < 0.02), 'preds'] = 0.0
        vp_wide = vp_df.pivot(index='ItemCode', columns='Date', values='preds').reindex(skus_order).values

        evaluator     = FastWRMSSEEvaluator(train_wide, valid_wide, weights_arr)
        best_mult_pub = find_best_multiplier(evaluator, vp_wide)
        score_pub     = evaluator.evaluate(vp_wide * best_mult_pub)
        print(f"\n>> WRMSSE Cục bộ Mô hình Công khai: {score_pub:.5f} (Multiplier: {best_mult_pub:.3f})")

        joblib.dump({'models': models_pub, 'mult': best_mult_pub, 'score': score_pub}, CKPT_PUB)
        print(f"[CHECKPOINT] ✓ Đã lưu checkpoint Công khai tại: {CKPT_PUB}")

    # =========================================================
    # PHASE 2: MÔ HÌNH RIÊNG TƯ (F29-F56, lag >= 56)
    # =========================================================
    print("\n>>> TẠO ĐẶC TRƯNG LAG RIÊNG TƯ (F29-F56)...")
    df_priv = df_base.copy()
    df_priv = create_lag_features_private(df_priv)
    df_priv['profit_weight'] = df_priv['ItemCode'].map(weight_map).fillna(0).astype(np.float32)
    df_priv['cluster']       = df_priv['ItemCode'].map(cluster_map).fillna(0).astype(int)
    feats_priv = _get_lag_cols(df_priv, exclude)

    if os.path.exists(CKPT_PRIV):
        print(f"[CHECKPOINT] Phát hiện checkpoint Riêng tư! Đang tải, bỏ qua huấn luyện...")
        c = joblib.load(CKPT_PRIV)
        models_priv, best_mult_priv = c['models'], c['mult']
        print(f"[CHECKPOINT] ✓ WRMSSE={c['score']:.5f} | Multiplier={best_mult_priv:.3f}")
    else:
        print("\n>>> HUẤN LUYỆN MÔ HÌNH RIÊNG TƯ (F29-F56) <<<")
        train_start2 = max(90, val_start - Config.TRAIN_WINDOW_DAYS)
        train_mask2  = (df_priv['day_int'] < val_start) & (df_priv['day_int'] >= train_start2) & (df_priv['Quantity'].notna())
        val_mask2    = (df_priv['day_int'] >= val_start) & (df_priv['day_int'] <= 1754)

        val_preds_priv = np.zeros(len(df_priv[val_mask2]))
        df_priv_val = df_priv[val_mask2].copy().reset_index(drop=True)
        df_priv_val['idx_val'] = df_priv_val.index
        models_priv   = {}
        cat_features2 = ['cat_proxy'] if 'cat_proxy' in feats_priv else []

        # --- Nhóm A: Standard & Superstars ---
        mask_tr2_a    = train_mask2 & (df_priv['cluster'].isin([0, 1]))
        mask_va2_a    = val_mask2   & (df_priv['cluster'].isin([0, 1]))
        indices_va2_a = df_priv_val[df_priv_val['cluster'].isin([0, 1])]['idx_val'].values
        print(f"\n[+] Nhóm A (Riêng tư: Standard & Superstars) - Train: {mask_tr2_a.sum()}, Val: {mask_va2_a.sum()}")

        X_tr2_a = df_priv[mask_tr2_a][feats_priv]
        y_tr2_a = np.clip(df_priv[mask_tr2_a]['Quantity'], 0, None)
        w_tr2_a = df_priv[mask_tr2_a]['profit_weight']
        X_va2_a = df_priv[mask_va2_a][feats_priv]
        y_va2_a = pd.Series(np.clip(df_priv[mask_va2_a]['Quantity'], 0, None)).fillna(0)
        w_va2_a = df_priv[mask_va2_a]['profit_weight']

        td2_a = lgb.Dataset(X_tr2_a, label=y_tr2_a, weight=w_tr2_a)
        vd2_a = lgb.Dataset(X_va2_a, label=y_va2_a, weight=w_va2_a, reference=td2_a)
        model_lgb2_a = lgb.train(Config.LGBM_PARAMS_PRIVATE, td2_a, valid_sets=[td2_a, vd2_a],
                                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])

        cat_params2_a = {'loss_function': 'Tweedie:variance_power=1.20', 'iterations': 800,
                         'learning_rate': 0.05, 'depth': 6, 'random_seed': Config.SEED,
                         'thread_count': -1, 'verbose': False}
        model_cat2_a = CatBoostRegressor(**cat_params2_a)
        model_cat2_a.fit(X_tr2_a, y_tr2_a, cat_features=cat_features2, sample_weight=w_tr2_a,
                         eval_set=(X_va2_a, y_va2_a), early_stopping_rounds=50, verbose=False)

        val_preds_priv[indices_va2_a] = 0.6 * model_lgb2_a.predict(X_va2_a) + 0.4 * model_cat2_a.predict(X_va2_a)
        models_priv['GroupA'] = (model_lgb2_a, model_cat2_a)

        # --- Nhóm B: Siêu thưa (Cụm 2) ---
        mask_tr2_b    = train_mask2 & (df_priv['cluster'] == 2)
        mask_va2_b    = val_mask2   & (df_priv['cluster'] == 2)
        indices_va2_b = df_priv_val[df_priv_val['cluster'] == 2]['idx_val'].values
        print(f"\n[+] Nhóm B (Riêng tư: Siêu thưa) - Train: {mask_tr2_b.sum()}, Val: {mask_va2_b.sum()}")

        X_tr2_b = df_priv[mask_tr2_b][feats_priv]
        y_tr2_b = np.clip(df_priv[mask_tr2_b]['Quantity'], 0, None)
        w_tr2_b = df_priv[mask_tr2_b]['profit_weight']
        X_va2_b = df_priv[mask_va2_b][feats_priv]
        y_va2_b = pd.Series(np.clip(df_priv[mask_va2_b]['Quantity'], 0, None)).fillna(0)
        w_va2_b = df_priv[mask_va2_b]['profit_weight']

        td2_b = lgb.Dataset(X_tr2_b, label=y_tr2_b, weight=w_tr2_b)
        vd2_b = lgb.Dataset(X_va2_b, label=y_va2_b, weight=w_va2_b, reference=td2_b)
        model_lgb2_b = lgb.train(PARAMS_B, td2_b, valid_sets=[td2_b, vd2_b],
                                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        val_preds_priv[indices_va2_b] = model_lgb2_b.predict(X_va2_b)
        models_priv['GroupB'] = model_lgb2_b

        # Đánh giá Riêng tư
        vp_df2 = df_priv[val_mask2][['ItemCode', 'Date']].copy()
        vp_df2['preds']   = val_preds_priv
        vp_df2['cluster'] = vp_df2['ItemCode'].map(cluster_map).fillna(0).astype(int)
        vp_df2.loc[(vp_df2['cluster'] == 2) & (vp_df2['preds'] < 0.02), 'preds'] = 0.0
        vp_wide2 = vp_df2.pivot(index='ItemCode', columns='Date', values='preds').reindex(skus_order).values

        evaluator2     = FastWRMSSEEvaluator(train_wide, valid_wide, weights_arr)
        best_mult_priv = find_best_multiplier(evaluator2, vp_wide2)
        score_priv     = evaluator2.evaluate(vp_wide2 * best_mult_priv)
        print(f"\n>> WRMSSE Cục bộ Mô hình Riêng tư: {score_priv:.5f} (Multiplier: {best_mult_priv:.3f})")

        joblib.dump({'models': models_priv, 'mult': best_mult_priv, 'score': score_priv}, CKPT_PRIV)
        print(f"[CHECKPOINT] ✓ Đã lưu checkpoint Riêng tư tại: {CKPT_PRIV}")

    return models_pub, feats_pub, df_pub, models_priv, feats_priv, df_priv, best_mult_pub, best_mult_priv


def generate_decentralized_submission(models_pub, feats_pub, df_pub,
                                      models_priv, feats_priv, df_priv,
                                      cluster_map, profit_df, best_mult_pub, best_mult_priv):
    print("\n" + "="*80)
    print("            ĐANG SINH TỆP NỘP BÀI KAGGLE PHI TẬP TRUNG")
    print("="*80)

    # 1. DỰ ĐOÁN CÔNG KHAI (F1..F28)
    test_pub = df_pub[df_pub['day_int'] > 1754].copy()
    test_pub['day_idx'] = test_pub.groupby('ItemCode').cumcount() + 1
    pub_28 = test_pub[test_pub['day_idx'] <= 28].copy()
    pub_28['cluster'] = pub_28['ItemCode'].map(cluster_map).fillna(0).astype(int)
    pub_28['preds']   = 0.0

    mask_a = pub_28['cluster'].isin([0, 1])
    if mask_a.sum() > 0:
        X_test_a = pub_28[mask_a][feats_pub]
        model_lgb, model_cat = models_pub['GroupA']
        pub_28.loc[mask_a, 'preds'] = 0.6 * model_lgb.predict(X_test_a) + 0.4 * model_cat.predict(X_test_a)

    mask_b = pub_28['cluster'] == 2
    if mask_b.sum() > 0:
        X_test_b = pub_28[mask_b][feats_pub]
        pub_28.loc[mask_b, 'preds'] = models_pub['GroupB'].predict(X_test_b)

    pub_28.loc[(pub_28['cluster'] == 2) & (pub_28['preds'] < 0.02), 'preds'] = 0.0
    pub_28['preds'] = np.clip(pub_28['preds'] * best_mult_pub, 0, None)
    pub_28['F']  = pub_28['day_idx'].apply(lambda x: f'F{x}')
    pub_28['id'] = pub_28['ItemCode'].astype(str) + '_validation'
    val_wide = pub_28.pivot(index='id', columns='F', values='preds').reset_index()

    # 2. DỰ ĐOÁN RIÊNG TƯ (F29..F56)
    test_priv = df_priv[df_priv['day_int'] > 1754].copy()
    test_priv['day_idx'] = test_priv.groupby('ItemCode').cumcount() + 1
    priv_56 = test_priv[test_priv['day_idx'] > 28].copy()
    priv_56['cluster'] = priv_56['ItemCode'].map(cluster_map).fillna(0).astype(int)
    priv_56['preds']   = 0.0

    mask_a2 = priv_56['cluster'].isin([0, 1])
    if mask_a2.sum() > 0:
        X_test_a2 = priv_56[mask_a2][feats_priv]
        model_lgb, model_cat = models_priv['GroupA']
        priv_56.loc[mask_a2, 'preds'] = 0.6 * model_lgb.predict(X_test_a2) + 0.4 * model_cat.predict(X_test_a2)

    mask_b2 = priv_56['cluster'] == 2
    if mask_b2.sum() > 0:
        X_test_b2 = priv_56[mask_b2][feats_priv]
        priv_56.loc[mask_b2, 'preds'] = models_priv['GroupB'].predict(X_test_b2)

    priv_56.loc[(priv_56['cluster'] == 2) & (priv_56['preds'] < 0.02), 'preds'] = 0.0
    priv_56['preds'] = np.clip(priv_56['preds'] * best_mult_priv, 0, None)
    priv_56['F']  = (priv_56['day_idx'] - 28).apply(lambda x: f'F{x}')
    priv_56['id'] = priv_56['ItemCode'].astype(str) + '_evaluation'
    eval_wide = priv_56.pivot(index='id', columns='F', values='preds').reset_index()

    # 3. Gộp và xuất file
    sub_wide     = pd.concat([val_wide, eval_wide], ignore_index=True)
    sub_template = pd.read_csv(os.path.join(Config.DATA_DIR, Config.SUB_FILE))
    final_sub    = sub_template[['id']].merge(sub_wide, on='id', how='left')
    final_sub    = final_sub[sub_template.columns]

    f_cols = [f'F{i}' for i in range(1, 29)]
    final_sub[f_cols] = final_sub[f_cols].fillna(0.0)

    # Áp dụng bộ lọc ngày đóng cửa (F2, F9, F16, F23 -> 0)
    print("[INFO] Áp dụng bộ lọc ngày đóng cửa (F2, F9, F16, F23 -> 0)...")
    for f in ['F2', 'F9', 'F16', 'F23']:
        final_sub[f] = 0.0

    output_path = os.path.join(Config.DATA_DIR, "v22_stage2_decentralized.csv")
    final_sub.to_csv(output_path, index=False)
    print(f"[SUCCESS] Đã ghi file nộp bài tại: {output_path}")
    print(final_sub.head(4))


def main():
    print("=" * 80)
    print("   PENGUINS HBAAC 2026 - PIPELINE PHI TẬP TRUNG NHÓM HỖN HỢP")
    print("=" * 80)

    if not os.path.exists(CLUSTERS_PATH):
        print(f"[-] Lỗi: Không tìm thấy {CLUSTERS_PATH}. Hãy chạy step0_clustering.py trước.")
        return

    clusters_df = pd.read_csv(CLUSTERS_PATH)
    cluster_map = clusters_df.set_index('ItemCode')['cluster'].to_dict()

    daily, profit = load_and_aggregate_data()
    full_df = build_dense_grid(daily, profit)
    full_df = create_shared_features(full_df, profit)

    (models_pub, feats_pub, df_pub,
     models_priv, feats_priv, df_priv,
     best_mult_pub, best_mult_priv) = train_decentralized_models(full_df, profit, cluster_map)

    generate_decentralized_submission(
        models_pub, feats_pub, df_pub,
        models_priv, feats_priv, df_priv,
        cluster_map, profit, best_mult_pub, best_mult_priv
    )

if __name__ == '__main__':
    main()
