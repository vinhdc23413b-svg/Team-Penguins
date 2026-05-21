import os
import gc
import warnings
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
os.environ['LIGHTGBM_VERBOSITY'] = '-1'  # Tắt log nội bộ của LightGBM (bao gồm cảnh báo whitespace tên cột)

# Tự động định cấu hình sys.path để hỗ trợ chạy độc lập
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Import các hàm đặc trưng nâng cao từ utils_features.py
from utils_features import load_raw_data, build_dense_grid, create_shared_features, create_lag_features_public, create_lag_features_private

# Luôn ưu tiên thư mục dữ liệu cục bộ (reproduce_v22/data)
RESOLVED_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, 'data'))

class Config:
    DATA_DIR = RESOLVED_DATA_DIR
    TRAIN_FILE = 'train.csv'
    SUB_FILE = 'sample_submission.csv'
    SEED = 2026
    HORIZON = 56
    
    # Kích thước cửa sổ huấn luyện (số ngày trước validation) để tránh OOM và tăng tốc CPU
    TRAIN_WINDOW_DAYS = 360

    # Model 1 (Public F1-F28): sử dụng lag >= 28
    LGBM_PARAMS_PUBLIC = {
        'objective': 'tweedie', 'tweedie_variance_power': 1.15,
        'metric': 'rmse', 'learning_rate': 0.05, 'num_leaves': 127,
        'min_data_in_leaf': 150, 'feature_fraction': 0.75,
        'bagging_fraction': 0.75, 'bagging_freq': 1,
        'n_estimators': 1200, 'random_state': SEED, 'n_jobs': -1,
    }
    # Model 2 (Private F29-F56): sử dụng lag >= 56
    LGBM_PARAMS_PRIVATE = {
        'objective': 'tweedie', 'tweedie_variance_power': 1.20,
        'metric': 'rmse', 'learning_rate': 0.05, 'num_leaves': 127,
        'min_data_in_leaf': 200, 'feature_fraction': 0.70,
        'bagging_fraction': 0.70, 'bagging_freq': 1,
        'n_estimators': 1200, 'random_state': SEED, 'n_jobs': -1,
    }
    MAGIC_MULTIPLIER = 0.99


# =====================================================================
# PART 1: ULTRA-FAST WRMSSE EVALUATOR
# =====================================================================
class FastWRMSSEEvaluator:
    """
    Ultra-Fast Flat per-SKU WRMSSE Evaluator for HBAAC 2026 - Final Check.
    Calculates WRMSSE directly on the 15,972 SKUs using custom profit weights.
    
    Mathematical details:
        WRMSSE = Sum_{i=1}^{15972} (w_i * RMSSE_i)
        Scale denominator is computed starting from the first non-zero sales day (active period).
    """
    def __init__(self, train_sales: np.ndarray, valid_sales: np.ndarray, weights: np.ndarray):
        """
        Initialize the evaluator.
        
        Args:
            train_sales: np.ndarray of shape (15972, n_days) containing historical daily sales.
            valid_sales: np.ndarray of shape (15972, 28) containing actual validation sales.
            weights: np.ndarray of shape (15972,) containing normalized profit weights.
        """
        print("Khởi tạo Fast WRMSSE Evaluator...")
        self.valid_sales = valid_sales.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.h = 28 # Forecasting horizon
        
        # 1. Tính toán hệ số scale (mẫu số của RMSSE) cho từng SKU
        print("Calculating scaling factors based on active sales history...")
        self.scale = self._calculate_scale(train_sales)
        print("FastWRMSSEEvaluator initialized successfully!")

    def _calculate_scale(self, train_sales: np.ndarray) -> np.ndarray:
        """
        Computes the scale (denominator of RMSSE) for each of the 15,972 SKUs.
        Calculated as the mean squared difference of consecutive days starting from the
        first non-zero sales day of that specific series.
        """
        has_sales = (train_sales != 0).any(axis=1)
        first_non_zero = (train_sales != 0).argmax(axis=1)
        
        diffs_sq = np.diff(train_sales, axis=1) ** 2
        cumsum_padded = np.zeros((train_sales.shape[0], train_sales.shape[1]), dtype=np.float32)
        np.cumsum(diffs_sq, axis=1, out=cumsum_padded[:, 1:])
        
        N = train_sales.shape[1]
        row_indices = np.arange(len(train_sales))
        sum_active = cumsum_padded[:, -1] - cumsum_padded[row_indices, first_non_zero]
        num_elements = (N - 1) - first_non_zero
        
        scale = np.ones(len(train_sales), dtype=np.float32)
        valid_mask = has_sales & (num_elements > 0)
        
        mean_diff_sq = np.zeros_like(scale)
        mean_diff_sq[valid_mask] = sum_active[valid_mask] / num_elements[valid_mask]
        
        return np.where(valid_mask & (mean_diff_sq > 0), mean_diff_sq, 1.0)

    def evaluate(self, preds: np.ndarray) -> float:
        """
        Evaluate predictions against actual validation sales.
        
        Args:
            preds: np.ndarray of shape (15972, 28) containing predictions.
            
        Returns:
            float: The calculated WRMSSE score.
        """
        # Clip negative predictions to 0
        preds_clipped = np.clip(preds, 0, None)
        
        # Calculate Mean Squared Error per SKU
        forecast_mse = np.mean((self.valid_sales - preds_clipped) ** 2, axis=1)
        
        # Calculate RMSSE
        rmsse = np.sqrt(forecast_mse / self.scale)
        
        # Apply profit-share weights and sum
        wrmsse = np.sum(self.weights * rmsse)
        return float(wrmsse)


# =====================================================================
# PART 2: ADVANCED FEATURE IMPORT - DUP REMOVED
# =====================================================================

def load_and_aggregate_data():
    train_path = os.path.join(Config.DATA_DIR, Config.TRAIN_FILE)
    return load_raw_data(train_path)

def find_best_multiplier(evaluator, val_preds_wide):
    best_score = float('inf')
    best_mult = 1.0
    multipliers = np.arange(0.30, 1.21, 0.01)
    for mult in multipliers:
        preds_adjusted = val_preds_wide * mult
        score = evaluator.evaluate(preds_adjusted)
        if score < best_score:
            best_score = score
            best_mult = mult
    return best_mult

def _get_lag_cols(df, exclude_base):
    return [c for c in df.columns if c not in exclude_base]

def find_best_multiplier_and_threshold(evaluator, val_preds_wide):
    best_score = float('inf')
    best_mult = 1.0
    best_thresh = 0.0
    
    multipliers = np.arange(0.30, 1.21, 0.05)      # coarse step
    thresholds = np.linspace(0.0, 0.40, 21)         # coarse step
    
    print("\nQuet tim he so Multiplier & Noise Threshold toi uu offline...")
    for mult in multipliers:
        for thresh in thresholds:
            preds_adjusted = val_preds_wide * mult
            preds_adjusted = np.where(preds_adjusted < thresh, 0.0, preds_adjusted)
            score = evaluator.evaluate(preds_adjusted)
            if score < best_score:
                best_score = score
                best_mult = mult
                best_thresh = thresh
                
    # Fine-tune grid search
    fine_multipliers = np.arange(max(0.30, best_mult - 0.04), min(1.20, best_mult + 0.04), 0.01)
    fine_thresholds = np.linspace(max(0.0, best_thresh - 0.02), min(0.50, best_thresh + 0.02), 11)
    
    for mult in fine_multipliers:
        for thresh in fine_thresholds:
            preds_adjusted = val_preds_wide * mult
            preds_adjusted = np.where(preds_adjusted < thresh, 0.0, preds_adjusted)
            score = evaluator.evaluate(preds_adjusted)
            if score < best_score:
                best_score = score
                best_mult = mult
                best_thresh = thresh
                
    print(f"  >> Parameter tot nhat: Multiplier={best_mult:.3f}, Threshold={best_thresh:.3f} (WRMSSE: {best_score:.5f})")
    return best_mult, best_thresh

def train_dual_models(df_base, profit_df):
    print("\n--- Huan Luyen Dual-Model (LightGBM + CatBoost Blend) ---")
    exclude = ['Date', 'Quantity', 'day_int', 'profit_weight', 'ItemCode', 'ReturnQty', 'GiftQty']

    val_start = 1754 - 28 + 1  # Ngay 1727
    skus_order = profit_df['ItemCode'].values
    weights_arr = profit_df['weight'].values

    weight_map = profit_df.set_index('ItemCode')['weight'].to_dict()

    # ---- MODEL 1: PUBLIC (lag >= 28) ----
    print("\n  == Model 1: PUBLIC (F1..F28, lags >= 28) ==")
    df_pub = df_base.copy()
    df_pub = create_lag_features_public(df_pub)
    df_pub['profit_weight'] = df_pub['ItemCode'].map(weight_map).fillna(0).astype(np.float32)

    feats_pub = _get_lag_cols(df_pub, exclude)
    print(f"  Features: {len(feats_pub)}")
    cat_features = ['cat_proxy'] if 'cat_proxy' in feats_pub else []

    train_start = max(60, val_start - Config.TRAIN_WINDOW_DAYS)
    print(f"  Train Range: Days {train_start} to {val_start-1} (Window size: {val_start - train_start} days)")
    train_mask = (df_pub['day_int'] < val_start) & (df_pub['day_int'] >= train_start) & (df_pub['Quantity'].notna())
    val_mask = (df_pub['day_int'] >= val_start) & (df_pub['day_int'] <= 1754)

    X_tr = df_pub[train_mask][feats_pub]
    y_tr = np.clip(df_pub[train_mask]['Quantity'], 0.0, None)
    w_tr = df_pub[train_mask]['profit_weight']
    X_va = df_pub[val_mask][feats_pub]
    y_va = np.clip(df_pub[val_mask]['Quantity'], 0.0, None)
    y_va = pd.Series(y_va).fillna(0.0)
    w_va = df_pub[val_mask]['profit_weight']

    # 1. LightGBM Public
    print("\n  [LGBM] Training LightGBM Public...")
    td = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    vd = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=td)
    model_pub_lgb = lgb.train(
        Config.LGBM_PARAMS_PUBLIC, td,
        valid_sets=[td, vd],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # 2. CatBoost Public
    print("\n  [CatBoost] Training CatBoost Public (objective=Tweedie:1.15)...")
    cat_params_pub = {
        'loss_function': 'Tweedie:variance_power=1.15',
        'iterations': 150,
        'learning_rate': 0.15,
        'depth': 6,
        'random_seed': Config.SEED,
        'thread_count': -1,
        'verbose': 50
    }
    model_pub_cat = CatBoostRegressor(**cat_params_pub)
    model_pub_cat.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        sample_weight=w_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=30,
        verbose=50
    )

    # Blended predictions for Public Model
    print("\n  Blending Public Model predictions (60% LGBM + 40% CatBoost)...")
    val_preds_lgb = model_pub_lgb.predict(X_va)
    val_preds_cat = model_pub_cat.predict(X_va)
    val_preds = 0.6 * val_preds_lgb + 0.4 * val_preds_cat
    val_preds = np.where(df_pub[val_mask]['days_since_first_sale'] < 0, 0.0, val_preds)

    vp_df = df_pub[val_mask][['ItemCode', 'Date']].copy()
    vp_df['preds'] = val_preds
    vp_wide = vp_df.pivot(index='ItemCode', columns='Date', values='preds').reindex(skus_order).values

    train_wide = np.clip(df_pub[df_pub['day_int'] < val_start].pivot(
        index='ItemCode', columns='Date', values='Quantity'
    ).reindex(skus_order).values, 0.0, None)
    train_wide = np.nan_to_num(train_wide, nan=0.0)
    valid_wide = np.clip(df_pub[val_mask].pivot(
        index='ItemCode', columns='Date', values='Quantity'
    ).reindex(skus_order).values, 0.0, None)
    valid_wide = np.nan_to_num(valid_wide, nan=0.0)

    evaluator = FastWRMSSEEvaluator(train_wide, valid_wide, weights_arr)
    
    # Dynamic Magic Multiplier & Threshold Search for Public
    best_mult_pub, best_thresh_pub = find_best_multiplier_and_threshold(evaluator, vp_wide)
    score_pub = evaluator.evaluate(np.where(vp_wide * best_mult_pub < best_thresh_pub, 0.0, vp_wide * best_mult_pub))
    print(f"\n  >> PUBLIC MODEL LOCAL WRMSSE (Blended & Optimized): {score_pub:.5f}")

    # ---- MODEL 2: PRIVATE (lag >= 56) ----
    print("\n  == Model 2: PRIVATE (F29..F56, lags >= 56) ==")
    df_priv = df_base.copy()
    df_priv = create_lag_features_private(df_priv)
    df_priv['profit_weight'] = df_priv['ItemCode'].map(weight_map).fillna(0).astype(np.float32)

    feats_priv = _get_lag_cols(df_priv, exclude)
    print(f"  Features: {len(feats_priv)}")

    train_start2 = max(90, val_start - Config.TRAIN_WINDOW_DAYS)
    print(f"  Train Range: Days {train_start2} to {val_start-1} (Window size: {val_start - train_start2} days)")
    train_mask2 = (df_priv['day_int'] < val_start) & (df_priv['day_int'] >= train_start2) & (df_priv['Quantity'].notna())
    val_mask2 = (df_priv['day_int'] >= val_start) & (df_priv['day_int'] <= 1754)

    X_tr2 = df_priv[train_mask2][feats_priv]
    y_tr2 = np.clip(df_priv[train_mask2]['Quantity'], 0.0, None)
    w_tr2 = df_priv[train_mask2]['profit_weight']
    X_va2 = df_priv[val_mask2][feats_priv]
    y_va2 = np.clip(df_priv[val_mask2]['Quantity'], 0.0, None)
    y_va2 = pd.Series(y_va2).fillna(0.0)
    w_va2 = df_priv[val_mask2]['profit_weight']

    # 1. LightGBM Private
    print("\n  [LGBM] Training LightGBM Private...")
    td2 = lgb.Dataset(X_tr2, label=y_tr2, weight=w_tr2)
    vd2 = lgb.Dataset(X_va2, label=y_va2, weight=w_va2, reference=td2)
    model_priv_lgb = lgb.train(
        Config.LGBM_PARAMS_PRIVATE, td2,
        valid_sets=[td2, vd2],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # 2. CatBoost Private
    print("\n  [CatBoost] Training CatBoost Private (objective=Tweedie:1.20)...")
    cat_params_priv = {
        'loss_function': 'Tweedie:variance_power=1.20',
        'iterations': 150,
        'learning_rate': 0.15,
        'depth': 6,
        'random_seed': Config.SEED,
        'thread_count': -1,
        'verbose': 50
    }
    model_priv_cat = CatBoostRegressor(**cat_params_priv)
    model_priv_cat.fit(
        X_tr2, y_tr2,
        cat_features=cat_features,
        sample_weight=w_tr2,
        eval_set=(X_va2, y_va2),
        early_stopping_rounds=30,
        verbose=50
    )

    # Blended predictions for Private Model
    print("\n  Blending Private Model predictions (60% LGBM + 40% CatBoost)...")
    val_preds2_lgb = model_priv_lgb.predict(X_va2)
    val_preds2_cat = model_priv_cat.predict(X_va2)
    val_preds2 = 0.6 * val_preds2_lgb + 0.4 * val_preds2_cat
    val_preds2 = np.where(df_priv[val_mask2]['days_since_first_sale'] < 0, 0.0, val_preds2)

    vp_df2 = df_priv[val_mask2][['ItemCode', 'Date']].copy()
    vp_df2['preds'] = val_preds2
    vp_wide2 = vp_df2.pivot(index='ItemCode', columns='Date', values='preds').reindex(skus_order).values

    evaluator2 = FastWRMSSEEvaluator(train_wide, valid_wide, weights_arr)
    
    # Dynamic Magic Multiplier & Threshold Search for Private
    best_mult_priv, best_thresh_priv = find_best_multiplier_and_threshold(evaluator2, vp_wide2)
    score_priv = evaluator2.evaluate(np.where(vp_wide2 * best_mult_priv < best_thresh_priv, 0.0, vp_wide2 * best_mult_priv))
    print(f"\n  >> PRIVATE MODEL LOCAL WRMSSE (Blended & Optimized): {score_priv:.5f}")

    print(f"\n{'='*55}")
    print(f"  PUBLIC  WRMSSE: {score_pub:.5f} (Multiplier: {best_mult_pub:.3f}, Thresh: {best_thresh_pub:.3f})")
    print(f"  PRIVATE WRMSSE: {score_priv:.5f} (Multiplier: {best_mult_priv:.3f}, Thresh: {best_thresh_priv:.3f})")
    print(f"{'='*55}")

    return (model_pub_lgb, model_pub_cat, feats_pub, df_pub, 
            model_priv_lgb, model_priv_cat, feats_priv, df_priv, 
            best_mult_pub, best_thresh_pub, best_mult_priv, best_thresh_priv)

def get_next_version(data_dir: str) -> int:
    import re
    if not os.path.exists(data_dir):
        return 1
    max_v = 0
    for filename in os.listdir(data_dir):
        match = re.match(r"submission_v(\d+)", filename)
        if match:
            v = int(match.group(1))
            if v > max_v:
                max_v = v
    return max_v + 1

def run_submission(model_pub_lgb, model_pub_cat, feats_pub, df_pub,
                   model_priv_lgb, model_priv_cat, feats_priv, df_priv, 
                   profit_df, best_mult_pub, best_thresh_pub, best_mult_priv, best_thresh_priv):
    print("\n--- Sinh Tep Nop Bai Kaggle (Dual-Model Blended) ---")

    # --- Public predictions (F1..F28) tu Model 1 ---
    test_pub = df_pub[df_pub['day_int'] > 1754].copy()
    test_pub['day_idx'] = test_pub.groupby('ItemCode').cumcount() + 1
    pub_28 = test_pub[test_pub['day_idx'] <= 28].copy()
    
    preds_lgb_pub = model_pub_lgb.predict(pub_28[feats_pub])
    preds_cat_pub = model_pub_cat.predict(pub_28[feats_pub])
    pub_28['preds'] = 0.6 * preds_lgb_pub + 0.4 * preds_cat_pub
    pub_28['preds'] = np.where(pub_28['days_since_first_sale'] < 0, 0.0, pub_28['preds'])
    pub_28['preds'] = pub_28['preds'] * best_mult_pub
    pub_28['preds'] = np.clip(np.where(pub_28['preds'] < best_thresh_pub, 0.0, pub_28['preds']), 0, None)
    pub_28['F'] = pub_28['day_idx'].apply(lambda x: f'F{x}')
    pub_28['id'] = pub_28['ItemCode'].astype(str) + '_validation'
    val_wide = pub_28.pivot(index='id', columns='F', values='preds').reset_index()

    # --- Private predictions (F29..F56) tu Model 2 ---
    test_priv = df_priv[df_priv['day_int'] > 1754].copy()
    test_priv['day_idx'] = test_priv.groupby('ItemCode').cumcount() + 1
    priv_56 = test_priv[test_priv['day_idx'] > 28].copy()
    
    preds_lgb_priv = model_priv_lgb.predict(priv_56[feats_priv])
    preds_cat_priv = model_priv_cat.predict(priv_56[feats_priv])
    priv_56['preds'] = 0.6 * preds_lgb_priv + 0.4 * preds_cat_priv
    priv_56['preds'] = np.where(priv_56['days_since_first_sale'] < 0, 0.0, priv_56['preds'])
    priv_56['preds'] = priv_56['preds'] * best_mult_priv
    priv_56['preds'] = np.clip(np.where(priv_56['preds'] < best_thresh_priv, 0.0, priv_56['preds']), 0, None)
    priv_56['F'] = (priv_56['day_idx'] - 28).apply(lambda x: f'F{x}')
    priv_56['id'] = priv_56['ItemCode'].astype(str) + '_evaluation'
    eval_wide = priv_56.pivot(index='id', columns='F', values='preds').reset_index()

    # Merge and match with sample_submission.csv
    sub_wide = pd.concat([val_wide, eval_wide], ignore_index=True)
    sub_template = pd.read_csv(os.path.join(Config.DATA_DIR, Config.SUB_FILE))
    final_sub = sub_template[['id']].merge(sub_wide, on='id', how='left')
    
    # Ép thứ tự các cột trùng khớp 100% với sample_submission.csv để tránh lỗi sắp xếp chữ cái (F1, F10, F11...)
    final_sub = final_sub[sub_template.columns]
    
    f_cols = [f'F{i}' for i in range(1, 29)]
    final_sub[f_cols] = final_sub[f_cols].fillna(0.0)

    # Save to data/ with clean name
    active_file = os.path.join(Config.DATA_DIR, "v22_stage1_dual_model.csv")
    final_sub.to_csv(active_file, index=False)
    print(f"-> SAVED: {active_file}")
    
    print(f"Xuat thanh cong ({final_sub.shape[0]} dong)")
    print(final_sub.head(4))

    print(f"[SUCCESS] Xuất thành công ({final_sub.shape[0]} dòng). Đã lưu tại: {active_file}")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    print("PENGUINS HBAAC 2026 - DUAL MODEL PIPELINE (GOOGLE COLAB VERSION)")
    print("=" * 55)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        train_path = os.path.join(Config.DATA_DIR, Config.TRAIN_FILE)
        daily, profit = load_raw_data(train_path)
        full_df = build_dense_grid(daily, profit, horizon=Config.HORIZON)
        full_df = create_shared_features(full_df, profit)

        (model_pub_lgb, model_pub_cat, feats_pub, df_pub,
         model_priv_lgb, model_priv_cat, feats_priv, df_priv,
         best_mult_pub, best_thresh_pub, best_mult_priv, best_thresh_priv) = train_dual_models(full_df, profit)

        run_submission(model_pub_lgb, model_pub_cat, feats_pub, df_pub,
                       model_priv_lgb, model_priv_cat, feats_priv, df_priv, 
                       profit, best_mult_pub, best_thresh_pub, best_mult_priv, best_thresh_priv)

        print("\nPipeline complete! Penguins ready for Leaderboard!")
    except FileNotFoundError as e:
        print(f"\n[LUU Y]: {e}")
        print("Pipeline OK! Tha du lieu vao data/ la san sang chay.")