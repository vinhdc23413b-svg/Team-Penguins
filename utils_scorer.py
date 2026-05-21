import os
import json
import argparse
import numpy as np
import pandas as pd

class FastWRMSSEEvaluator:
    """
    Ultra-Fast Flat per-SKU WRMSSE Evaluator for HBAAC 2026.
    Calculates WRMSSE directly on the 15,972 SKUs using custom profit weights.
    """
    def __init__(self, train_sales: np.ndarray, valid_sales: np.ndarray, weights: np.ndarray):
        self.valid_sales = valid_sales.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.h = 28
        self.scale = self._calculate_scale(train_sales)

    def _calculate_scale(self, train_sales: np.ndarray) -> np.ndarray:
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
        preds_clipped = np.clip(preds, 0, None)
        forecast_mse = np.mean((self.valid_sales - preds_clipped) ** 2, axis=1)
        rmsse = np.sqrt(forecast_mse / self.scale)
        wrmsse = np.sum(self.weights * rmsse)
        return float(wrmsse)

class LocalScorer:
    """
    Self-Calibrating Local Validation Scorer for Penguins HBAAC 2026.
    Computes exact local WRMSSE on the historical validation window (days 1727-1754)
    and estimates the Kaggle Public & Private Leaderboard scores using a high-precision quadratic calibration model.
    """
    def __init__(self, data_dir='data', train_file='train.csv'):
        # Luôn ưu tiên thư mục dữ liệu cục bộ (reproduce_v22/data)
        local_train = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', train_file))
        parent_train = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', train_file))
        if os.path.exists(local_train):
            self.train_path = local_train
            self.data_dir = os.path.dirname(local_train)
        elif os.path.exists(parent_train):
            self.train_path = parent_train
            self.data_dir = os.path.dirname(parent_train)
        else:
            self.data_dir = data_dir
            self.train_path = os.path.join(data_dir, train_file)
            
        self.train_sales = None
        self.valid_sales = None
        self.weights = None
        self.skus_order = None
        self.evaluator = None
        
        self._prepare_data()

    def _prepare_data(self):
        cache_path = os.path.join(self.data_dir, 'local_scorer_cache.npz')
        
        use_cache = False
        if os.path.exists(cache_path):
            train_mtime = os.path.getmtime(self.train_path)
            cache_mtime = os.path.getmtime(cache_path)
            if cache_mtime > train_mtime:
                use_cache = True
                
        if use_cache:
            print("[INFO] Dang doc du lieu danh gia local da duoc chuan bi tu cache (NPZ)...")
            try:
                with np.load(cache_path, allow_pickle=True) as data:
                    self.train_sales = data['train_sales']
                    self.valid_sales = data['valid_sales']
                    self.weights = data['weights']
                    self.skus_order = data['skus_order']
                self.evaluator = FastWRMSSEEvaluator(self.train_sales, self.valid_sales, self.weights)
                print("[SUCCESS] Da load cache va khoi tao Evaluator thanh cong!")
                return
            except Exception as e:
                print(f"[WARNING] Loi khi doc cache: {e}. Tien hanh tinh toan lai...")
        
        print("[INFO] Dang chuan bi du lieu danh gia local (Days 1727 - 1754)...")
        raw = pd.read_csv(self.train_path)
        
        # Clean numerical columns
        for col in ['UnitPrice', 'Unit Cost', 'SalesAmount', 'Cost Amount']:
            if col in raw.columns:
                if pd.api.types.is_numeric_dtype(raw[col]):
                    raw[col] = raw[col].astype(float)
                else:
                    raw[col] = raw[col].astype(str).str.replace(',', '.', regex=False).astype(float)
        
        # Compute weights exactly as in the training pipeline (Gross Profit)
        raw['Profit'] = raw['SalesAmount'] - raw['Cost Amount']
        profit_df = raw.groupby('ItemCode')['Profit'].sum().reset_index()
        profit_df['Profit'] = np.clip(profit_df['Profit'], 0, None)
        total_profit = profit_df['Profit'].sum()
        profit_df['weight'] = profit_df['Profit'] / (total_profit if total_profit > 0 else 1.0)
        
        profit_df = profit_df.sort_values('ItemCode').reset_index(drop=True)
        self.skus_order = profit_df['ItemCode'].values
        self.weights = profit_df['weight'].values
        
        # Create daily aggregated sales
        raw_gross = raw[raw['Quantity'] > 0]
        daily = raw_gross.groupby(['Date', 'ItemCode'])['Quantity'].sum().reset_index()
        
        all_dates = pd.date_range(start='2020-11-17', end='2025-09-05')
        all_dates_str = all_dates.strftime('%Y-%m-%d')
        
        daily_wide = daily.pivot(index='ItemCode', columns='Date', values='Quantity')
        daily_wide = daily_wide.reindex(index=self.skus_order, columns=all_dates_str, fill_value=0.0).fillna(0.0)
        sales_wide = np.clip(daily_wide.values, 0.0, None)
        
        self.train_sales = sales_wide[:, :1726]
        self.valid_sales = sales_wide[:, 1726:1754]
        
        try:
            np.savez(cache_path, 
                     train_sales=self.train_sales.astype(np.float32), 
                     valid_sales=self.valid_sales.astype(np.float32), 
                     weights=self.weights.astype(np.float32), 
                     skus_order=self.skus_order)
            print("[INFO] Da luu du lieu chuan bi vao cache.")
        except Exception as e:
            print(f"[WARNING] Khong the luu cache: {e}")
            
        self.evaluator = FastWRMSSEEvaluator(self.train_sales, self.valid_sales, self.weights)
        print("[SUCCESS] Da khoi tao Evaluator thanh cong!")

    def calibrate_score(self, local_score: float) -> float:
        """
        Applies the high-precision quadratic calibration model to map Local Score to Estimated Kaggle Score.
        Formula: Kaggle = 0.136847 * Local^2 + 0.782040 * Local + 0.149696
        """
        return 0.136847 * (local_score ** 2) + 0.782040 * local_score + 0.149696

    def score(self, val_preds: np.ndarray) -> tuple:
        if val_preds.shape != (15972, 28):
            raise ValueError(f"Du bao phai co kich thuoc (15972, 28), nhung nhan duoc {val_preds.shape}!")
            
        local_score = self.evaluator.evaluate(val_preds)
        estimated_kaggle = self.calibrate_score(local_score)
        return local_score, estimated_kaggle

def run_cli():
    parser = argparse.ArgumentParser(description="Cong cu cham diem Local & Tu dong uoc tinh diem Kaggle Public/Private cho Penguins Team")
    parser.add_argument('--preds', type=str, help='Duong dan toi file csv chua predictions')
    args = parser.parse_args()
    
    if args.preds is not None:
        scorer = LocalScorer()
        print(f"\n[INFO] Dang doc va phan tich file du bao: {args.preds}...")
        if args.preds.endswith('.csv'):
            df = pd.read_csv(args.preds)
            f_cols = [f'F{i}' for i in range(1, 29)]
            
            df_val = pd.DataFrame()
            if 'id' in df.columns:
                df_val = df[df['id'].str.endswith('_validation')].copy()
                df_val['ItemCode'] = df_val['id'].str.replace('_validation', '', regex=False)
            elif 'ItemCode' in df.columns:
                df_val = df[df['ItemCode'].str.endswith('_validation')].copy()
                df_val['ItemCode'] = df_val['ItemCode'].str.replace('_validation', '', regex=False)
            
            df_eval = pd.DataFrame()
            if 'id' in df.columns:
                df_eval = df[df['id'].str.endswith('_evaluation')].copy()
                df_eval['ItemCode'] = df_eval['id'].str.replace('_evaluation', '', regex=False)
            elif 'ItemCode' in df.columns:
                df_eval = df[df['ItemCode'].str.endswith('_evaluation')].copy()
                df_eval['ItemCode'] = df_eval['ItemCode'].str.replace('_evaluation', '', regex=False)
                
            if len(df_val) == 0 and len(df_eval) == 0:
                if len(df) == 15972:
                    df_val = df.copy()
                    if 'id' in df_val.columns:
                        df_val['ItemCode'] = df_val['id'].str.replace('_validation', '', regex=False)
                    elif 'ItemCode' in df_val.columns:
                        df_val['ItemCode'] = df_val['ItemCode'].str.replace('_validation', '', regex=False)
                    else:
                        df_val['ItemCode'] = scorer.skus_order
                else:
                    print("[WARNING] File khong co hau to _validation hay _evaluation. Tu dong gia dinh 15,972 dong dau tien la Validation.")
                    df_val = df.iloc[:15972].copy()
                    df_val['ItemCode'] = scorer.skus_order

            val_preds = None
            if len(df_val) > 0:
                df_val = df_val.set_index('ItemCode').reindex(scorer.skus_order).reset_index()
                val_preds = df_val[f_cols].values
                
            eval_preds = None
            if len(df_eval) > 0:
                df_eval = df_eval.set_index('ItemCode').reindex(scorer.skus_order).reset_index()
                eval_preds = df_eval[f_cols].values
        else:
            raise ValueError("Dinh dang file khong ho tro! Vui long truyen file .csv")
            
        print("\n" + "="*70)
        print(f"             KET QUA DANH GIA TUONG DONG KAGGLE (PHI TUYEN)")
        print("="*70)
        
        if val_preds is not None:
            local_val, est_pub = scorer.score(val_preds)
            print(f" [*] PHAN PUBLIC (F1..F28 - Validation):")
            print(f"    * Diem Local WRMSSE:          {local_val:.6f}")
            print(f"    * DIEM KAGGLE PUBLIC DU DOAN: {est_pub:.5f}")
            print("-"*70)
        else:
            print(" [*] PHAN PUBLIC (F1..F28): [Khong tim thay du lieu validation]")
            print("-"*70)
            
        if eval_preds is not None:
            local_eval, est_priv = scorer.score(eval_preds)
            print(f" [*] PHAN PRIVATE (F29..F56 - Evaluation):")
            print(f"    * Diem Local WRMSSE:           {local_eval:.6f}")
            print(f"    * DIEM KAGGLE PRIVATE DU DOAN: {est_priv:.5f}")
            print("="*70)
        else:
            print(" [*] PHAN PRIVATE (F29..F56): [WARNING] File khong co phan Evaluation!")
            print("="*70)
    else:
        print("Vui long truyen --preds <file_submission.csv> de bat dau danh gia.")

if __name__ == '__main__':
    run_cli()
