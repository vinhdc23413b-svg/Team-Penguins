"""
PENGUINS TEAM - HIGH-PRECISION BLEND GRID SEARCH
------------------------------------------------
Author: Penguins Team
Description: Performs an ultra-fast grid search on blend weights, magic multipliers,
             and noise-cutting thresholds to find the mathematically optimal submission
             parameters to minimize WRMSSE.
"""

import pandas as pd
import numpy as np
import os
import time
import sys
# Tự động định cấu hình sys.path để hỗ trợ chạy độc lập
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# We can import LocalScorer directly
from utils_scorer import LocalScorer

def main():
    print("=" * 70)
    print("   PENGUINS TEAM - ULTRA-FAST HYPERPARAMETER GRID SEARCH")
    print("=" * 70)
    
    # 1. Initialize the Scorer (takes ~30s, done once)
    start_init = time.time()
    scorer = LocalScorer()
    print(f"-> LocalScorer initialized in {time.time() - start_init:.2f} seconds.")
    
    # 2. Load submission files
    gbdt_path = os.path.join(scorer.data_dir, "v22_stage1_dual_model.csv")
    base_path = os.path.join(scorer.data_dir, "baseline_submission.csv")
    
    if not os.path.exists(gbdt_path) or not os.path.exists(base_path):
        print("[ERROR] Missing required submission files in data/ directory!")
        return
        
    df_gbdt = pd.read_csv(gbdt_path)
    df_base = pd.read_csv(base_path)
    
    # Extract only validation rows and reindex to match the evaluator's SKU order
    f_cols = [f'F{i}' for i in range(1, 29)]
    
    # Process GBDT validation predictions
    df_gbdt_val = df_gbdt[df_gbdt['id'].str.endswith('_validation')].copy()
    df_gbdt_val['ItemCode'] = df_gbdt_val['id'].str.replace('_validation', '', regex=False)
    df_gbdt_val = df_gbdt_val.set_index('ItemCode').reindex(scorer.skus_order).reset_index()
    gbdt_preds = df_gbdt_val[f_cols].values.astype(np.float32)
    
    # Process Baseline validation predictions
    df_base_val = df_base[df_base['id'].str.endswith('_validation')].copy()
    df_base_val['ItemCode'] = df_base_val['id'].str.replace('_validation', '', regex=False)
    df_base_val = df_base_val.set_index('ItemCode').reindex(scorer.skus_order).reset_index()
    base_preds = df_base_val[f_cols].values.astype(np.float32)
    
    print(f"-> GBDT predictions shape: {gbdt_preds.shape}")
    print(f"-> Baseline predictions shape: {base_preds.shape}")
    
    # Verify shape
    if gbdt_preds.shape != (15972, 28) or base_preds.shape != (15972, 28):
        print("[ERROR] Predictions do not match required shape (15972, 28)!")
        return
        
    # 3. Grid Search Space Definition
    print("\nStarting high-precision grid search...")
    start_search = time.time()
    
    best_score = 999.0
    best_params = {}
    
    # Grid parameters
    weights = np.linspace(0.0, 1.0, 51)           # Blend weight for GBDT (0.0 to 1.0, step 0.02)
    multipliers = np.linspace(0.85, 1.05, 41)     # Magic Multiplier (0.85 to 1.05, step 0.005)
    thresholds = np.linspace(0.0, 0.06, 13)       # Zero-threshold (0.0 to 0.06, step 0.005)
    
    total_iterations = len(weights) * len(multipliers) * len(thresholds)
    print(f"Quetting {total_iterations:,} combinations...")
    
    count = 0
    for w in weights:
        # Pre-blend to save operations in the inner loops
        blended_base = w * gbdt_preds + (1.0 - w) * base_preds
        
        for m in multipliers:
            scaled = blended_base * m
            
            for t in thresholds:
                count += 1
                
                # Apply dynamic thresholding
                if t > 0:
                    preds = np.where(scaled < t, 0.0, scaled)
                else:
                    preds = scaled
                    
                # Evaluate using Fast Evaluator (under 0.001 seconds)
                score = scorer.evaluator.evaluate(preds)
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'weight_gbdt': w,
                        'multiplier': m,
                        'threshold': t
                    }
                    print(f"  [Iter {count:,}/{total_iterations:,}] New Best Local WRMSSE: {best_score:.6f} | GBDT Weight: {w:.2f}, Multiplier: {m:.3f}, Threshold: {t:.3f}")
                    
    search_duration = time.time() - start_search
    print(f"\n-> Grid search completed in {search_duration:.2f} seconds ({count / search_duration:.0f} iterations/sec).")
    
    # 4. Show optimized results
    print("\n" + "="*70)
    print("                OPTIMIZED SUBMISSION PARAMETERS")
    print("="*70)
    print(f" [*] Best Local WRMSSE Score:  {best_score:.6f}")
    print(f" [*] Estimated Kaggle Score:    {scorer.calibrate_score(best_score):.5f}")
    print(f" [*] Optimal GBDT Weight:       {best_params['weight_gbdt']:.2f}")
    print(f" [*] Optimal Baseline Weight:   {1.0 - best_params['weight_gbdt']:.2f}")
    print(f" [*] Optimal Magic Multiplier:  {best_params['multiplier']:.3f}")
    print(f" [*] Optimal Noise Threshold:   {best_params['threshold']:.3f}")
    print("="*70)
    
    # 5. Generate and export the ultimate submission file
    print("\nGenerating final optimized submission...")
    opt_w = best_params['weight_gbdt']
    opt_m = best_params['multiplier']
    opt_t = best_params['threshold']
    
    # Blend full GBDT submission with Baseline submission
    final_sub = df_gbdt.copy()
    for col in f_cols:
        # Blend
        blend_col = opt_w * df_gbdt[col] + (1.0 - opt_w) * df_base[col]
        # Multiply
        blend_col = blend_col * opt_m
        # Threshold
        if opt_t > 0:
            blend_col = np.where(blend_col < opt_t, 0.0, blend_col)
        final_sub[col] = blend_col
        
    output_path = os.path.join(scorer.data_dir, "v22_stage3_optimized_blend.csv")
    final_sub.to_csv(output_path, index=False)
    print(f"[SUCCESS] Exported ultimate optimized blend to: {output_path}")

if __name__ == "__main__":
    main()
