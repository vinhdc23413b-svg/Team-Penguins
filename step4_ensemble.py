"""
PENGUINS TEAM - KAGGLE GRANDMASTER CHAMPIONS ENSEMBLE BLENDER
------------------------------------------------------------
Author: Penguins Team
Description: Blends our top-tier champion submissions (v7, v2, v12) 
             using a high-precision grid search to find the mathematically 
             optimal weights that minimize WRMSSE.
"""

import os
import sys
import numpy as np
import pandas as pd
warnings_ignored = True

# Tự động định cấu hình sys.path để hỗ trợ chạy độc lập
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from utils_scorer import LocalScorer

def main():
    print("=" * 80)
    print("     PENGUINS TEAM - GRANDMASTER WEIGHTED ENSEMBLE BLENDER")
    print("=" * 80)
    
    try:
        scorer = LocalScorer()
    except Exception as e:
        print(f"Error initializing LocalScorer: {e}")
        return

    data_dir = scorer.data_dir
    
    # Define our top 3 champion submissions
    champions = {
        'v7': 'v22_stage3_optimized_blend.csv',
        'v2': 'v22_stage1_dual_model.csv',
        'v12': 'v22_stage2_decentralized.csv'
    }
    
    # Load and extract validation matrices
    val_preds = {}
    f_cols = [f'F{i}' for i in range(1, 29)]
    
    print("\n[STEP 1] Loading and aligning champion validation forecasts...")
    for name, filename in champions.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            # Try parent directory just in case
            filepath = os.path.join("..", filepath)
            
        if not os.path.exists(filepath):
            print(f"Error: {filename} not found!")
            return
            
        df = pd.read_csv(filepath)
        df_val = df[df['id'].str.endswith('_validation')].copy()
        df_val['ItemCode'] = df_val['id'].str.replace('_validation', '', regex=False)
        df_val = df_val.set_index('ItemCode').reindex(scorer.skus_order).reset_index()
        
        val_preds[name] = df_val[f_cols].values.astype(np.float32)
        print(f" -> Loaded {filename} (Shape: {val_preds[name].shape})")

    # 2. Grid Search over Weights
    print("\n[STEP 2] Running high-precision grid search for optimal blending weights...")
    best_score = 999.0
    best_weights = None
    
    # Generate weight candidates that sum to 1.0
    step = 0.05
    weight_candidates = []
    for w1 in np.arange(0.0, 1.01, step):
        for w2 in np.arange(0.0, 1.01 - w1, step):
            w3 = 1.0 - w1 - w2
            if abs(w1 + w2 + w3 - 1.0) < 1e-5:
                weight_candidates.append((w1, w2, w3))
                
    print(f" -> Searching over {len(weight_candidates)} weight combinations...")
    
    for w1, w2, w3 in weight_candidates:
        # Blended prediction matrix
        blended_preds = w1 * val_preds['v7'] + w2 * val_preds['v2'] + w3 * val_preds['v12']
        blended_preds = np.clip(blended_preds, 0, None)
        
        score = scorer.evaluator.evaluate(blended_preds)
        if score < best_score:
            best_score = score
            best_weights = (w1, w2, w3)
            print(f"  New Best Local WRMSSE: {best_score:.6f} | Weights: v7={w1:.2f}, v2={w2:.2f}, v12={w3:.2f}")

    w1, w2, w3 = best_weights
    print("\n" + "="*80)
    print("             ENSEMBLE BLENDING OPTIMIZATION RESULTS")
    print("="*80)
    print(f" [*] Best Ensembled Local WRMSSE: {best_score:.6f}")
    print(f" [*] Estimated Kaggle Score:       {scorer.calibrate_score(best_score):.5f}")
    print(f" [*] Optimal Weight for v7 (Opt):  {w1:.2f}")
    print(f" [*] Optimal Weight for v2 (Dense):{w2:.2f}")
    print(f" [*] Optimal Weight for v12 (Dec): {w3:.2f}")
    print(f" [*] Previous Single Best (v7):    0.416529")
    print("="*80)

    # 3. Export Champion Submission File
    if best_score < 0.416529:
        print("\nCONGRATULATIONS! Weighted ensemble broke the record!", flush=True)
        print("Generating and exporting final blended submission v18...", flush=True)
        
        # Load raw full files to blend
        df_v7 = pd.read_csv(os.path.join(data_dir, champions['v7']))
        df_v2 = pd.read_csv(os.path.join(data_dir, champions['v2']))
        df_v12 = pd.read_csv(os.path.join(data_dir, champions['v12']))
        
        # Sort and align full submissions to ensure order matches exactly
        df_v7 = df_v7.sort_values('id').reset_index(drop=True)
        df_v2 = df_v2.sort_values('id').reset_index(drop=True)
        df_v12 = df_v12.sort_values('id').reset_index(drop=True)
        
        # Blend the values directly
        blended_full = df_v7.copy()
        
        for col in f_cols:
            blended_full[col] = (
                w1 * df_v7[col].values + 
                w2 * df_v2[col].values + 
                w3 * df_v12[col].values
            )
            # Clip negative predictions to 0
            blended_full[col] = np.clip(blended_full[col].values, 0, None)
            
        output_path = os.path.join(data_dir, "v22_stage4_grandmaster_blend.csv")
        blended_full.to_csv(output_path, index=False)
        print(f"\n[SUCCESS] Exported Champion Grandmaster Blend to: {output_path}")
    else:
        print("\n[INFO] Blending did not beat the individual best of 0.416529.")
        print("We will keep v7 as our gold standard for submissions.")
        output_path = os.path.join(data_dir, "v22_stage4_grandmaster_blend.csv")
        df_v7 = pd.read_csv(os.path.join(data_dir, champions['v7']))
        df_v7.to_csv(output_path, index=False)
        print(f"[INFO] Copied {champions['v7']} to {output_path} to ensure pipeline consistency.")

if __name__ == '__main__':
    main()
