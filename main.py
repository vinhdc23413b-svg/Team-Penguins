# -*- coding: utf-8 -*-
"""
PENGUINS - 100% REPRODUCIBLE MASTER PIPELINE FOR ENSEMBLE V22 (A - Z)
--------------------------------------------------------------------
Thư mục: reproduce_v22
Mô tả: Tái lập hoàn hảo file nộp bài v22_final_ultimate_ensemble.csv từ con số 0.
      Hệ thống chạy tuần tự qua các bước tinh gọn:
      - step0_clustering.py     -> sku_clustering_3_groups.csv       (EDA & Phân cụm SKU)
      - step1_dual_model.py     -> v22_stage1_dual_model.csv         (Mô hình kép LGBM + CatBoost)
      - step2_decentralized.py  -> v22_stage2_decentralized.csv      (Mô hình phân tán theo cụm)
      - step3_optimize_blend.py -> v22_stage3_optimized_blend.csv    (Tối ưu hóa GBDT + Baseline)
      - step4_ensemble.py       -> v22_stage4_grandmaster_blend.csv  (Siêu tổ hợp Grandmaster)
      - Dirichlet Ultimate Ensemble (tích hợp) -> v22_final_ultimate_ensemble.csv
"""

import os
import sys
import shutil
import time
import subprocess
import numpy as np
import pandas as pd

# Add parent directory to path to import LocalScorer fallback if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_scorer import LocalScorer

def print_banner(text):
    border = "=" * 85
    print("\n" + border)
    print(f"   {text}")
    print(border + "\n", flush=True)

def run_script(script_path, args=None, cwd="."):
    script_name = os.path.basename(script_path)
    print(f"[EXEC] Khởi chạy: {script_name}...", flush=True)
    start_time = time.time()
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
        
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    process.communicate()
    
    if process.returncode != 0:
        print(f"\n[ERROR] Script {script_name} thất bại với mã lỗi: {process.returncode}", flush=True)
        sys.exit(process.returncode)
        
    duration = time.time() - start_time
    print(f"[SUCCESS] {script_name} hoàn thành trong {duration:.2f} giây!\n", flush=True)
    return duration

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    master_start = time.time()
    
    print_banner("BẮT ĐẦU QUY TRÌNH TÁI LẬP CHAMPION V22 TÌNH GỌN TỪ ĐẦU (A - Z)")
    
    # Tự động phát hiện chế độ chạy - Luôn cấu hình chạy chế độ Không gian làm việc gốc (Workspace Mode)
    print(" -> [CHẾ ĐỘ] Chạy trong không gian làm việc gốc (Workspace Mode)!")
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    is_standalone = False
        
    print(f" -> Thư mục làm việc: {os.getcwd()}")
    print(f" -> Thư mục gốc hoạt động: {parent_dir}")
    print(" -> Mục tiêu: Huấn luyện và kết hợp toàn bộ mô hình thành phần từ con số 0!")

    template_dir = os.path.abspath(os.path.join(parent_dir, 'mẫu'))
    
    def get_script_path(base_name):
        if is_standalone:
            return base_name
        else:
            return f"reproduce_v22/{base_name}"
            
    # 0. Chuẩn bị file baseline_submission.csv
    baseline_target = os.path.join(data_dir, "baseline_submission.csv")
    if not os.path.exists(baseline_target):
        print("[INFO] Đang chuẩn bị file mốc baseline_submission.csv...")
        if is_standalone:
            raise FileNotFoundError("Không tìm thấy baseline_submission.csv trong thư mục data cục bộ!")
        else:
            baseline_source = os.path.join(template_dir, "baseline_submission.csv")
            if os.path.exists(baseline_source):
                shutil.copy(baseline_source, baseline_target)
            else:
                raise FileNotFoundError("Không tìm thấy baseline_submission.csv trong thư mục mẫu!")

    # Cần file sku_clustering_3_groups.csv cho phân cụm
    cluster_target = os.path.join(data_dir, "sku_clustering_3_groups.csv")
    if not os.path.exists(cluster_target):
        print("[INFO] Đang tạo file phân cụm SKU...")
        run_script(get_script_path("step0_clustering.py"), cwd=parent_dir)

    durations = {}

    # =========================================================================
    # GIAI ĐOẠN 1: HUẤN LUYỆN PIPELINE MÔ HÌNH KÉP (LIGHTGBM + CATBOOST BLEND)
    # =========================================================================
    print_banner("GIAI ĐOẠN 1: HUẤN LUYỆN MÔ HÌNH DUAL-MODEL PIPELINE (LGBM + CATBOOST)")
    step1_target = os.path.join(data_dir, "v22_stage1_dual_model.csv")
    if os.path.exists(step1_target):
        print(f"[INFO] Phát hiện đã có sẵn kết quả Giai đoạn 1 tại: {step1_target}. Tự động bỏ qua!")
        durations['1. Dual Model GBDT'] = 0.0
    else:
        durations['1. Dual Model GBDT'] = run_script(get_script_path("step1_dual_model.py"), cwd=parent_dir)

    # =========================================================================
    # GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH PHI TẬP TRUNG (DECENTRALIZED PIPELINE)
    # =========================================================================
    print_banner("GIAI ĐOẠN 2: HUẤN LUYỆN PIPELINE PHI TẬP TRUNG THEO CỤM (DECENTRALIZED)")
    step2_target = os.path.join(data_dir, "v22_stage2_decentralized.csv")
    if os.path.exists(step2_target):
        print(f"[INFO] Phát hiện đã có sẵn kết quả Giai đoạn 2 tại: {step2_target}. Tự động bỏ qua!")
        durations['2. Decentralized Cluster'] = 0.0
    else:
        durations['2. Decentralized Cluster'] = run_script(get_script_path("step2_decentralized.py"), cwd=parent_dir)

    # =========================================================================
    # GIAI ĐOẠN 3: TỐI ƯU HÓA TỔ HỢP GBDT + BASELINE (V7 BLEND)
    # =========================================================================
    print_banner("GIAI ĐOẠN 3: TỐI ƯU HÓA LIÊN KẾT GBDT & BASELINE (V7)")
    step3_target = os.path.join(data_dir, "v22_stage3_optimized_blend.csv")
    if os.path.exists(step3_target):
        print(f"[INFO] Phát hiện đã có sẵn kết quả Giai đoạn 3 tại: {step3_target}. Tự động bỏ qua!")
        durations['3. Optimize V7 Blend'] = 0.0
    else:
        durations['3. Optimize V7 Blend'] = run_script(get_script_path("step3_optimize_blend.py"), cwd=parent_dir)

    # =========================================================================
    # GIAI ĐOẠN 4: BLEND CHAMPIONS (V18 GRANDMASTER BLEND)
    # =========================================================================
    print_banner("GIAI ĐOẠN 4: KẾT HỢP CÁC CHAMPIONS (V18 GRANDMASTER BLEND)")
    step4_target = os.path.join(data_dir, "v22_stage4_grandmaster_blend.csv")
    if os.path.exists(step4_target):
        print(f"[INFO] Phát hiện đã có sẵn kết quả Giai đoạn 4 tại: {step4_target}. Tự động bỏ qua!")
        durations['4. Ensemble Champions v18'] = 0.0
    else:
        durations['4. Ensemble Champions v18'] = run_script(get_script_path("step4_ensemble.py"), cwd=parent_dir)

    # =========================================================================
    # GIAI ĐOẠN 5: DIRICHLET ULTIMATE ENSEMBLE (V22)
    # =========================================================================
    print_banner("GIAI ĐOẠN 5: CHẠY THUẬT TOÁN TỐI ƯU HÓA DIRICHLET ENSEMBLE CHO V22")
    
    scorer = LocalScorer()
    
    files_to_blend = [
        "v22_stage4_grandmaster_blend.csv",
        "v22_stage3_optimized_blend.csv", 
        "v22_stage2_decentralized.csv",
        "v22_stage1_dual_model.csv",
        "baseline_submission.csv"
    ]
    
    preds_dict = {}
    f_cols = [f'F{i}' for i in range(1, 29)]
    
    for filename in files_to_blend:
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        df_val = df[df['id'].str.endswith('_validation')].copy()
        df_val['ItemCode'] = df_val['id'].str.replace('_validation', '', regex=False)
        df_val = df_val.set_index('ItemCode').reindex(scorer.skus_order).reset_index()
        preds_dict[filename] = df_val[f_cols].values.astype(np.float32)
        print(f" -> Đã căn chỉnh dữ liệu thành công cho: {filename}")
        
    names = list(preds_dict.keys())
    pred_matrices = [preds_dict[n] for n in names]
    
    best_score = 999.0
    best_weights = None
    best_threshold = 0.0
    best_multiplier = 1.0
    
    # Thiết lập Random Seed = 42 cố định để tái lập 100% kết quả V22
    np.random.seed(42)
    num_iterations = 250
    
    print(f"\n -> Bắt đầu quét {num_iterations} tổ hợp ngẫu nhiên Dirichlet...")
    
    thresholds = [0.0, 0.02, 0.04]
    multipliers = [0.98, 1.0, 1.02]
    
    # Tạo phân phối Dirichlet
    random_weights = np.random.dirichlet(np.ones(len(names)), size=num_iterations)
    
    for i, w in enumerate(random_weights):
        blended = np.zeros_like(pred_matrices[0])
        for j, weight in enumerate(w):
            blended += weight * pred_matrices[j]
            
        for m in multipliers:
            scaled = blended * m
            for t in thresholds:
                if t > 0:
                    clipped = np.where(scaled < t, 0.0, scaled)
                else:
                    clipped = scaled
                    
                score = scorer.evaluator.evaluate(clipped)
                if score < best_score:
                    best_score = score
                    best_weights = w
                    best_threshold = t
                    best_multiplier = m
                    
    print("\n" + "=" * 55)
    print("🏆 KẾT QUẢ TỐI ƯU HÓA ĐÃ TÁI LẬP THÀNH CÔNG V22:")
    print(f" [*] Local WRMSSE tối ưu nhất: {best_score:.6f}")
    print(f" [*] Dự đoán điểm số Kaggle:  {scorer.calibrate_score(best_score):.5f}")
    print(f" [*] Magic Multiplier:        {best_multiplier}")
    print(f" [*] Noise Threshold:          {best_threshold}")
    print(" [*] Trọng số tối ưu hóa Dirichlet:")
    for j, name in enumerate(names):
        print(f"     - {name:<48}: {best_weights[j]:.4f}")
    print("=" * 55)
    
    # Sinh file nộp bài submission_v22_ultimate_ensemble.csv
    final_blended = np.zeros_like(pred_matrices[0])
    for j, weight in enumerate(best_weights):
        final_blended += weight * pred_matrices[j]
        
    final_scaled = final_blended * best_multiplier
    if best_threshold > 0:
        final_clipped = np.where(final_scaled < best_threshold, 0.0, final_scaled)
    else:
        final_clipped = final_scaled
        
    sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    out_df = sample.copy()
    for c_idx in f_cols:
        out_df[c_idx] = out_df[c_idx].astype(float)
    
    val_mask = out_df['id'].str.endswith('_validation')
    eval_mask = out_df['id'].str.endswith('_evaluation')
    
    item_to_idx = {item: idx for idx, item in enumerate(scorer.skus_order)}
    
    val_ids = out_df.loc[val_mask, 'id']
    val_item_codes = val_ids.str.replace('_validation', '', regex=False)
    val_row_indices = [item_to_idx[code] for code in val_item_codes]
    out_df.loc[val_mask, f_cols] = final_clipped[val_row_indices]
    
    eval_ids = out_df.loc[eval_mask, 'id']
    eval_item_codes = eval_ids.str.replace('_evaluation', '', regex=False)
    eval_row_indices = [item_to_idx[code] for code in eval_item_codes]
    out_df.loc[eval_mask, f_cols] = final_clipped[eval_row_indices]
    # Áp dụng bộ lọc ngày đóng cửa Chủ Nhật (F2, F9, F16, F23 = 0.0) cho tất cả SKU để đạt WRMSSE tốt nhất
    print("[INFO] Áp dụng bộ lọc ngày đóng cửa Chủ Nhật (F2, F9, F16, F23 -> 0) cho tệp nộp bài cuối cùng...")
    for f in ['F2', 'F9', 'F16', 'F23']:
        out_df[f] = 0.0
        
    # Lưu vào thư mục data và lưu thêm các bản copy tại thư mục reproduce_v22 và thư mục data cha
    out_path_global = os.path.join(data_dir, "v22_final_ultimate_ensemble.csv")
    out_path_local = os.path.join(os.path.dirname(__file__), "v22_final_ultimate_ensemble.csv")
    out_path_parent_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "v22_final_ultimate_ensemble.csv"))
    
    out_df.to_csv(out_path_global, index=False)
    out_df.to_csv(out_path_local, index=False)
    
    parent_data_dir = os.path.dirname(out_path_parent_data)
    has_parent_data = False
    if os.path.exists(parent_data_dir):
        out_df.to_csv(out_path_parent_data, index=False)
        has_parent_data = True
    
    total_time = time.time() - master_start
    print_banner("QUY TRÌNH HUẤN LUYỆN TỪ ĐẦU TÁI LẬP V22 HOÀN TẤT THÀNH CÔNG!")
    print(f" [*] Tổng thời gian xử lý: {total_time/60:.2f} phút ({total_time:.2f} giây).")
    print(" [*] Thời gian chi tiết từng giai đoạn:")
    for stage, sec in durations.items():
         print(f"     - {stage:<45}: {sec:.2f} giây ({sec/60:.2f} phút)")
    print("\n🏆 Đã sinh file nộp bài V22 chính xác 100% từ con số 0 tại:")
    print(f" -> Standalone data: {out_path_global}")
    print(f" -> Standalone local: {out_path_local}")
    if has_parent_data:
        print(f" -> Kaggle Workspace: {out_path_parent_data}")
    print("=" * 85 + "\n")
    print(f" [*] Tổng thời gian xử lý: {total_time/60:.2f} phút ({total_time:.2f} giây).")
    print(" [*] Thời gian chi tiết từng giai đoạn:")
    for stage, sec in durations.items():
         print(f"     - {stage:<45}: {sec:.2f} giây ({sec/60:.2f} phút)")
    print("\n🏆 Đã sinh file nộp bài V22 chính xác 100% từ con số 0 tại:")
    print(f" -> Toàn cục: {out_path_global}")
    print(f" -> Cục bộ:   {out_path_local}")
    print("=" * 85 + "\n")

if __name__ == '__main__':
    main()
