import pandas as pd
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')

# Tự động xác định thư mục dữ liệu cục bộ hoặc gốc dự án
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, 'data', 'train.csv')):
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
else:
    DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')

def clean_column(df, col):
    if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return df[col].astype(float)
        else:
            return df[col].astype(str).str.replace(',', '.', regex=False).astype(float)
    return 0.0

def main():
    print("[INFO] Đang đọc và phân tích dữ liệu từ data/train.csv...")
    if not os.path.exists(TRAIN_PATH):
        print(f"[-] Lỗi: Không tìm thấy {TRAIN_PATH}")
        return
        
    raw = pd.read_csv(TRAIN_PATH)
    
    # 1. Làm sạch các cột số
    print("[INFO] Đang làm sạch các cột số...")
    raw['Quantity'] = clean_column(raw, 'Quantity')
    raw['SalesAmount'] = clean_column(raw, 'SalesAmount')
    raw['Cost Amount'] = clean_column(raw, 'Cost Amount')
    
    # 2. Trích xuất đặc trưng Trả hàng / Hoàn tiền ở cấp độ giao dịch
    print("[INFO] Đang phân tích các giao dịch trả hàng...")
    raw['is_return'] = (raw['Quantity'] < 0).astype(int)
    raw['abs_quantity'] = raw['Quantity'].abs()
    
    # 3. Tính toán lợi nhuận cấp SKU (SalesAmount - Cost Amount)
    raw['Profit'] = raw['SalesAmount'] - raw['Cost Amount']
    
    # Nhóm theo SKU để tính toán các đặc trưng
    sku_groups = raw.groupby('ItemCode')
    
    sku_features = pd.DataFrame()
    sku_features['total_transactions'] = sku_groups.size()
    sku_features['return_transactions'] = sku_groups['is_return'].sum()
    sku_features['return_ratio'] = sku_features['return_transactions'] / sku_features['total_transactions']
    sku_features['has_returns'] = (sku_features['return_transactions'] > 0).astype(int)
    
    # Tính số lượng bán dương trung bình và lợi nhuận tổng/tuyệt đối
    pos_sales = raw[raw['Quantity'] > 0]
    pos_groups = pos_sales.groupby('ItemCode')
    sku_features['avg_quantity'] = pos_groups['Quantity'].mean().reindex(sku_features.index).fillna(0.0)
    
    sku_features['total_profit'] = sku_groups['Profit'].sum()
    sku_features['abs_profit'] = sku_features['total_profit'].abs()
    
    # Xử lý trọng số lợi nhuận âm (giới hạn dưới bằng 0 để tránh trọng số âm)
    sku_features['profit_weight'] = sku_features['total_profit'].clip(lower=0.0)
    
    # 4. Căn chỉnh lưới dày đặc (1754 ngày) để tính toán độ thưa daily chính xác
    print("[INFO] Căn chỉnh với Lưới Dày Đặc (1754 ngày) để tính Sparsity...")
    all_dates = pd.date_range(start='2020-11-17', end='2025-09-05')
    total_days = len(all_dates) # 1754 ngày
    
    # Số ngày hoạt động bán hàng thực tế của từng SKU
    active_days_df = pos_sales.groupby(['ItemCode', 'Date']).size().reset_index()
    active_days_count = active_days_df.groupby('ItemCode').size().reindex(sku_features.index).fillna(0.0)
    
    sku_features['active_days'] = active_days_count
    sku_features['sparsity'] = 1.0 - (sku_features['active_days'] / total_days)
    
    # Điền các giá trị còn thiếu
    sku_features = sku_features.fillna(0.0)
    
    print(f"[SUCCESS] Đã trích xuất đặc trưng cho {len(sku_features)} SKUs.")
    
    # 5. Phân cụm K-Means thành 3 nhóm
    print("[INFO] Đang thực hiện phân cụm K-Means (K=3)...")
    cluster_cols = ['sparsity', 'avg_quantity', 'abs_profit', 'return_ratio']
    X = sku_features[cluster_cols].values
    
    # Chuẩn hóa các đặc trưng
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Huấn luyện mô hình KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    sku_features['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 6. Phân tích chi tiết 3 cụm vừa tạo
    print("\n" + "="*80)
    print("                      KẾT QUẢ PHÂN CỤM SKU CHI TIẾT (K=3)")
    print("="*80)
    
    for c in range(3):
        c_data = sku_features[sku_features['cluster'] == c]
        print(f"\n[+] CỤM {c}: ({len(c_data)} SKUs, Chiếm {len(c_data)/len(sku_features)*100:.2f}%)")
        print(f"    - Sparsity trung bình:          {c_data['sparsity'].mean()*100:.2f}%")
        print(f"    - Số ngày bán hàng trung bình:  {c_data['active_days'].mean():.1f} / {total_days} ngày")
        print(f"    - Số lượng bán trung bình (Q):  {c_data['avg_quantity'].mean():.4f}")
        print(f"    - Lợi nhuận tuyệt đối trung bình: {c_data['abs_profit'].mean():,.1f} VND")
        print(f"    - Tỉ lệ giao dịch trả hàng:     {c_data['return_ratio'].mean()*100:.4f}%")
        print(f"    - Số lượng SKU có trả hàng:     {c_data['has_returns'].sum()} / {len(c_data)}")
        print(f"    - SKU tiêu biểu của cụm:        {c_data.index[:5].tolist()}")
    
    # Lưu bảng ánh xạ phân cụm phục vụ tích hợp vào pipeline sau này
    sku_features = sku_features.reset_index()
    output_path = os.path.join(DATA_DIR, 'sku_clustering_3_groups.csv')
    sku_features[['ItemCode', 'cluster', 'sparsity', 'avg_quantity', 'abs_profit', 'return_ratio', 'has_returns', 'profit_weight']].to_csv(output_path, index=False)
    print("\n" + "="*80)
    print(f"[SUCCESS] Đã ghi file phân cụm vào: {output_path}")
    print("="*80)

if __name__ == '__main__':
    main()
