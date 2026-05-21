import numpy as np
import pandas as pd
import os
import gc

TET_DATES = {
    2021: '2021-02-12',
    2022: '2022-02-01',
    2023: '2023-01-22',
    2024: '2024-02-10',
    2025: '2025-01-29',
    2026: '2026-02-17',
}

FIXED_HOLIDAYS_MD = [
    (1, 1),
    (4, 30),
    (5, 1),
    (9, 2),
]

HUNG_VUONG_DATES = {
    2021: '2021-04-21',
    2022: '2022-04-10',
    2023: '2023-04-29',
    2024: '2024-04-18',
    2025: '2025-04-07',
    2026: '2026-04-26',
}

def load_raw_data(train_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n--- Doc & Gop Du Lieu Giao Dich ---")
    raw = pd.read_csv(train_path)
    print(f"So dong giao dich tho: {raw.shape[0]}")

    for col in ['UnitPrice', 'Unit Cost', 'SalesAmount', 'Cost Amount']:
        if col in raw.columns:
            if pd.api.types.is_numeric_dtype(raw[col]):
                raw[col] = raw[col].astype(float)
            else:
                raw[col] = raw[col].astype(str).str.replace(',', '.', regex=False).astype(float)

    raw['Profit'] = raw['SalesAmount'] - raw['Cost Amount']
    profit_df = raw.groupby('ItemCode')['Profit'].sum().reset_index()
    profit_df['Profit'] = np.clip(profit_df['Profit'], 0, None)
    total_profit = profit_df['Profit'].sum()
    profit_df['weight'] = profit_df['Profit'] / (total_profit if total_profit > 0 else 1.0)

    # Return Rate Feature
    print("  -> Tinh toan ty le doi tra hang (Return Rate) cua tung SKU...")
    neg_qty = raw[raw['Quantity'] < 0].groupby('ItemCode')['Quantity'].sum().abs()
    pos_qty = raw[raw['Quantity'] > 0].groupby('ItemCode')['Quantity'].sum()
    return_rate_df = (neg_qty / (pos_qty + 1e-5)).reset_index(name='return_rate')
    profit_df = profit_df.merge(return_rate_df, on='ItemCode', how='left').fillna({'return_rate': 0.0})
    raw_gross = raw[raw['Quantity'] > 0]
    daily = raw_gross.groupby(['Date', 'ItemCode']).agg({
        'Quantity': 'sum', 'UnitPrice': 'mean', 'Unit Cost': 'mean'
    }).reset_index()

    return daily, profit_df

def build_dense_grid(daily_df, profit_df, horizon=56):
    print("\n--- Dung Luoi Cartesian Dense Grid ---")
    all_dates = pd.date_range(start='2020-11-17', end='2025-09-05')
    all_skus = profit_df['ItemCode'].unique()
    print(f"SKUs: {len(all_skus)}, Days: {len(all_dates)}")

    grid = pd.MultiIndex.from_product(
        [all_dates.strftime('%Y-%m-%d'), all_skus], names=['Date', 'ItemCode']
    ).to_frame().reset_index(drop=True)
    grid = grid.merge(daily_df, on=['Date', 'ItemCode'], how='left')
    grid['Quantity'] = grid['Quantity'].fillna(0.0)
    for c in ['UnitPrice', 'Unit Cost']:
        grid[c] = grid[c].fillna(grid.groupby('ItemCode')[c].transform('mean')).fillna(0.0)

    # Skill 1: days_since_first_sale & Active product filtering (M5 top 3% technique)
    print("  -> Tinh toan chu ky active (days_since_first_sale)...")
    first_sale = daily_df[daily_df['Quantity'] > 0].groupby('ItemCode')['Date'].min().reset_index()
    first_sale.columns = ['ItemCode', 'first_sale_date']
    grid = grid.merge(first_sale, on='ItemCode', how='left')
    
    # Set Quantity to NaN for days before the product was first launched (removes zero-bias noise)
    inactive_mask = grid['Date'] < grid['first_sale_date']
    grid.loc[inactive_mask, 'Quantity'] = np.nan
    grid['days_since_first_sale'] = (pd.to_datetime(grid['Date']) - pd.to_datetime(grid['first_sale_date'])).dt.days
    grid['days_since_first_sale'] = grid['days_since_first_sale'].fillna(-1).astype(np.float32)
    grid = grid.drop(columns=['first_sale_date'])

    future_dates = pd.date_range(start='2025-09-06', periods=horizon)
    fut = pd.MultiIndex.from_product(
        [future_dates.strftime('%Y-%m-%d'), all_skus], names=['Date', 'ItemCode']
    ).to_frame().reset_index(drop=True)
    fut['Quantity'] = np.nan
    for c in ['UnitPrice', 'Unit Cost']:
        mean_prices = grid.groupby('ItemCode')[c].mean()
        fut[c] = fut['ItemCode'].map(mean_prices).fillna(0.0)

    # Propagate days_since_first_sale into the future
    fut = fut.merge(first_sale, on='ItemCode', how='left')
    fut['days_since_first_sale'] = (pd.to_datetime(fut['Date']) - pd.to_datetime(fut['first_sale_date'])).dt.days
    fut['days_since_first_sale'] = fut['days_since_first_sale'].fillna(-1).astype(np.float32)
    fut = fut.drop(columns=['first_sale_date'])

    full = pd.concat([grid, fut], ignore_index=True)
    full = full.sort_values(['ItemCode', 'Date']).reset_index(drop=True)
    full['ItemCode'] = full['ItemCode'].astype('category')
    full['Quantity'] = full['Quantity'].astype(np.float32)
    full['day_int'] = full.groupby('ItemCode').cumcount() + 1
    return full

def add_tet_features(df, date_col='Date_dt'):
    unique_dates = pd.Series(df[date_col].unique()).sort_values().reset_index(drop=True)
    days_to_tet = pd.Series(np.nan, index=unique_dates.index, dtype=np.float32)
    
    for yr, tet_date in TET_DATES.items():
        tet_ts = pd.Timestamp(tet_date)
        mask = (unique_dates >= tet_ts - pd.Timedelta(days=60)) & (unique_dates <= tet_ts + pd.Timedelta(days=30))
        days_to_tet[mask] = (unique_dates[mask] - tet_ts).dt.days.astype(np.float32)
    
    if days_to_tet.isna().any():
        all_tets = sorted([pd.Timestamp(d) for d in TET_DATES.values()])
        for idx in unique_dates.index[days_to_tet.isna()]:
            dt = unique_dates.iloc[idx]
            dists = [(dt - t).days for t in all_tets]
            abs_dists = [abs(d) for d in dists]
            min_idx = np.argmin(abs_dists)
            days_to_tet.iloc[idx] = dists[min_idx]
            
    date_to_days_to_tet = dict(zip(unique_dates, days_to_tet))
    
    df['days_to_tet'] = df[date_col].map(date_to_days_to_tet).astype(np.float32)
    df['abs_days_to_tet'] = np.abs(df['days_to_tet']).astype(np.float32)
    df['is_tet_window'] = ((df['days_to_tet'] >= -7) & (df['days_to_tet'] <= 7)).astype(np.int8)
    df['is_pre_tet_rush'] = ((df['days_to_tet'] >= -21) & (df['days_to_tet'] <= -1)).astype(np.int8)
    df['is_tet_shutdown'] = ((df['days_to_tet'] >= 0) & (df['days_to_tet'] <= 6)).astype(np.int8)
    return df

def add_fixed_holiday_features(df, date_col='Date_dt'):
    unique_dates = pd.Series(df[date_col].unique()).sort_values().reset_index(drop=True)
    years = unique_dates.dt.year.unique()
    
    all_holidays = set()
    for yr in years:
        for m, d in FIXED_HOLIDAYS_MD:
            try:
                all_holidays.add(pd.Timestamp(year=yr, month=m, day=d))
            except ValueError:
                pass
        if yr in HUNG_VUONG_DATES:
            all_holidays.add(pd.Timestamp(HUNG_VUONG_DATES[yr]))
            
    all_holidays = sorted(all_holidays)
    holidays_arr = np.array([h.value for h in all_holidays])
    
    is_holiday_unique = unique_dates.isin(all_holidays).astype(np.int8)
    
    dates_arr = unique_dates.values.astype('datetime64[ns]').astype(np.int64)
    min_dists = np.zeros(len(dates_arr), dtype=np.float32)
    for i, d in enumerate(dates_arr):
        diffs = np.abs(holidays_arr - d) / 1e9 / 86400  # ns -> days
        min_dists[i] = diffs.min()
        
    date_to_is_holiday = dict(zip(unique_dates, is_holiday_unique))
    date_to_nearest_holiday = dict(zip(unique_dates, min_dists))
    
    df['is_holiday'] = df[date_col].map(date_to_is_holiday).astype(np.int8)
    df['days_to_nearest_holiday'] = df[date_col].map(date_to_nearest_holiday).astype(np.float32)
    return df

def add_payday_features(df, date_col='Date_dt'):
    unique_dates = pd.Series(df[date_col].unique()).sort_values().reset_index(drop=True)
    day_of_month = unique_dates.dt.day.values
    paydays = np.array([5, 10, 15])
    dists = np.abs(day_of_month[:, None] - paydays[None, :])
    min_dist = dists.min(axis=1).astype(np.float32)
    
    date_to_payday = dict(zip(unique_dates, min_dist))
    df['days_to_payday'] = df[date_col].map(date_to_payday).astype(np.float32)
    df['is_payday_window'] = (df['days_to_payday'] <= 2).astype(np.int8)
    return df

def add_price_features(df):
    avg_price = df.groupby('ItemCode')['UnitPrice'].transform('mean')
    df['price_ratio'] = (df['UnitPrice'] / (avg_price + 1e-5)).astype(np.float32)
    df['margin_ratio'] = (df['UnitPrice'] / (df['Unit Cost'] + 1e-5)).astype(np.float32)
    df['price_momentum'] = (
        df['UnitPrice'] - df.groupby('ItemCode')['UnitPrice'].shift(56)
    ).astype(np.float32)
    
    # Dynamic Elasticity Features
    df['price_roll_mean_28'] = df.groupby('ItemCode')['UnitPrice'].transform(lambda x: x.rolling(28, min_periods=1).mean()).astype(np.float32)
    df['price_ratio_roll_28'] = (df['UnitPrice'] / (df['price_roll_mean_28'] + 1e-5)).astype(np.float32)
    df['price_roll_mean_56'] = df.groupby('ItemCode')['UnitPrice'].transform(lambda x: x.rolling(56, min_periods=1).mean()).astype(np.float32)
    df['price_ratio_roll_56'] = (df['UnitPrice'] / (df['price_roll_mean_56'] + 1e-5)).astype(np.float32)
    return df

def create_shared_features(df, profit_df):
    print("\n--- Generating Shared Features ---")
    df['Date_dt'] = pd.to_datetime(df['Date'])
    df['dayofweek'] = df['Date_dt'].dt.dayofweek.astype(np.int8)
    df['dayofmonth'] = df['Date_dt'].dt.day.astype(np.int8)
    df['month'] = df['Date_dt'].dt.month.astype(np.int8)
    df['year'] = df['Date_dt'].dt.year.astype(np.int16)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(np.int8)
    df['weekofyear'] = df['Date_dt'].dt.isocalendar().week.astype(np.int8)
    df['quarter'] = df['Date_dt'].dt.quarter.astype(np.int8)
    df['margin'] = (df['UnitPrice'] - df['Unit Cost']).astype(np.float32)

    df = add_tet_features(df, 'Date_dt')
    df = add_fixed_holiday_features(df, 'Date_dt')
    df = add_payday_features(df, 'Date_dt')
    df = add_price_features(df)

    # E-commerce mega-sale days (9.9, 10.10, 11.11, 12.12)
    df['is_mega_sale'] = (((df['dayofmonth'] == 9) & (df['month'] == 9)) |
                          ((df['dayofmonth'] == 10) & (df['month'] == 10)) |
                          ((df['dayofmonth'] == 11) & (df['month'] == 11)) |
                          ((df['dayofmonth'] == 12) & (df['month'] == 12))).astype(np.int8)

    # Category Proxy (SKU groups)
    df['cat_proxy'] = df['ItemCode'].astype(str).str[:7].astype('category')

    return_rate_map = profit_df.set_index('ItemCode')['return_rate'].to_dict()
    df['return_rate'] = df['ItemCode'].map(return_rate_map).fillna(0.0).astype(np.float32)

    # Load and merge Weather Data
    weather_path = os.path.join("data", "historical_weather.csv")
    if os.path.exists(weather_path):
        print("  -> Integration: Historical Weather Data merged (with thermal stress & rolling precipitation)...")
        wdf = pd.read_csv(weather_path)
        wdf['Date'] = wdf['Date'].astype(str)
        
        # Calculate daily rolling aggregates & range on the global 1,810-row table first (instantly fast!)
        wdf['temp_range'] = (wdf['temp_max'] - wdf['temp_min']).astype(np.float32)
        wdf['rainfall_roll_sum_7'] = wdf['rainfall'].rolling(7, min_periods=1).sum().astype(np.float32)
        wdf['rainfall_roll_sum_30'] = wdf['rainfall'].rolling(30, min_periods=1).sum().astype(np.float32)
        
        df = df.merge(wdf[['Date', 'temp_max', 'temp_min', 'rainfall', 'temp_range', 'rainfall_roll_sum_7', 'rainfall_roll_sum_30']], on='Date', how='left')
        for col in ['temp_max', 'temp_min', 'rainfall', 'temp_range', 'rainfall_roll_sum_7', 'rainfall_roll_sum_30']:
            df[col] = df[col].astype(np.float32).fillna(0.0)

    cpi_path = os.path.join("data", "historical_cpi.csv")
    if os.path.exists(cpi_path):
        print("  -> Integration: Monthly CPI Data merged...")
        cpidf = pd.read_csv(cpi_path)
        # Tối ưu hóa tốc độ merge: dùng year và month kiểu số nguyên (vectorized) thay vì strftime cực chậm!
        cpidf['Date'] = pd.to_datetime(cpidf['Month'])
        cpidf['year'] = cpidf['Date'].dt.year.astype(np.int16)
        cpidf['month'] = cpidf['Date'].dt.month.astype(np.int8)
        
        df = df.merge(cpidf[['year', 'month', 'cpi_index', 'cpi_mom_growth']], on=['year', 'month'], how='left')
        for col in ['cpi_index', 'cpi_mom_growth']:
            df[col] = df[col].astype(np.float32).fillna(100.0)

    # Load and merge WTI Oil Price Data
    oil_path = os.path.join("data", "wti_oil_prices_clean.csv")
    if os.path.exists(oil_path):
        print("  -> Integration: Daily WTI Crude Oil prices merged...")
        odf = pd.read_csv(oil_path)
        odf['Date'] = odf['Date'].astype(str)
        df = df.merge(odf, on='Date', how='left')
        
        # Fill missing values (weekends/holidays) via ffill and bfill
        df['wti_price'] = df.groupby('ItemCode')['wti_price'].ffill().bfill().astype(np.float32).fillna(60.0)
        
        # WTI Rolling Features (Grouped by ItemCode)
        df['wti_roll_mean_7'] = df.groupby('ItemCode')['wti_price'].transform(lambda x: x.rolling(7, min_periods=1).mean()).astype(np.float32)
        df['wti_roll_mean_28'] = df.groupby('ItemCode')['wti_price'].transform(lambda x: x.rolling(28, min_periods=1).mean()).astype(np.float32)

    df = df.drop(columns=['Date_dt'])
    gc.collect()
    return df

def create_lag_features_public(df):
    print("\n--- Generating Lags for Public (>=28) ---")
    for lag in [28, 29, 30, 31, 35, 42, 56, 70, 84]:
        df[f'qty_lag_{lag}'] = df.groupby('ItemCode')['Quantity'].shift(lag).astype(np.float32)
        
    # Sửa rò rỉ biên bằng cách nhóm theo ItemCode trước khi tính rolling trễ
    shifted_group = df.groupby('ItemCode')['Quantity'].shift(28)
    grouped_shifted = shifted_group.groupby(df['ItemCode'])
    
    for w in [7, 14, 28, 56]:
        df[f'qty_roll_mean_{w}'] = grouped_shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).astype(np.float32)
        df[f'qty_roll_std_{w}'] = grouped_shifted.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).astype(np.float32)
        df[f'qty_roll_ema_{w}'] = shifted_group.groupby(df['ItemCode']).ewm(span=w, min_periods=1).mean().reset_index(level=0, drop=True).astype(np.float32)
    gc.collect()
    return df

def create_lag_features_private(df):
    print("\n--- Generating Lags for Private (>=56) ---")
    for lag in [56, 57, 58, 59, 63, 70, 84, 112]:
        df[f'qty_lag_{lag}'] = df.groupby('ItemCode')['Quantity'].shift(lag).astype(np.float32)
        
    # Sửa rò rỉ biên bằng cách nhóm theo ItemCode trước khi tính rolling trễ
    shifted_group = df.groupby('ItemCode')['Quantity'].shift(56)
    grouped_shifted = shifted_group.groupby(df['ItemCode'])
    
    for w in [7, 14, 28, 56]:
        df[f'qty_roll_mean_{w}'] = grouped_shifted.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).astype(np.float32)
        df[f'qty_roll_std_{w}'] = grouped_shifted.rolling(w, min_periods=1).std().reset_index(level=0, drop=True).astype(np.float32)
        df[f'qty_roll_ema_{w}'] = shifted_group.groupby(df['ItemCode']).ewm(span=w, min_periods=1).mean().reset_index(level=0, drop=True).astype(np.float32)
    gc.collect()
    return df
