import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true_log, y_pred_log, model_name):
    # Nghịch đảo log (y_log = log(1+y) -> y = exp(y_log) - 1)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true_log, y_pred_log) # R2 thường tính trên quy mô log nếu train trên log
    
    return {
        "Model": model_name,
        "MAE (Tỷ VNĐ)": mae,
        "RMSE (Tỷ VNĐ)": rmse,
        "R2 Score": r2
    }

def main():
    print("--- 🤖 KHỞI ĐỘNG GIAI ĐOẠN HUẤN LUYỆN MÔ HÌNH (SỬA LỖI LEAKAGE) ---")
    
    # Load data
    df = pd.read_csv('HN_Houseprice_Encoded.csv')
    
    # Xác định các cột cần loại bỏ (Metadata, Target, và Leakage features)
    # Giữ lại: Area_m2, Bedrooms, Bathrooms, Floors, Width, Entrance_width
    drop_cols = [
        'Title', 'Address', 'PostingDate', 'PostType', 'Area', 'Direction', 
        'Width_meters', 'Legal', 'Interior', 'Entrancewidth', 'Price', 'Price_per_m2'
    ]
    
    # Chỉ giữ lại các cột số thực sự là features
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['Price']
    
    # Fill NaNs for baseline models (Linear Regression)
    X = X.fillna(X.median())
    
    print(f"Số lượng Features sử dụng: {X.shape[1]}")
    print(f"Các features quan trọng: {X.columns[:10].tolist()}...")
    
    # 1. Chia tập Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Xử lý biến mục tiêu: Log Transformation
    print("[1/5] Đang áp dụng Log Transformation...")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    results = []
    
    # 3. Model 1: Linear Regression (Baseline)
    print("[2/5] Đang huấn luyện Model 1: Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    y_pred_lr = lr.predict(X_test)
    results.append(evaluate_model(y_test_log, y_pred_lr, "Linear Regression"))
    
    # 4. Model 2: Random Forest Regressor
    print("[3/5] Đang huấn luyện Model 2: Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train_log)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate_model(y_test_log, y_pred_rf, "Random Forest"))
    
    # 5. Model 3: XGBoost
    print("[4/5] Đang huấn luyện Model 3: XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train_log)
    y_pred_xgb = xgb_model.predict(X_test)
    results.append(evaluate_model(y_test_log, y_pred_xgb, "XGBoost"))
    
    # 6. So sánh kết quả
    print("\n" + "="*60)
    print("📊 BẢNG SO SÁNH KẾT QUẢ CÁC MÔ HÌNH (SAU KHI SỬA)")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    best_model = results_df.loc[results_df['MAE (Tỷ VNĐ)'].idxmin(), 'Model']
    print(f"\n=> Mô hình hiệu quả nhất: {best_model}")
    
    results_df.to_csv('model_comparison.csv', index=False)

if __name__ == "__main__":
    main()
