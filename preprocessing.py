import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

def extract_number(text):
    if pd.isna(text) or text == '0' or text == 0:
        return np.nan
    # Use regex to find floats/ints (handling comma as decimal separator)
    match = re.search(r'(\d+[.,]?\d*)', str(text))
    if match:
        val = match.group(1).replace(',', '.')
        try:
            return float(val)
        except:
            return np.nan
    return np.nan

def convert_price(price_str):
    if pd.isna(price_str) or 'Thỏa thuận' in str(price_str):
        return np.nan
    
    # Extract number and unit
    price_str = str(price_str).lower().replace(',', '.')
    match = re.search(r'(\d+\.?\d*)', price_str)
    if not match:
        return np.nan
    
    val = float(match.group(1))
    
    if 'tỷ' in price_str:
        return val
    elif 'triệu' in price_str:
        return val / 1000
    return val

def main():
    print("--- 🚀 KHỞI ĐỘNG PIPELINE TIỀN XỬ LÝ (BATCH PROCESSING) ---")
    df = pd.read_csv('HN_Houseprice.csv')
    initial_shape = df.shape
    
    # 1. Số hóa cột Price & Loại bỏ 'Thỏa thuận'
    print("[1/5] Đang số hóa cột Price và lọc bỏ 'Thỏa thuận'...")
    df['Price'] = df['Price'].apply(convert_price)
    df = df.dropna(subset=['Price']) # Loại bỏ Thỏa thuận và Null
    
    # 2. Bóc tách kích thước (Area, Entrancewidth, Width_meters, Floors, Bedrooms)
    print("[2/5] Đang trích xuất đặc trưng số (Dimensions extraction)...")
    dim_cols = {
        'Area': 'Area_m2',
        'Entrancewidth': 'Entrance_width',
        'Width_meters': 'Width',
        'Floors': 'Floors',
        'Bedrooms': 'Bedrooms',
        'Bathrooms': 'Bathrooms'
    }
    
    for col, new_name in dim_cols.items():
        if col in df.columns:
            df[new_name] = df[col].apply(extract_number)
            # Replace 0 with NaN if not already handled by extract_number
            df.loc[df[new_name] == 0, new_name] = np.nan
    
    # 3. Làm sạch ngoại lệ (Outliers)
    print("[3/5] Đang xử lý các giá trị ngoại lệ (Outliers detection)...")
    # Theo quy luật thị trường Hà Nội: 
    # - Diện tích thường > 10m2 và < 500m2 cho nhà ở thông thường
    # - Giá tỷ VNĐ: loại bỏ các giá trị quá nhỏ (< 0.5 tỷ) hoặc quá lớn (> 200 tỷ cho dự án môn học)
    df = df[(df['Area_m2'] >= 10) & (df['Area_m2'] <= 500)]
    df = df[(df['Price'] >= 0.5) & (df['Price'] <= 200)]
    
    # 4. Xử lý dữ liệu thiếu (Median Imputation by District)
    print("[4/5] Đang điền khuyết (Imputation) bằng Trung vị theo Quận...")
    impute_cols = ['Entrance_width', 'Width']
    for col in impute_cols:
        if col in df.columns:
            # Tính trung vị theo Quận
            df[col] = df.groupby('District')[col].transform(lambda x: x.fillna(x.median()))
            # Nếu vẫn còn NaN (do Quận đó toàn NaN), điền bằng trung vị toàn bộ dataset
            df[col] = df[col].fillna(df[col].median())

    # 5. Báo cáo kết quả
    print("\n" + "="*50)
    print("📊 THỐNG KÊ SAU KHI LÀM SẠCH")
    print("="*50)
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.describe().to_string())
    
    print(f"\nDữ liệu ban đầu: {initial_shape[0]} dòng")
    print(f"Dữ liệu sau xử lý: {df.shape[0]} dòng")
    print(f"Tỷ lệ giữ lại: {round(df.shape[0]/initial_shape[0]*100, 2)}%")

    # Vẽ Correlation Heatmap
    try:
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap: Các yếu tố ảnh hưởng đến Giá (Hanoi 2024)')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        print("\n[V] Đã lưu biểu đồ tương quan tại 'correlation_heatmap.png'")
    except Exception as e:
        print(f"\n[!] Không thể vẽ biểu đồ: {e}")

    # Lưu file sạch
    df.to_csv('HN_Houseprice_Cleaned.csv', index=False)
    print("[V] Đã lưu dữ liệu sạch: 'HN_Houseprice_Cleaned.csv'")

if __name__ == "__main__":
    main()
