import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

def main():
    print("--- 🔍 KHỞI ĐỘNG PHÂN TÍCH EDA CHUYÊN SÂU ---")
    df = pd.read_csv('HN_Houseprice_Cleaned.csv')
    
    # Tạo biến Price_per_m2
    df['Price_per_m2'] = (df['Price'] * 1000) / df['Area_m2'] # Triệu/m2
    
    # 1. Phân tích phân phối (Distribution Analysis)
    print("[1/3] Đang phân tích phân phối của Price và Price_per_m2...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price Distribution
    sns.histplot(df['Price'], kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('Phân phối Giá (Tỷ VNĐ)')
    
    sns.boxplot(x=df['Price'], ax=axes[0, 1], color='blue')
    axes[0, 1].set_title('Biểu đồ Boxplot Giá')
    
    # Price_per_m2 Distribution
    sns.histplot(df['Price_per_m2'], kde=True, ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Phân phối Giá/m2 (Triệu VNĐ)')
    
    sns.boxplot(x=df['Price_per_m2'], ax=axes[1, 1], color='green')
    axes[1, 1].set_title('Biểu đồ Boxplot Giá/m2')
    
    plt.tight_layout()
    plt.savefig('distribution_analysis.png')
    
    price_skew = skew(df['Price'])
    price_m2_skew = skew(df['Price_per_m2'])
    
    print(f"- Độ lệch (Skewness) của Price: {price_skew:.2f}")
    print(f"- Độ lệch (Skewness) của Price_per_m2: {price_m2_skew:.2f}")
    
    # 2. Phân tích theo khu vực (District Insight)
    print("[2/3] Đang phân tích giá theo Quận...")
    district_price = df.groupby('District')['Price_per_m2'].mean().sort_values(ascending=False).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=district_price, x='Price_per_m2', y='District', palette='viridis')
    plt.title('Giá trung bình mỗi m2 theo Quận (Hà Nội 2024)')
    plt.xlabel('Giá trung bình (Triệu VNĐ/m2)')
    plt.ylabel('Quận/Huyện')
    plt.tight_layout()
    plt.savefig('district_price_analysis.png')
    
    print("\n--- TOP 10 QUẬN ĐẮT NHẤT ---")
    print(district_price.head(10).to_string(index=False))

    # 3. Chuẩn bị biến Phân loại (Encoding)
    print("\n[3/3] Đang thực hiện Encoding cho biến phân loại...")
    
    # Sử dụng One-Hot Encoding cho District và Direction (Phổ biến nhất cho Baseline)
    # Lưu ý: Với District nhiều giá trị, Target Encoding sẽ tốt hơn nhưng One-Hot trực quan hơn cho báo cáo
    df_encoded = pd.get_dummies(df, columns=['District', 'Direction'], prefix=['Dist', 'Dir'])
    
    # Lưu kết quả
    df_encoded.to_csv('HN_Houseprice_Encoded.csv', index=False)
    print(f"[V] Đã thực hiện One-Hot Encoding. Số lượng cột mới: {df_encoded.shape[1]}")
    print("[V] Đã lưu dữ liệu Encoded: 'HN_Houseprice_Encoded.csv'")

    # Nội dung cho Markdown Report
    print("\n" + "="*50)
    print("📝 NỘI DUNG CHO BÁO CÁO")
    print("="*50)
    print(f"**Nhận xét về phân phối:**")
    if price_skew > 1:
        print(f"- Giá (Price) bị lệch phải mạnh (Skewness = {price_skew:.2f}).")
        print("- Khuyến nghị: Sử dụng Log Transformation (np.log1p) để chuẩn hóa Price giúp mô hình đạt sai số thấp hơn.")
    else:
        print(f"- Giá (Price) có độ lệch vừa phải (Skewness = {price_skew:.2f}).")
    
    print(f"\n**Nhận xét về khu vực:**")
    top_district = district_price.iloc[0]['District']
    print(f"- {top_district} là khu vực có giá trung bình cao nhất Hà Nội năm 2024.")

if __name__ == "__main__":
    main()
