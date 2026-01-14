"""eda_analysis.py

EDA nhanh cho bộ dữ liệu mới (Distance).
Đọc: HN_Houseprice_Cleaned.csv
Tạo 1 vài biểu đồ cơ bản:
- Phân phối giá (tỷ)
- Phân phối giá/m2 (triệu)
- Giá/m2 trung bình theo quận/huyện

Chạy:
    python eda_analysis.py

"""

import pandas as pd
import matplotlib.pyplot as plt

CLEAN_FILE = 'HN_Houseprice_Cleaned.csv'
TARGET_COL = 'Gia_ban_ty'


def main():
    df = pd.read_csv(CLEAN_FILE)

    # Nếu preprocessing.py đã tạo Gia_trieu_m2 thì dùng luôn
    if 'Gia_trieu_m2' not in df.columns:
        df['Gia_trieu_m2'] = (df[TARGET_COL] * 1000) / df['Dien_tich_m2']

    # 1) Distribution: price
    plt.figure(figsize=(10, 5))
    plt.hist(df[TARGET_COL], bins=60)
    plt.title('Phân phối Giá bán (tỷ VNĐ)')
    plt.xlabel('Giá (tỷ)')
    plt.ylabel('Tần suất')
    plt.tight_layout()
    plt.savefig('distribution_price.png', dpi=160)

    # 2) Distribution: price per m2
    plt.figure(figsize=(10, 5))
    plt.hist(df['Gia_trieu_m2'].dropna(), bins=60)
    plt.title('Phân phối Giá / m² (triệu VNĐ/m²)')
    plt.xlabel('Giá/m² (triệu)')
    plt.ylabel('Tần suất')
    plt.tight_layout()
    plt.savefig('distribution_price_per_m2.png', dpi=160)

    # 3) Mean price per m2 by district
    district_price = (
        df.groupby('Quan')['Gia_trieu_m2']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    plt.barh(district_price['Quan'], district_price['Gia_trieu_m2'])
    plt.title('Giá / m² trung bình theo Quận/Huyện')
    plt.xlabel('Triệu VNĐ/m²')
    plt.ylabel('Quận/Huyện')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('district_price_per_m2.png', dpi=160)

    print('✅ Đã tạo biểu đồ:')
    print('- distribution_price.png')
    print('- distribution_price_per_m2.png')
    print('- district_price_per_m2.png')


if __name__ == '__main__':
    main()
