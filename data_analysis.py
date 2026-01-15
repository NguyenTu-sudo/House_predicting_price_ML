import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

file_path = 'HN_Houseprice.csv'

TARGET_COL = 'Gia_ban_ty'

def perform_analysis():
    df = pd.read_csv(file_path)
    
    print("### 1. Load dữ liệu (5 dòng đầu)")
    print(df.head().to_markdown())
    
    print("\n### 2. Cấu trúc dữ liệu (info)")
    df.info()
    
    print("\n### 3. Kiểm tra dữ liệu thiếu (Literal Nulls)")
    missing_count = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Số lượng thiếu': missing_count, 'Tỷ lệ (%)': missing_percentage})
    print(missing_df.to_markdown())
    
    if TARGET_COL in df.columns:
        print(f"\n### 4. Thống kê cột '{TARGET_COL}'")
        print(df[TARGET_COL].describe().to_markdown())
    else:
        print(f"\n### 4. Không tìm thấy cột target '{TARGET_COL}' trong file.")

    print("\n### Các giá trị '0' (Nghi ngờ là missing values)")
    zero_counts = {}
    for col in df.columns:
        # Check both numeric 0 and string '0'
        count = ((df[col] == 0) | (df[col] == '0')).sum()
        if count > 0:
            zero_counts[col] = count
    print(pd.Series(zero_counts).to_markdown())

if __name__ == "__main__":
    perform_analysis()
