"""inference.py

Dự đoán thử bằng model đã train.

- Load: best_model.pkl + model_features.pkl
- Encode 1 record raw -> đúng vector feature (one-hot giống preprocessing.py)

Chạy:
    python inference.py
"""

import joblib
import numpy as np
import pandas as pd

CATEGORICAL_COLS = [
    "Quan_Huyen",
    "Nhom_Khu_vuc",
    "Dac_diem_khu_vuc",
    "Loai_dat",
    "Loai_duong",
    "Huong_nha",
    "Phap_ly",
    "Mat_do_dan_cu",
    "An_ninh",
    "Gan_Tien_ich",
    "Gan_Giao_thong",
    "Noi_that",
    "Tinh_trang_Dien_Nuoc",
    "Muc_do_xuong_cap",
]


def encode_input(record: dict, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([record])
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS, drop_first=False)

    for c in feature_cols:
        if c not in df_encoded.columns:
            df_encoded[c] = 0

    return df_encoded[feature_cols]


def predict_price(model, feature_cols: list[str], record: dict) -> float:
    X = encode_input(record, feature_cols)
    y = float(model.predict(X)[0])

    # heuristic: nếu output nhỏ -> log-space
    if y < 20:
        return float(np.expm1(y))
    return y


if __name__ == "__main__":
    model = joblib.load("best_model.pkl")
    feature_cols = joblib.load("model_features.pkl")

    # Ví dụ record (hãy thay bằng dữ liệu thật)
    record = {
        "Quan_Huyen": "Hai Bà Trưng",
        "Nhom_Khu_vuc": "Trung tâm",
        "Khoang_cach_TT_km": 0.0,
        "Dac_diem_khu_vuc": "Đông đúc",
        "Loai_dat": "Đất ở",
        "Dien_tich_m2": 45,
        "Mat_tien_m": 4,
        "So_tang": 4,
        "So_phong_ngu": 3,
        "So_phong_tam": 3,
        "Do_rong_duong_m": 4,
        "Loai_duong": "Ngõ",
        "O_to_vao": 1,
        "Co_Gara": 0,
        "Co_San_thuong": 1,
        "Huong_nha": "Đông",
        "Phap_ly": "Sổ đỏ",
        "Mat_do_dan_cu": "Cao",
        "An_ninh": "Tốt",
        "Gan_nghia_trang_bai_rac": 0,
        "Co_bi_ngap": 0,
        "Gan_Tien_ich": "Gần chợ",
        "Gan_Giao_thong": "Gần đường lớn",
        "Noi_that": "Đầy đủ",
        "Tinh_trang_Dien_Nuoc": "Ổn định",
        "Muc_do_xuong_cap": "Thấp",
        "Tuoi_nha_nam": 8,
    }

    price = predict_price(model, feature_cols, record)
    print(f"Predicted price: {price:.2f} tỷ")
