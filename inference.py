"""inference.py

Dự đoán thử bằng model đã train.

- Load: best_model.pkl + model_features.pkl
- Encode 1 record raw -> đúng vector feature

Chạy:
    python inference.py

"""

import joblib
import numpy as np
import pandas as pd

CATEGORICAL_COLS = [
    "Quan",
    "Loai_duong",
    "Mat_do_dan_cu",
    "An_ninh",
    "Tinh_trang_ngap",
    "Noi_that",
    "Tinh_trang_Dien_Nuoc",
]


def encode_input(record: dict, feature_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([record])
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS, prefix_sep="__")

    for c in feature_cols:
        if c not in df_encoded.columns:
            df_encoded[c] = 0

    return df_encoded[feature_cols]


def predict_price(model, feature_cols: list[str], record: dict) -> float:
    X = encode_input(record, feature_cols)
    pred_log = float(model.predict(X)[0])
    return float(np.expm1(pred_log))


def main():
    model = joblib.load("best_model.pkl")
    feature_cols = joblib.load("model_features.pkl")

    # Ví dụ 1 căn nhà mẫu
    sample = {
        "Quan": "Cầu Giấy",
        "Khoang_cach_TT_km": 6.0,
        "Dien_tich_m2": 60,
        "Mat_tien_m": 6.0,
        "So_tang": 5,
        "So_phong_ngu": 4,
        "So_phong_tam": 3,
        "Do_rong_duong_m": 6.0,
        "Loai_duong": "Đường nhựa",
        "O_to_vao": 1,
        "Co_Gara": 0,
        "Co_San_thuong": 1,
        "Gan_Metro_Bus": 1,
        "Mat_do_dan_cu": "Cao",
        "An_ninh": "Tốt",
        "Gan_nghia_trang_bai_rac": 0,
        "Tinh_trang_ngap": "Không ngập",
        "Noi_that": "Cơ bản",
        "Tinh_trang_Dien_Nuoc": "Tốt",
        "Tuoi_nha_nam": 10,
    }

    price = predict_price(model, feature_cols, sample)
    print("=== DỰ ĐOÁN THỬ ===")
    print(sample)
    print(f"=> Giá dự kiến: {price:.2f} tỷ VNĐ")


if __name__ == "__main__":
    main()
