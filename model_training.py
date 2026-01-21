"""model_training.py

Huấn luyện mô hình dự đoán giá nhà Hà Nội (đơn vị: tỷ VNĐ).

Điểm thay đổi theo yêu cầu:
- Loại bỏ thuộc tính `Nhom_Khu_vuc`.
- Dùng sklearn Pipeline + ColumnTransformer + OneHotEncoder(handle_unknown='ignore')
  để đảm bảo *tất cả biến phân loại* đều được đưa vào mô hình một cách nhất quán.
- Không phụ thuộc vào `model_features.pkl` (hạn chế lỗi lệch schema khi đổi dữ liệu).

Chạy:
    python model_training.py --data HN_Houseprice_Cleaned.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "Gia_ban_ty"

# Các cột phân loại (khớp preprocessing.py)
CATEGORICAL_COLS = [
    "Quan_Huyen",
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

BINARY_COLS = [
    "O_to_vao",
    "Co_Gara",
    "Co_San_thuong",
    "Gan_nghia_trang_bai_rac",
    "Co_bi_ngap",
]

NUMERIC_COLS = [
    "Khoang_cach_TT_km",
    "Dien_tich_m2",
    "Mat_tien_m",
    "So_tang",
    "So_phong_ngu",
    "So_phong_tam",
    "Do_rong_duong_m",
    "Tuoi_nha_nam",
]


def build_pipeline(alpha: float = 2.0) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS + BINARY_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    model = Ridge(alpha=alpha)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="HN_Houseprice_Cleaned.csv")
    parser.add_argument("--out_model", type=str, default="best_model.pkl")
    parser.add_argument("--alpha", type=float, default=2.0)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    # Minimal checks
    missing = [c for c in (CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS + [TARGET_COL]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    X = df[CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS].copy()
    y = df[TARGET_COL].astype(float).copy()

    # log target để ổn định (giảm ảnh hưởng cực trị)
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    pipe = build_pipeline(alpha=args.alpha)
    pipe.fit(X_train, y_train)

    # Evaluate
    pred_log = pipe.predict(X_test)
    pred = np.expm1(pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)

    # Save pipeline model
    joblib.dump(pipe, args.out_model)

    # Save feature names (optional, để debug)
    try:
        feat_names = pipe.named_steps["preprocess"].get_feature_names_out().tolist()
        joblib.dump(feat_names, "model_features.pkl")
    except Exception:
        pass

    print("✅ Training completed")
    print(f"MAE (tỷ VNĐ): {mae:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"Saved model to: {args.out_model}")


if __name__ == "__main__":
    main()
