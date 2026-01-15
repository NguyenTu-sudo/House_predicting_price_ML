"""inference.py

Chạy thử dự đoán bằng mô hình đã train (Pipeline).

Khác với bản cũ:
- Model trong project hiện tại được train theo Pipeline (preprocess + model),
  vì vậy inference KHÔNG cần tự one-hot bằng tay.

Chạy:
    python inference.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd


if __name__ == "__main__":
    model = joblib.load("best_model.pkl")

    # Ví dụ record (hãy thay bằng dữ liệu thật)
    record = {
        "Quan_Huyen": "Hoàn Kiếm",
        "Khoang_cach_TT_km": 0.0,
        "Dac_diem_khu_vuc": "Đông đúc",
        "Loai_dat": "Đất ở",
        "Loai_duong": "Ngõ",
        "Huong_nha": "Đông",
        "Phap_ly": "Sổ đỏ",
        "Mat_do_dan_cu": "Cao",
        "An_ninh": "Tốt",
        "Gan_Tien_ich": "Gần chợ",
        "Gan_Giao_thong": "Gần đường lớn",
        "Noi_that": "Đầy đủ",
        "Tinh_trang_Dien_Nuoc": "Ổn định",
        "Muc_do_xuong_cap": "Thấp",
        "Dien_tich_m2": 45,
        "Mat_tien_m": 4,
        "So_tang": 4,
        "So_phong_ngu": 3,
        "So_phong_tam": 3,
        "Do_rong_duong_m": 4,
        "Tuoi_nha_nam": 8,
        "O_to_vao": 1,
        "Co_Gara": 0,
        "Co_San_thuong": 1,
        "Gan_nghia_trang_bai_rac": 0,
        "Co_bi_ngap": 0,
    }

    X_in = pd.DataFrame([record])

    # Model dự đoán trên log1p(price)
    pred_log = float(model.predict(X_in)[0])
    pred_ty = float(np.expm1(pred_log))
    pred_ty = max(pred_ty, 0.0)

    print(f"Predicted price: {pred_ty:.3f} tỷ")
