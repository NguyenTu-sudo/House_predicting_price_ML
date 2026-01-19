"""model_training.py

Huấn luyện mô hình dự đoán giá nhà Hà Nội (đơn vị: tỷ VNĐ) – phiên bản MULTI-MODEL.

Mục tiêu của bản này:
- Giữ pipeline tiền xử lý (ColumnTransformer + OneHotEncoder(handle_unknown='ignore')) để
  tránh lỗi lệch schema, đảm bảo thay đổi *biến phân loại* sẽ ảnh hưởng tới dự đoán.
- Bổ sung NHIỀU THUẬT TOÁN (như bản cũ) để bạn có thể so sánh và tự động chọn mô hình tốt nhất.

Đầu vào mặc định:
    - HN_Houseprice_Cleaned.csv  (tạo bởi preprocessing.py)

Đầu ra:
    - best_model.pkl          : pipeline tốt nhất (train trên log1p(Gia_ban_ty))
    - model_comparison.csv    : bảng so sánh các mô hình
    - model_info.json         : thông tin mô hình tốt nhất
    - (tuỳ chọn) models/*.pkl : lưu tất cả mô hình để bạn chọn trong UI

Chạy nhanh (khuyến nghị):
    python model_training.py --sample 15000

Chạy đầy đủ + lưu tất cả mô hình:
    python model_training.py --save_all

Ghi chú:
- App Streamlit dự đoán theo log-target, vì vậy trong app sẽ dùng expm1() để đưa về tỷ VNĐ.
- XGBoost là tuỳ chọn; nếu bạn không cài được xgboost thì script tự bỏ qua.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

# Giới hạn số luồng mặc định để tránh tình trạng quá tải trên máy yếu / môi trường bị giới hạn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


try:
    from xgboost import XGBRegressor  # type: ignore

    HAS_XGB = True
except Exception:
    XGBRegressor = None  # type: ignore
    HAS_XGB = False


TARGET_COL = "Gia_ban_ty"

# Các cột phân loại (khớp preprocessing.py / app.py)
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


def _make_ohe_dense() -> OneHotEncoder:
    """Tạo OneHotEncoder output dạng dense để mọi thuật toán đều chạy ổn.

    - sklearn >= 1.2 dùng sparse_output
    - sklearn cũ hơn dùng sparse
    """

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS + BINARY_COLS),
            ("cat", _make_ohe_dense(), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Đánh giá trên thang giá gốc (tỷ) và R2 trên log."""

    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_log = float(r2_score(y_true_log, y_pred_log))
    return {
        "MAE (Tỷ VNĐ)": mae,
        "RMSE (Tỷ VNĐ)": rmse,
        "R2 (log-scale)": r2_log,
    }


def get_model_candidates(random_state: int, n_jobs: int, fast: bool) -> list[tuple[str, object]]:
    """Danh sách mô hình để so sánh.

    fast=True: giảm n_estimators để train nhanh hơn.
    """

    rf_estimators = 60 if fast else 150
    et_estimators = 200 if fast else 350
    xgb_estimators = 300 if fast else 700

    candidates: list[tuple[str, object]] = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=2.0, random_state=random_state)),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=rf_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
                max_depth=14,
                min_samples_leaf=2,
                max_features="sqrt",
            ),
        ),
        (
            "Extra Trees",
            ExtraTreesRegressor(
                n_estimators=et_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
                max_depth=None,
                min_samples_leaf=1,
                max_features="sqrt",
            ),
        ),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=random_state)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=random_state)),
        ("KNN Regression", KNeighborsRegressor(n_neighbors=15, weights="distance")),
    ]

    if HAS_XGB and XGBRegressor is not None:
        candidates.append(
            (
                "XGBoost Regressor",
                XGBRegressor(
                    n_estimators=xgb_estimators,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=n_jobs,
                    tree_method="hist",
                ),
            )
        )

    return candidates


def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: pd.Series,
    y_test_log: pd.Series,
    random_state: int,
    n_jobs: int,
    fast: bool,
) -> tuple[pd.DataFrame, Pipeline, str]:
    """Train nhiều mô hình và chọn best theo MAE nhỏ nhất."""

    preprocessor = build_preprocessor()
    results: list[dict] = []
    trained: dict[str, Pipeline] = {}

    for name, estimator in get_model_candidates(random_state=random_state, n_jobs=n_jobs, fast=fast):
        t0 = perf_counter()
        pipe = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", estimator),
            ]
        )

        pipe.fit(X_train, y_train_log)
        pred_log = pipe.predict(X_test)
        metrics = evaluate(y_test_log.values, pred_log)
        t1 = perf_counter()

        row = {"Model": name, **metrics, "Train+Eval (s)": round(t1 - t0, 3)}
        results.append(row)
        trained[name] = pipe

    results_df = pd.DataFrame(results).sort_values("MAE (Tỷ VNĐ)").reset_index(drop=True)
    best_name = str(results_df.iloc[0]["Model"])
    best_model = trained[best_name]
    return results_df, best_model, best_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="HN_Houseprice_Cleaned.csv")
    parser.add_argument("--out_model", type=str, default="best_model.pkl")
    parser.add_argument("--out_report", type=str, default="model_comparison.csv")
    parser.add_argument("--out_info", type=str, default="model_info.json")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Nếu >0: lấy mẫu ngẫu nhiên N dòng để train nhanh (0 = dùng toàn bộ).",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Số luồng cho mô hình tree/boosting. Mặc định 1 để ổn định.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Giảm tham số (n_estimators) để train nhanh hơn.",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Lưu tất cả mô hình (models/*.pkl) để chọn trong giao diện.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy '{args.data}'. Hãy chạy: python preprocessing.py trước."
        )

    df = pd.read_csv(data_path)

    # Kiểm tra cột bắt buộc
    required = set(CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS + [TARGET_COL])
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    if args.sample and 0 < int(args.sample) < len(df):
        df = df.sample(n=int(args.sample), random_state=int(args.random_state)).reset_index(drop=True)
        print(f"[i] Dùng sample {len(df)} dòng để train nhanh")

    X = df[CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS].copy()
    y = df[TARGET_COL].astype(float).copy()
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_log,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
    )

    print("--- 🤖 TRAIN MULTI-MODEL (log-target) ---")
    print(f"Rows train/test: {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"Categorical: {len(CATEGORICAL_COLS)} | Numeric: {len(NUMERIC_COLS)} | Binary: {len(BINARY_COLS)}")
    print(f"XGBoost available: {HAS_XGB}")

    results_df, best_model, best_name = train_and_compare(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
        fast=bool(args.fast),
    )

    # Lưu report
    results_df.to_csv(args.out_report, index=False, encoding="utf-8")
    print("\n=== 📊 MODEL COMPARISON (sorted by MAE) ===")
    print(results_df.to_string(index=False))

    # Lưu best model
    joblib.dump(best_model, args.out_model)
    print(f"\n✅ Best model: {best_name}")
    print(f"✅ Saved best model to: {args.out_model}")
    print(f"✅ Saved comparison to: {args.out_report}")

    # Lưu info
    info = {
        "best_model": best_name,
        "data": str(data_path.name),
        "sample": int(args.sample) if args.sample else 0,
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "n_jobs": int(args.n_jobs),
        "fast": bool(args.fast),
        "metrics": results_df.iloc[0].to_dict(),
    }
    Path(args.out_info).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    # Tuỳ chọn lưu tất cả models
    if args.save_all:
        out_dir = Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Lưu các pipeline đã train lại bằng cách re-fit nhanh trên full train+test
        # (để mỗi model có thể dùng ngay trong app; vẫn dùng log-target).
        X_full = X
        y_full = y_log

        preprocessor = build_preprocessor()
        for name, estimator in get_model_candidates(random_state=int(args.random_state), n_jobs=int(args.n_jobs), fast=bool(args.fast)):
            pipe = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor)),
                    ("model", estimator),
                ]
            )
            pipe.fit(X_full, y_full)
            safe_name = (
                name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
            )
            joblib.dump(pipe, out_dir / f"{safe_name}.pkl")
        print(f"✅ Saved all models to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
