"""model_training.py

Huấn luyện mô hình dự đoán giá nhà Hà Nội dựa trên dataset đã PROCESSED.

Theo pipeline mới, dữ liệu sau khi làm sạch mặc định còn khoảng ~15k dòng,
vì vậy ta có thể thử thêm nhiều thuật toán để giảm sai số.

Đầu vào mặc định:
    - HN_Houseprice_Processed.csv  (tạo bởi preprocessing.py)

Đầu ra:
    - best_model.pkl          : model tốt nhất (train trên log(1+price))
    - model_features.pkl      : danh sách cột feature dùng để train
    - model_comparison.csv    : bảng so sánh các mô hình

Chạy:
    python model_training.py

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Giới hạn số luồng mặc định để tránh tình trạng quá tải trên máy yếu / môi trường bị giới hạn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    XGBRegressor = None  # type: ignore
    HAS_XGB = False


TARGET_COL = "Gia_ban_ty"
TARGET_LOG_COL = "Gia_ban_ty_log"


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Đánh giá trên thang giá gốc (tỷ) và R2 trên log."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_log = r2_score(y_true_log, y_pred_log)
    return {"MAE (Tỷ VNĐ)": float(mae), "RMSE (Tỷ VNĐ)": rmse, "R2 (log-scale)": float(r2_log)}


def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: pd.Series,
    y_test_log: pd.Series,
    random_state: int = 42,
    tune: bool = False,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, object, str]:
    results: list[dict] = []

    # 1) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    pred_lr = lr.predict(X_test)
    results.append({"Model": "Linear Regression", **evaluate(y_test_log.values, pred_lr)})

    # 2) Ridge Regression
    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(X_train, y_train_log)
    pred_ridge = ridge.predict(X_test)
    results.append({"Model": "Ridge Regression", **evaluate(y_test_log.values, pred_ridge)})

    # 3) Random Forest (mặc định tốt với dữ liệu nhiều dạng)
    # Random Forest là mô hình dễ dùng và khá ổn định.
    # Giảm số cây + giới hạn độ sâu để train nhanh (phù hợp chạy demo/app).
    rf = RandomForestRegressor(
        n_estimators=40,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=14,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    rf.fit(X_train, y_train_log)
    pred_rf = rf.predict(X_test)
    results.append({"Model": "Random Forest", **evaluate(y_test_log.values, pred_rf)})

    # 4) Extra Trees (thường mạnh với tabular, train nhanh)
    et = ExtraTreesRegressor(
        n_estimators=250,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
    )
    et.fit(X_train, y_train_log)
    pred_et = et.predict(X_test)
    results.append({"Model": "Extra Trees", **evaluate(y_test_log.values, pred_et)})

    # 5) Gradient Boosting (baseline boosting)
    gbr = GradientBoostingRegressor(random_state=random_state)
    gbr.fit(X_train, y_train_log)
    pred_gbr = gbr.predict(X_test)
    results.append({"Model": "Gradient Boosting", **evaluate(y_test_log.values, pred_gbr)})

    # 6) HistGradientBoosting (mạnh và nhanh)
    hgb = HistGradientBoostingRegressor(random_state=random_state)
    hgb.fit(X_train, y_train_log)
    pred_hgb = hgb.predict(X_test)
    results.append({"Model": "HistGradientBoosting", **evaluate(y_test_log.values, pred_hgb)})

    # 7) KNN Regression (cần scale)
    knn = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("knn", KNeighborsRegressor(n_neighbors=15, weights="distance")),
        ]
    )
    knn.fit(X_train, y_train_log)
    pred_knn = knn.predict(X_test)
    results.append({"Model": "KNN Regression", **evaluate(y_test_log.values, pred_knn)})

    # 8) XGBoost (nếu đã cài xgboost)
    xgb = None
    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
        )
        xgb.fit(X_train, y_train_log)
        pred_xgb = xgb.predict(X_test)
        results.append({"Model": "XGBoost Regressor", **evaluate(y_test_log.values, pred_xgb)})

    tuned_models: dict[str, object] = {}

    # 9) Tuning nhanh (tuỳ chọn)
    if tune:
        print("[tune] Đang chạy hyperparameter tuning (có thể mất vài phút)...")

        # 9.1) Tune Extra Trees (thường ổn định)
        et_search = RandomizedSearchCV(
            estimator=ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs),
            param_distributions={
                "n_estimators": [200, 400, 700],
                "max_depth": [None, 20, 30],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.5, 0.8],
            },
            n_iter=12,
            scoring="neg_mean_absolute_error",
            cv=3,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0,
        )
        et_search.fit(X_train, y_train_log)
        et_tuned = et_search.best_estimator_
        pred_et_tuned = et_tuned.predict(X_test)
        results.append({"Model": "Extra Trees (Tuned)", **evaluate(y_test_log.values, pred_et_tuned)})
        tuned_models["Extra Trees (Tuned)"] = et_tuned

        # 9.2) Tune XGBoost nếu có
        if HAS_XGB:
            xgb_search = RandomizedSearchCV(
                estimator=XGBRegressor(
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=n_jobs,
                    tree_method="hist",
                ),
                param_distributions={
                    "n_estimators": [400, 700, 1000],
                    "learning_rate": [0.03, 0.05, 0.08, 0.1],
                    "max_depth": [4, 5, 6, 7],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "reg_lambda": [1.0, 2.0, 4.0],
                },
                n_iter=15,
                scoring="neg_mean_absolute_error",
                cv=3,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=0,
            )
            xgb_search.fit(X_train, y_train_log)
            xgb_tuned = xgb_search.best_estimator_
            pred_xgb_tuned = xgb_tuned.predict(X_test)
            results.append({"Model": "XGBoost (Tuned)", **evaluate(y_test_log.values, pred_xgb_tuned)})
            tuned_models["XGBoost (Tuned)"] = xgb_tuned

    results_df = pd.DataFrame(results)

    # Best theo MAE nhỏ nhất
    best_row = results_df.sort_values("MAE (Tỷ VNĐ)").iloc[0]
    best_name = str(best_row["Model"])
    model_map = {
        "Linear Regression": lr,
        "Ridge Regression": ridge,
        "Random Forest": rf,
        "Extra Trees": et,
        "Gradient Boosting": gbr,
        "HistGradientBoosting": hgb,
        "KNN Regression": knn,
    }
    if xgb is not None:
        model_map["XGBoost Regressor"] = xgb

    # Thêm các model đã tune (nếu có)
    model_map.update(tuned_models)

    best_model = model_map[best_name]

    return results_df, best_model, best_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="HN_Houseprice_Processed.csv")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Nếu >0: lấy mẫu ngẫu nhiên N dòng để train nhanh (0 = dùng toàn bộ).",
    )
    parser.add_argument("--out_model", default="best_model.pkl")
    parser.add_argument("--out_features", default="model_features.pkl")
    parser.add_argument("--out_report", default="model_comparison.csv")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Bật hyperparameter tuning nhanh (RandomizedSearchCV) cho một số mô hình.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Số luồng cho mô hình tree/boosting. Mặc định 1 để ổn định trên máy yếu.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Không tìm thấy '{args.input}'. Hãy chạy: python preprocessing.py trước.")

    print("--- 🤖 HUẤN LUYỆN MÔ HÌNH (TRAIN ON LOG TARGET) ---")

    df = pd.read_csv(in_path)

    if args.sample and args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"[i] Dùng sample {len(df)} dòng để train nhanh")
    if TARGET_LOG_COL not in df.columns:
        df[TARGET_LOG_COL] = np.log1p(df[TARGET_COL].astype(float))

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, TARGET_LOG_COL]]
    X = df[feature_cols]
    y_log = df[TARGET_LOG_COL].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    print(f"Số lượng feature: {len(feature_cols)}")
    print(f"Số dòng train/test: {X_train.shape[0]} / {X_test.shape[0]}")

    results_df, best_model, best_name = train_and_compare(
        X_train, X_test, y_train, y_test, tune=bool(args.tune), n_jobs=int(args.n_jobs)
    )

    print("\n=== 📊 KẾT QUẢ SO SÁNH ===")
    print(results_df.to_string(index=False))

    results_df.to_csv(args.out_report, index=False)
    joblib.dump(best_model, args.out_model)
    joblib.dump(feature_cols, args.out_features)

    print(f"\n✅ Best model: {best_name}")
    print(f"✅ Đã lưu model: {args.out_model}")
    print(f"✅ Đã lưu feature list: {args.out_features}")
    print(f"✅ Đã lưu report: {args.out_report}")


if __name__ == "__main__":
    main()
