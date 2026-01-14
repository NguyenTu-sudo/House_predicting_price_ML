"""preprocessing.py

Tiền xử lý dữ liệu giá nhà Hà Nội (30 quận/huyện).

Theo yêu cầu mới:
- LOẠI BỎ biến `Nhom_Khu_vuc` (nếu có).
- Xuất `feature_schema.json` có RÀNG BUỘC theo quận/huyện dựa trên dữ liệu thô:
  * Numeric: thống kê min/max + (p05, median, p95) theo từng quận/huyện.
  * Categorical/Binary: danh sách lựa chọn theo từng quận/huyện (chỉ hiện giá trị có trong dữ liệu).
- Quy ước "Hai Bà Trưng là trung tâm" -> ép `Khoang_cach_TT_km = 0` cho quận Hai Bà Trưng
  (áp dụng cho cả schema & dữ liệu cleaned để mô hình học đúng quy ước UI).

Ghi chú:
- Dữ liệu làm sạch lưu <= max_rows (mặc định 20,000) theo lấy mẫu phân tầng theo quận/huyện.
- Schema ràng buộc được tính trên TẬP FULL sau làm sạch (không lấy mẫu) để chính xác hơn.

Chạy:
    python preprocessing.py --input HN_Houseprice.csv --max_rows 20000

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


TARGET_COL = "Gia_ban_ty"
DISTRICT_COL = "Quan_Huyen"
CENTER_DISTRICT = "Hai Bà Trưng"
CENTER_DISTANCE_VALUE = 0.0

# Biến phân loại (đã bỏ Nhom_Khu_vuc)
CATEGORICAL_COLS: List[str] = [
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

# Biến số
NUMERIC_COLS: List[str] = [
    "Khoang_cach_TT_km",
    "Dien_tich_m2",
    "Mat_tien_m",
    "So_tang",
    "So_phong_ngu",
    "So_phong_tam",
    "Do_rong_duong_m",
    "Tuoi_nha_nam",
]

# Biến nhị phân
BINARY_COLS: List[str] = [
    "O_to_vao",
    "Co_Gara",
    "Co_San_thuong",
    "Gan_nghia_trang_bai_rac",
    "Co_bi_ngap",
]

ALL_FEATURE_COLS: List[str] = CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS

# Các numeric là số nguyên (dùng step=1 trên UI)
INT_NUMERIC_COLS: set[str] = {"So_tang", "So_phong_ngu", "So_phong_tam", "Tuoi_nha_nam"}

# Step gợi ý cho UI (nếu không có thì tự suy)
UI_STEP_HINTS: Dict[str, float] = {
    "Khoang_cach_TT_km": 0.1,
    "Dien_tich_m2": 1.0,
    "Mat_tien_m": 0.1,
    "Do_rong_duong_m": 0.5,
}


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _coerce_binary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Chuẩn hoá về 0/1. Nếu dữ liệu là 'Có/Không' sẽ map tương ứng."""
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            df[c] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
            df[c] = df[c].clip(lower=0, upper=1)
        else:
            ss = s.astype(str).str.strip().str.lower()
            df[c] = ss.map(
                {
                    "1": 1,
                    "0": 0,
                    "true": 1,
                    "false": 0,
                    "có": 1,
                    "co": 1,
                    "không": 0,
                    "khong": 0,
                    "yes": 1,
                    "no": 0,
                }
            ).fillna(0).astype(int)
    return df


def _coerce_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan", "None", "NaN", ""]), c] = np.nan
    return df


def _clip_outliers_iqr(df: pd.DataFrame, col: str, k: float = 2.0) -> pd.DataFrame:
    """Clip outliers theo IQR (nới k=2.0 để không quá gắt)."""
    if col not in df.columns:
        return df
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return df
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return df
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    df[col] = s.clip(lo, hi)
    return df


def stratified_sample_by_district(
    df: pd.DataFrame, max_rows: int = 20000, seed: int = 42
) -> pd.DataFrame:
    """Lấy mẫu phân tầng theo quận/huyện để đảm bảo 30 quận/huyện đều có dữ liệu."""
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.copy()

    if DISTRICT_COL not in df.columns:
        return df.sample(n=max_rows, random_state=seed).copy()

    rng = np.random.default_rng(seed)
    districts = df[DISTRICT_COL].dropna().unique().tolist()
    n_d = max(len(districts), 1)

    # Mỗi quận/huyện lấy tối thiểu min_per (nếu đủ dữ liệu)
    min_per = max(20, max_rows // (n_d * 10))

    parts = []
    used_idx = set()

    for d in districts:
        sub = df[df[DISTRICT_COL] == d]
        take = min(min_per, len(sub))
        if take <= 0:
            continue
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        used_idx.update(idx.tolist())
        parts.append(df.loc[idx])

    remaining = max_rows - sum(len(p) for p in parts)
    if remaining > 0:
        rest = df.drop(index=list(used_idx), errors="ignore")
        if len(rest) > 0:
            take2 = min(remaining, len(rest))
            idx2 = rng.choice(rest.index.to_numpy(), size=take2, replace=False)
            parts.append(df.loc[idx2])

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def _numeric_stats(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"min": 0.0, "max": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0, "p01": 0.0, "p99": 0.0}

    qs = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "p01": float(qs.get(0.01, s.min())),
        "p05": float(qs.get(0.05, s.min())),
        "median": float(qs.get(0.5, s.median())),
        "p95": float(qs.get(0.95, s.max())),
        "p99": float(qs.get(0.99, s.max())),
    }


def _categorical_options(s: pd.Series) -> List[str]:
    s = s.dropna().astype(str).str.strip()
    if s.empty:
        return []
    # sort theo tần suất giảm dần để UI hiển thị "phổ biến" trước
    vc = s.value_counts()
    return vc.index.tolist()


def clean_full_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Làm sạch nhưng KHÔNG lấy mẫu (dùng để tính schema ràng buộc)."""
    df = df_raw.copy()

    report: Dict = {
        "raw_rows": int(len(df)),
        "raw_cols": int(df.shape[1]),
        "dropped_na_target": 0,
        "dropped_nonpositive_target": 0,
        "deduped": 0,
        "notes": [],
    }

    # Bỏ Nhom_Khu_vuc nếu có
    df = df.drop(columns=["Nhom_Khu_vuc"], errors="ignore")

    # Chỉ giữ các cột cần thiết nếu có đủ
    keep_cols = [c for c in (ALL_FEATURE_COLS + [TARGET_COL]) if c in df.columns]
    df = df[keep_cols].copy()

    # Coerce types
    df = _coerce_categorical(df, CATEGORICAL_COLS)
    df = _coerce_numeric(df, NUMERIC_COLS + [TARGET_COL])
    df = _coerce_binary(df, BINARY_COLS)

    # Drop NA target
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    report["dropped_na_target"] = int(before - len(df))

    # Drop non-positive target
    before = len(df)
    df = df[df[TARGET_COL] > 0].copy()
    report["dropped_nonpositive_target"] = int(before - len(df))

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    report["deduped"] = int(before - len(df))

    # Clip outliers nhẹ cho một số numeric
    for c in NUMERIC_COLS + [TARGET_COL]:
        df = _clip_outliers_iqr(df, c, k=2.0)

    # Missing categorical -> mode (theo toàn bộ tập)
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Không rõ"
            df[c] = df[c].fillna(fill)

    # Missing numeric -> median
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Quy ước trung tâm
    if DISTRICT_COL in df.columns and "Khoang_cach_TT_km" in df.columns:
        mask = df[DISTRICT_COL].astype(str).str.strip().eq(CENTER_DISTRICT)
        if mask.any():
            df.loc[mask, "Khoang_cach_TT_km"] = float(CENTER_DISTANCE_VALUE)
            report["notes"].append(f"Force Khoang_cach_TT_km={CENTER_DISTANCE_VALUE} for {CENTER_DISTRICT}")

    report["clean_full_rows"] = int(len(df))
    report["clean_cols"] = int(df.shape[1])
    report["districts"] = int(df[DISTRICT_COL].nunique()) if DISTRICT_COL in df.columns else None

    return df, report


def export_schema(df_clean_full: pd.DataFrame, out_path: Path) -> Dict:
    """Xuất schema gồm ràng buộc theo quận/huyện."""
    schema: Dict = {
        "version": "30area_with_district_constraints_no_Nhom_Khu_vuc",
        "target": TARGET_COL,
        "center_district": CENTER_DISTRICT,
        "center_distance_value": CENTER_DISTANCE_VALUE,
        "ui_note": "Ràng buộc (range & lựa chọn) được tính theo dữ liệu thô sau làm sạch, theo từng quận/huyện.",
        "categorical": {},  # global options
        "numeric": {},       # global stats
        "binary": {},        # global allowed values
        "per_district": {},  # district-specific
    }

    # Global
    for c in CATEGORICAL_COLS:
        if c in df_clean_full.columns:
            schema["categorical"][c] = _categorical_options(df_clean_full[c])

    for c in NUMERIC_COLS:
        if c in df_clean_full.columns:
            stats = _numeric_stats(df_clean_full[c])
            stats["is_int"] = bool(c in INT_NUMERIC_COLS)
            stats["step"] = float(UI_STEP_HINTS.get(c, 1.0 if stats["is_int"] else 0.1))
            schema["numeric"][c] = stats

    for c in BINARY_COLS:
        if c in df_clean_full.columns:
            vals = sorted(pd.to_numeric(df_clean_full[c], errors="coerce").dropna().astype(int).unique().tolist())
            schema["binary"][c] = vals if vals else [0, 1]

    # Per district
    if DISTRICT_COL in df_clean_full.columns:
        for d, sub in df_clean_full.groupby(DISTRICT_COL):
            d = str(d)
            d_info: Dict = {
                "n": int(len(sub)),
                "categorical": {},
                "numeric": {},
                "binary": {},
            }

            for c in CATEGORICAL_COLS:
                if c in sub.columns:
                    d_info["categorical"][c] = _categorical_options(sub[c])

            for c in NUMERIC_COLS:
                if c in sub.columns:
                    stt = _numeric_stats(sub[c])
                    # Override trung tâm
                    if d == CENTER_DISTRICT and c == "Khoang_cach_TT_km":
                        stt = {
                            "min": float(CENTER_DISTANCE_VALUE),
                            "max": float(CENTER_DISTANCE_VALUE),
                            "p01": float(CENTER_DISTANCE_VALUE),
                            "p05": float(CENTER_DISTANCE_VALUE),
                            "median": float(CENTER_DISTANCE_VALUE),
                            "p95": float(CENTER_DISTANCE_VALUE),
                            "p99": float(CENTER_DISTANCE_VALUE),
                        }
                    stt["is_int"] = bool(c in INT_NUMERIC_COLS)
                    stt["step"] = float(UI_STEP_HINTS.get(c, 1.0 if stt["is_int"] else 0.1))
                    d_info["numeric"][c] = stt

            for c in BINARY_COLS:
                if c in sub.columns:
                    vals = sorted(pd.to_numeric(sub[c], errors="coerce").dropna().astype(int).unique().tolist())
                    d_info["binary"][c] = vals if vals else [0, 1]

            schema["per_district"][d] = d_info

    out_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema


def encode_data(df_clean: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS, drop_first=False)
    return df_encoded


def add_log_target(df_encoded: pd.DataFrame) -> pd.DataFrame:
    df = df_encoded.copy()
    df["Gia_ban_ty_log"] = np.log1p(df[TARGET_COL])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="HN_Houseprice.csv")
    parser.add_argument("--max_rows", type=int, default=20000)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {inp.resolve()}")

    df_raw = pd.read_csv(inp)

    # 1) Clean full để tính schema ràng buộc (không lấy mẫu)
    df_clean_full, report = clean_full_data(df_raw)

    # 2) Export schema dựa trên full
    export_schema(df_clean_full, out_dir / "feature_schema.json")

    # 3) Lấy mẫu <= max_rows để lưu cleaned phục vụ training/EDA
    df_clean = stratified_sample_by_district(df_clean_full, max_rows=args.max_rows, seed=42)

    # Lưu các file theo naming quen thuộc trong repo
    df_raw_out = df_raw.drop(columns=["Nhom_Khu_vuc"], errors="ignore")
    df_raw_out.to_csv(out_dir / "HN_Houseprice.csv", index=False, encoding="utf-8")

    df_clean.to_csv(out_dir / "HN_Houseprice_Cleaned.csv", index=False, encoding="utf-8")

    df_encoded = encode_data(df_clean)
    df_encoded.to_csv(out_dir / "HN_Houseprice_Encoded.csv", index=False, encoding="utf-8")

    df_processed = add_log_target(df_encoded)
    df_processed.to_csv(out_dir / "HN_Houseprice_Processed.csv", index=False, encoding="utf-8")

    # Report
    report["max_rows"] = int(args.max_rows)
    report["clean_rows_sampled"] = int(len(df_clean))
    (out_dir / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Done. Saved:")
    print("- HN_Houseprice.csv (raw without Nhom_Khu_vuc)")
    print("- HN_Houseprice_Cleaned.csv (<= max_rows)")
    print("- HN_Houseprice_Encoded.csv")
    print("- HN_Houseprice_Processed.csv")
    print("- feature_schema.json (with per-district constraints)")
    print("- cleaning_report.json")


if __name__ == "__main__":
    main()
