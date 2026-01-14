"""preprocessing.py

Pipeline tiền xử lý cho bộ dữ liệu giá nhà Hà Nội.

Mục tiêu của pipeline:
1) RAW  -> CLEANED  : chuẩn hoá kiểu dữ liệu, lọc giá trị lỗi/outlier, (tuỳ chọn) giảm còn ~15k dòng.
2) CLEANED -> ENCODED : one-hot encode tất cả biến phân loại (bao gồm đủ 30 quận/huyện/thị xã).
3) ENCODED -> PROCESSED (train-ready) : thêm cột log(target) để mô hình ổn định hơn.
4) Xuất feature_schema.json: phục vụ Streamlit UI (đặc biệt là ràng buộc theo từng quận/huyện/thị xã).

Chạy nhanh:
    python preprocessing.py

Tuỳ chọn:
    python preprocessing.py --input "HaNoi_Housing_Final_Distance (1).csv" --max_rows 15000
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ============================
# CONFIG
# ============================

TARGET_COL = "Gia_ban_ty"

# Theo yêu cầu UI mới: coi Hai Bà Trưng là "trung tâm"
CENTER_DISTRICT = "Hai Bà Trưng"

# Danh sách đầy đủ 30 đơn vị hành chính cấp huyện của Hà Nội (12 quận + 17 huyện + 1 thị xã)
ALL_HANOI_UNITS: list[str] = [
    # 12 quận
    "Ba Đình",
    "Bắc Từ Liêm",
    "Cầu Giấy",
    "Đống Đa",
    "Hà Đông",
    "Hai Bà Trưng",
    "Hoàn Kiếm",
    "Hoàng Mai",
    "Long Biên",
    "Nam Từ Liêm",
    "Thanh Xuân",
    "Tây Hồ",
    # 1 thị xã
    "Sơn Tây",
    # 17 huyện
    "Ba Vì",
    "Chương Mỹ",
    "Đan Phượng",
    "Đông Anh",
    "Gia Lâm",
    "Hoài Đức",
    "Mê Linh",
    "Mỹ Đức",
    "Phú Xuyên",
    "Phúc Thọ",
    "Quốc Oai",
    "Sóc Sơn",
    "Thạch Thất",
    "Thanh Oai",
    "Thanh Trì",
    "Thường Tín",
    "Ứng Hòa",
]

def _ascii_key(s: str) -> str:
    """Chuẩn hoá chuỗi để so khớp không phân biệt dấu/hoa-thường."""
    s = str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = " ".join(s.split())
    return s

# Map tên không dấu -> tên chuẩn có dấu
ASCII_TO_OFFICIAL = {_ascii_key(name): name for name in ALL_HANOI_UNITS}


CATEGORICAL_COLS: list[str] = [
    "Quan",
    "Loai_duong",
    "Mat_do_dan_cu",
    "An_ninh",
    "Tinh_trang_ngap",
    "Noi_that",
    "Tinh_trang_Dien_Nuoc",
]

BINARY_COLS: list[str] = [
    "O_to_vao",
    "Co_Gara",
    "Co_San_thuong",
    "Gan_Metro_Bus",
    "Gan_nghia_trang_bai_rac",
]

NUMERIC_COLS: list[str] = [
    "Khoang_cach_TT_km",
    "Dien_tich_m2",
    "Mat_tien_m",
    "So_tang",
    "So_phong_ngu",
    "So_phong_tam",
    "Do_rong_duong_m",
    "Tuoi_nha_nam",
]

ALL_COLS = CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS + [TARGET_COL]


# ============================
# HELPERS
# ============================

def _normalize_quan(x: Any) -> str:
    """Chuẩn hoá tên quận/huyện:
    - strip khoảng trắng
    - bỏ tiền tố "Quận/Huyện/Thị xã" nếu người dùng có đưa vào
    """
    s = str(x).strip()
    # bỏ các tiền tố hay gặp
    for prefix in ["Quận ", "Huyện ", "Thị xã ", "Thi xa ", "Quan ", "Huyen "]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    # chuẩn hoá nhiều khoảng trắng
    s = " ".join(s.split())

    # map không dấu -> tên chuẩn (nếu khớp)
    key = _ascii_key(s)
    return ASCII_TO_OFFICIAL.get(key, s)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _coerce_binary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round()
        df[c] = df[c].clip(0, 1)
        df[c] = df[c].astype("Int64")
    return df


def _quantile_filter(df: pd.DataFrame, cols: list[str], q: float) -> pd.DataFrame:
    """Lọc outlier theo quantile 2 phía."""
    if q <= 0:
        return df
    lo = df[cols].quantile(q)
    hi = df[cols].quantile(1 - q)
    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= df[c].between(lo[c], hi[c])
    return df.loc[mask].copy()


def _stratified_sample(df: pd.DataFrame, by: str, n: int, random_state: int = 42) -> pd.DataFrame:
    """Lấy mẫu theo tỉ lệ nhóm (Quan) để giảm dataset mà vẫn giữ phân phối."""
    if n <= 0 or len(df) <= n:
        return df

    grp_sizes = df[by].value_counts()
    ratio = n / len(df)
    target = (grp_sizes * ratio).round().astype(int)

    # đảm bảo mỗi nhóm có ít nhất 1 nếu nhóm tồn tại
    target[target < 1] = 1

    # hiệu chỉnh tổng cho đúng n
    diff = int(n - target.sum())
    order = target.sort_values(ascending=False).index.tolist()
    i = 0
    step = 1 if diff > 0 else -1
    while diff != 0 and i < 100000:
        k = order[i % len(order)]
        # không giảm dưới 1
        if step < 0 and target[k] <= 1:
            i += 1
            continue
        target[k] += step
        diff -= step
        i += 1

    parts: list[pd.DataFrame] = []
    for grp, k in target.items():
        part = df[df[by] == grp].sample(n=int(k), random_state=random_state)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _safe_float(x: Any) -> float | None:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _numeric_summary(s: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"min": np.nan, "max": np.nan, "q05": np.nan, "q95": np.nan, "median": np.nan}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "q05": float(s.quantile(0.05)),
        "q95": float(s.quantile(0.95)),
        "median": float(s.median()),
    }


# ============================
# CORE PIPELINE
# ============================

def clean_data(
    df_raw: pd.DataFrame,
    *,
    outlier_q: float = 0.06,
    max_rows: int = 15000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """RAW -> CLEANED.

    outlier_q:
        - lọc outlier theo quantile 2 phía (áp cho numeric + target)
        - nếu muốn giữ nhiều dữ liệu hơn, giảm q
    max_rows:
        - nếu > 0: lấy mẫu stratified theo Quan
    """
    report: dict[str, Any] = {"raw_rows": int(df_raw.shape[0])}

    df = df_raw.copy()

    # 0) Kiểm tra cột tối thiểu
    missing_cols = [c for c in ALL_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu cột trong dữ liệu: {missing_cols}")

    # 1) Chuẩn hoá kiểu string cho categorical
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype(str).map(lambda x: x.strip())

    # Chuẩn hoá tên quận/huyện
    df["Quan"] = df["Quan"].map(_normalize_quan)

    # 2) Giữ lại đúng các đơn vị Hà Nội
    df = df[df["Quan"].isin(ALL_HANOI_UNITS)].copy()
    report["rows_after_valid_quan"] = int(len(df))

    # 3) Coerce numeric/binary
    df = _coerce_numeric(df, NUMERIC_COLS + [TARGET_COL])
    df = _coerce_binary(df, BINARY_COLS)

    # 4) Áp quy ước: Hai Bà Trưng là trung tâm -> khoảng cách = 0
    # (Nếu dữ liệu của bạn đã tính đúng theo Hai Bà Trưng thì bước này không làm thay đổi;
    #  nếu chưa, đây là ép theo yêu cầu UI.)
    if CENTER_DISTRICT in df["Quan"].unique():
        df.loc[df["Quan"] == CENTER_DISTRICT, "Khoang_cach_TT_km"] = 0.0

    # 5) Drop rows thiếu target hoặc feature quan trọng
    df = df.dropna(subset=[TARGET_COL, "Quan", "Dien_tich_m2", "Khoang_cach_TT_km"])
    report["rows_after_dropna"] = int(len(df))

    # 6) Lọc theo miền hợp lý (đặt rộng để loại lỗi rõ ràng)
    filters = [
        ("Khoang_cach_TT_km", 0.0, 80.0),
        ("Dien_tich_m2", 10.0, 1500.0),
        ("Mat_tien_m", 0.5, 100.0),
        ("So_tang", 0.0, 80.0),
        ("So_phong_ngu", 0.0, 80.0),
        ("So_phong_tam", 0.0, 80.0),
        ("Do_rong_duong_m", 0.5, 80.0),
        ("Tuoi_nha_nam", 0.0, 300.0),
        (TARGET_COL, 0.05, 1000.0),
    ]
    for col, lo, hi in filters:
        df = df[df[col].between(lo, hi)]
    report["rows_after_range_filter"] = int(len(df))

    # 7) Ép kiểu int cho các cột đếm
    int_cols = ["So_tang", "So_phong_ngu", "So_phong_tam", "Tuoi_nha_nam"]
    for c in int_cols:
        df[c] = df[c].round().astype(int)

    # 8) Outlier filter (để giảm nhiễu & giảm dòng)
    before = len(df)
    df = _quantile_filter(df, cols=NUMERIC_COLS + [TARGET_COL], q=float(outlier_q))
    after = len(df)
    report["rows_before_outlier"] = int(before)
    report["rows_after_outlier"] = int(after)
    report["outlier_q"] = float(outlier_q)

    # 9) Lấy mẫu ~15k nếu còn quá nhiều
    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = _stratified_sample(df, by="Quan", n=max_rows, random_state=random_state)
    report["rows_after_sampling"] = int(len(df))
    report["max_rows"] = int(max_rows)

    # 10) Feature phụ trợ (không encode để tránh leakage)
    df["Gia_trieu_m2"] = (df[TARGET_COL] * 1000) / df["Dien_tich_m2"].replace(0, np.nan)

    # 11) Sắp xếp lại cột
    ordered = (
        ["Quan"]
        + ["Khoang_cach_TT_km", "Dien_tich_m2", "Mat_tien_m", "Do_rong_duong_m"]
        + ["So_tang", "So_phong_ngu", "So_phong_tam", "Tuoi_nha_nam"]
        + ["Loai_duong", "O_to_vao", "Co_Gara", "Co_San_thuong", "Gan_Metro_Bus"]
        + [
            "Mat_do_dan_cu",
            "An_ninh",
            "Gan_nghia_trang_bai_rac",
            "Tinh_trang_ngap",
            "Noi_that",
            "Tinh_trang_Dien_Nuoc",
        ]
        + [TARGET_COL, "Gia_trieu_m2"]
    )
    df = df[ordered].reset_index(drop=True)

    # 12) Thông tin số lượng quận/huyện thực sự có mẫu
    report["districts_in_data"] = sorted(df["Quan"].unique().tolist())
    report["n_districts_in_data"] = int(df["Quan"].nunique())
    report["missing_units"] = sorted(list(set(ALL_HANOI_UNITS) - set(report["districts_in_data"])))

    return df, report


def encode_data(df_clean: pd.DataFrame) -> pd.DataFrame:
    """CLEANED -> ENCODED (one-hot)."""
    df = df_clean.copy()

    # Không encode cột phụ trợ (tính từ target)
    if "Gia_trieu_m2" in df.columns:
        df = df.drop(columns=["Gia_trieu_m2"])

    # Ép Quan là categorical với đủ 30 categories để one-hot luôn đủ cột
    df["Quan"] = pd.Categorical(df["Quan"], categories=ALL_HANOI_UNITS)

    # One-hot encode
    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        prefix=CATEGORICAL_COLS,
        prefix_sep="__",
        dtype=np.uint8,
    )
    return df_encoded


def build_processed_for_training(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """ENCODED -> PROCESSED: thêm cột log(target)."""
    df = df_encoded.copy()
    df["Gia_ban_ty_log"] = np.log1p(df[TARGET_COL].astype(float))
    return df


def export_schema(df_clean: pd.DataFrame, out_path: Path, report: dict[str, Any] | None = None) -> None:
    """Xuất schema phục vụ Streamlit UI:
    - danh sách category toàn cục
    - min/max/median toàn cục
    - ràng buộc theo từng quận/huyện/thị xã (per_district)
    """
    schema: dict[str, Any] = {
        "target": TARGET_COL,
        "center_district": CENTER_DISTRICT,
        "row_count": int(df_clean.shape[0]),
        "districts_in_data": sorted(df_clean["Quan"].dropna().unique().tolist()),
        "all_hanoi_units": ALL_HANOI_UNITS,
        "categorical": {c: sorted(df_clean[c].dropna().unique().tolist()) for c in CATEGORICAL_COLS},
        "binary": BINARY_COLS,
        "numeric": {},
        "per_district": {},
        "cleaning_report": report or {},
    }

    # Global numeric summary (bao gồm cả target để tham khảo)
    for c in NUMERIC_COLS + [TARGET_COL]:
        schema["numeric"][c] = _numeric_summary(df_clean[c])

    # Per-district constraints
    for unit in ALL_HANOI_UNITS:
        sub = df_clean[df_clean["Quan"] == unit]
        entry: dict[str, Any] = {
            "n_rows": int(sub.shape[0]),
            "numeric": {},
            "categorical": {},
            "binary": {},
        }

        if sub.empty:
            # để UI có thể fallback
            for c in NUMERIC_COLS:
                entry["numeric"][c] = None
            for c in CATEGORICAL_COLS:
                entry["categorical"][c] = []
            for c in BINARY_COLS:
                entry["binary"][c] = []
        else:
            for c in NUMERIC_COLS:
                entry["numeric"][c] = _numeric_summary(sub[c])
            for c in CATEGORICAL_COLS:
                entry["categorical"][c] = sorted(sub[c].dropna().unique().tolist())
            for c in BINARY_COLS:
                vals = sorted(pd.to_numeric(sub[c], errors="coerce").dropna().astype(int).unique().tolist())
                entry["binary"][c] = vals

        # Rule đặc biệt: trung tâm -> ép khoảng cách = 0
        if unit == CENTER_DISTRICT:
            entry.setdefault("force", {})
            entry["force"]["Khoang_cach_TT_km"] = 0.0

        schema["per_district"][unit] = entry

    out_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="HN_Houseprice_Raw.csv")
    parser.add_argument("--out_clean", type=str, default="HN_Houseprice_Cleaned.csv")
    parser.add_argument("--out_encoded", type=str, default="HN_Houseprice_Encoded.csv")
    parser.add_argument("--out_processed", type=str, default="HN_Houseprice_Processed.csv")
    parser.add_argument("--out_schema", type=str, default="feature_schema.json")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=15000,
        help="Giới hạn số dòng sau làm sạch (stratified theo Quan). 0 = không giới hạn.",
    )
    parser.add_argument(
        "--outlier_q",
        type=float,
        default=0.06,
        help="Quantile lọc outlier hai phía (ví dụ 0.06). Giảm q -> giữ nhiều dữ liệu hơn.",
    )
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {in_path.resolve()}")

    print("--- 🚀 PIPELINE: RAW -> CLEANED -> ENCODED -> PROCESSED ---")
    df_raw = pd.read_csv(in_path)
    print(f"[0] RAW: {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột")

    df_clean, report = clean_data(
        df_raw,
        outlier_q=float(args.outlier_q),
        max_rows=int(args.max_rows),
        random_state=int(args.random_state),
    )
    df_clean.to_csv(args.out_clean, index=False, encoding="utf-8-sig")
    print(f"[1] CLEANED: {args.out_clean}  ({df_clean.shape[0]} dòng, {df_clean.shape[1]} cột)")
    print(f"    - Số quận/huyện/thị xã trong CLEANED: {report.get('n_districts_in_data')}")

    df_encoded = encode_data(df_clean)
    df_encoded.to_csv(args.out_encoded, index=False, encoding="utf-8-sig")
    print(f"[2] ENCODED: {args.out_encoded}  ({df_encoded.shape[0]} dòng, {df_encoded.shape[1]} cột)")

    df_processed = build_processed_for_training(df_encoded)
    df_processed.to_csv(args.out_processed, index=False, encoding="utf-8-sig")
    print(f"[3] PROCESSED: {args.out_processed}  ({df_processed.shape[0]} dòng, {df_processed.shape[1]} cột)")

    export_schema(df_clean, Path(args.out_schema), report=report)
    print(f"[4] SCHEMA: {args.out_schema}")

    # Xuất report riêng (tiện debug)
    Path("cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[5] REPORT: cleaning_report.json")

    print("✅ DONE!")


if __name__ == "__main__":
    main()