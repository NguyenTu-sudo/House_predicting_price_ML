import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================
# PAGE CONFIG (phải đặt sớm nhất)
# ============================
st.set_page_config(page_title="Dự đoán giá nhà Hà Nội", page_icon="🏠", layout="wide")

# ============================
# CONFIG
# ============================
TARGET_COL = "Gia_ban_ty"

CATEGORICAL_COLS = [
    "Quan",
    "Loai_duong",
    "Mat_do_dan_cu",
    "An_ninh",
    "Tinh_trang_ngap",
    "Noi_that",
    "Tinh_trang_Dien_Nuoc",
]

BINARY_COLS = [
    "O_to_vao",
    "Co_Gara",
    "Co_San_thuong",
    "Gan_Metro_Bus",
    "Gan_nghia_trang_bai_rac",
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

# Danh sách đầy đủ theo tài liệu ràng buộc + app cũ
ALL_DISTRICTS = [
    "Ba Vì",
    "Ba Đình",
    "Bắc Từ Liêm",
    "Chương Mỹ",
    "Cầu Giấy",
    "Gia Lâm",
    "Hai Bà Trưng",
    "Hoài Đức",
    "Hoàn Kiếm",
    "Hoàng Mai",
    "Hà Đông",
    "Long Biên",
    "Mê Linh",
    "Mỹ Đức",
    "Nam Từ Liêm",
    "Phú Xuyên",
    "Phúc Thọ",
    "Quốc Oai",
    "Sóc Sơn",
    "Sơn Tây",
    "Thanh Oai",
    "Thanh Trì",
    "Thanh Xuân",
    "Thường Tín",
    "Thạch Thất",
    "Tây Hồ",
    "Đan Phượng",
    "Đông Anh",
    "Đống Đa",
    "Ứng Hòa",
]

# ============================
# RÀNG BUỘC THEO DOCX
# (Các giá trị có dấu ❌ trong doc -> coi là KHÔNG cho chọn)
# ============================
GROUP_RULES = [
    # ============================
    # RÀNG BUỘC THEO NHÓM ĐỊA LÝ (Hà Nội có 30 đơn vị cấp huyện)
    # Quy ước:
    # - disallow: giá trị ❌ không cho chọn
    # - force: ép giá trị (UI chỉ còn 1 lựa chọn)
    # - num_range: giới hạn min/max cho biến số
    # - defaults: gợi ý mặc định
    # ============================

    # 1) Lõi trung tâm: mật độ cao, đường đất & ngập sâu gần như không phù hợp
    {
        "name": "Quận lõi trung tâm",
        "districts": ["Hoàn Kiếm", "Ba Đình", "Đống Đa", "Hai Bà Trưng"],
        "disallow": {
            "Mat_do_dan_cu": {"Thấp"},
            "Tinh_trang_ngap": {"Ngập sâu"},
            "Loai_duong": {"Đường đất"},
            "An_ninh": {"Kém (Hay mất trộm)"},
        },
        "num_range": {"Khoang_cach_TT_km": (0.1, 6.0)},
        "defaults": {"Gan_Metro_Bus": 1, "Mat_do_dan_cu": "Cao", "An_ninh": "Tốt"},
    },

    # 2) Nội thành mở rộng: vẫn đô thị, nhưng xa hơn lõi
    {
        "name": "Quận nội thành mở rộng",
        "districts": ["Cầu Giấy", "Thanh Xuân", "Tây Hồ", "Hoàng Mai", "Long Biên"],
        "disallow": {
            "Tinh_trang_ngap": {"Ngập sâu"},
            "Loai_duong": {"Đường đất"},
        },
        "num_range": {"Khoang_cach_TT_km": (3.0, 14.0)},
        "defaults": {"Gan_Metro_Bus": 1, "Mat_do_dan_cu": "Cao"},
    },

    # 3) Đô thị mới/giãn dân: trục phía Tây – Tây Nam
    {
        "name": "Quận đô thị mới (Tây/Tây Nam)",
        "districts": ["Nam Từ Liêm", "Bắc Từ Liêm", "Hà Đông"],
        "disallow": {
            "Tinh_trang_ngap": {"Ngập sâu"},
            "Loai_duong": {"Đường đất"},
        },
        "num_range": {"Khoang_cach_TT_km": (6.0, 22.0)},
        "defaults": {"Gan_Metro_Bus": 1, "Mat_do_dan_cu": "Trung bình", "An_ninh": "Tốt"},
    },

    # 4) Vành đai phía Đông & khu vực bắc sông: đô thị hóa mạnh
    {
        "name": "Đông & bắc sông (đô thị hóa mạnh)",
        "districts": ["Gia Lâm", "Đông Anh"],
        "disallow": {
            "Tinh_trang_ngap": {"Ngập sâu"},
            "Loai_duong": {"Đường đất"},
        },
        "num_range": {"Khoang_cach_TT_km": (8.0, 28.0)},
        "defaults": {"Gan_Metro_Bus": 0, "Mat_do_dan_cu": "Trung bình"},
    },

    # 5) Phía Bắc xa hơn (gần sân bay / vệ tinh): mật độ thường trung bình - thấp
    {
        "name": "Phía Bắc (Sóc Sơn/Mê Linh)",
        "districts": ["Sóc Sơn", "Mê Linh"],
        "disallow": {
            "Mat_do_dan_cu": {"Cao"},
            "An_ninh": {"Rất tốt (VIP)"},
        },
        "num_range": {"Khoang_cach_TT_km": (18.0, 45.0)},
        "defaults": {"Gan_Metro_Bus": 0, "Mat_do_dan_cu": "Trung bình"},
    },

    # 6) Vành đai phía Tây (cận đô, đô thị hoá nhanh)
    {
        "name": "Phía Tây cận đô (Hoài Đức/Đan Phượng/Phúc Thọ/Quốc Oai/Thạch Thất)",
        "districts": ["Hoài Đức", "Đan Phượng", "Phúc Thọ", "Quốc Oai", "Thạch Thất"],
        "disallow": {
            "Tinh_trang_ngap": {"Ngập sâu"},
        },
        "num_range": {"Khoang_cach_TT_km": (12.0, 38.0)},
        "defaults": {"Gan_Metro_Bus": 0, "Mat_do_dan_cu": "Trung bình", "An_ninh": "Bình thường"},
    },

    # 7) Thị xã/vùng vệ tinh phía Tây (Sơn Tây) & vùng đồi núi (Ba Vì)
    {
        "name": "Vệ tinh phía Tây (Sơn Tây/Ba Vì)",
        "districts": ["Sơn Tây", "Ba Vì"],
        "disallow": {
            "Mat_do_dan_cu": {"Cao"},
            "An_ninh": {"Rất tốt (VIP)"},
        },
        "num_range": {"Khoang_cach_TT_km": (30.0, 65.0)},
        "defaults": {"Gan_Metro_Bus": 0, "Mat_do_dan_cu": "Trung bình", "Loai_duong": "Đường bê tông"},
    },

    # 8) Hành lang phía Nam gần (áp lực ngập cao hơn -> tránh ngập sâu)
    {
        "name": "Phía Nam gần (Thanh Trì/Thanh Oai/Thường Tín/Chương Mỹ)",
        "districts": ["Thanh Trì", "Thanh Oai", "Thường Tín", "Chương Mỹ"],
        "disallow": {
            "Tinh_trang_ngap": {"Ngập sâu"},
            "An_ninh": {"Rất tốt (VIP)"},
        },
        "num_range": {"Khoang_cach_TT_km": (12.0, 45.0)},
        "defaults": {"Gan_Metro_Bus": 0, "Mat_do_dan_cu": "Trung bình"},
    },

    # 9) Phía Nam xa (thuần nông hơn)
    {
        "name": "Phía Nam xa (Mỹ Đức/Phú Xuyên/Ứng Hòa)",
        "districts": ["Mỹ Đức", "Phú Xuyên", "Ứng Hòa"],
        "disallow": {
            "Mat_do_dan_cu": {"Cao"},
            "An_ninh": {"Rất tốt (VIP)"},
        },
        "num_range": {"Khoang_cach_TT_km": (30.0, 80.0)},
        "defaults": {"Gan_Metro_Bus": 0, "Mat_do_dan_cu": "Thấp", "Loai_duong": "Đường bê tông"},
    },
]

# Ràng buộc bổ sung (doc có đoạn “quận nội thành”)
DISTRICT_OVERRIDES = {
    # Ba Đình + Hoàn Kiếm + Tây Hồ: đường quá hẹp / ô tô không vào bị coi là ngoại lệ theo doc
    "Ba Đình": {"force": {"O_to_vao": 1}, "num_min": {"Do_rong_duong_m": 2.5}},
    "Hoàn Kiếm": {"force": {"O_to_vao": 1}, "num_min": {"Do_rong_duong_m": 2.5}},
    "Tây Hồ": {"force": {"O_to_vao": 1}, "num_min": {"Do_rong_duong_m": 2.5}},
}


# ============================
# HELPERS
# ============================
@st.cache_resource
def load_artifacts():
    """Load model + feature list + schema (nếu có)."""

    model = None
    feature_cols = None

    # Ưu tiên best_model.pkl (tạo bởi model_training.py)
    for candidate in ["best_model.pkl", "best_rf_model.pkl", "gia_nha_model.joblib"]:
        if Path(candidate).exists():
            model = joblib.load(candidate)
            break

    if Path("model_features.pkl").exists():
        feature_cols = joblib.load("model_features.pkl")

    schema = None
    if Path("feature_schema.json").exists():
        schema = json.loads(Path("feature_schema.json").read_text(encoding="utf-8"))

    return model, feature_cols, schema


def find_group(district: str) -> str:
    for rule in GROUP_RULES:
        if district in rule.get("districts", []):
            return rule["name"]
    return "(Chưa phân nhóm trong tài liệu)"


def default_schema_fallback():
    """Fallback nếu chưa có feature_schema.json."""
    return {
        "categorical": {
            "Loai_duong": ["Đường nhựa", "Đường bê tông", "Đường đất"],
            "Mat_do_dan_cu": ["Thấp", "Trung bình", "Cao"],
            "An_ninh": ["Kém (Hay mất trộm)", "Bình thường", "Tốt", "Rất tốt (VIP)"],
            "Tinh_trang_ngap": ["Không ngập", "Ngập nhẹ", "Ngập sâu"],
            "Noi_that": ["Nhà trống", "Cơ bản", "Đồ gỗ xịn", "Full cao cấp"],
            "Tinh_trang_Dien_Nuoc": ["Hay hỏng", "Tốt"],
            "Quan": ALL_DISTRICTS,
        },
        "numeric": {
            "Khoang_cach_TT_km": {"min": 0.1, "max": 60.0, "median": 8.0},
            "Dien_tich_m2": {"min": 10, "max": 1000, "median": 60},
            "Mat_tien_m": {"min": 1.0, "max": 50.0, "median": 6.0},
            "So_tang": {"min": 1, "max": 50, "median": 4},
            "So_phong_ngu": {"min": 1, "max": 50, "median": 4},
            "So_phong_tam": {"min": 1, "max": 50, "median": 3},
            "Do_rong_duong_m": {"min": 0.5, "max": 50.0, "median": 6.0},
            "Tuoi_nha_nam": {"min": 0, "max": 200, "median": 15},
        },
        "binary": BINARY_COLS,
        "districts": [],
    }


def build_constraints(district: str, schema: dict) -> dict:
    """Từ district -> trả ra:
    - allowed_cat: danh sách giá trị được chọn cho từng biến phân loại
    - allowed_bin: [0,1] hoặc bị bó hẹp theo district
    - num_range : min/max/default cho biến số (ưu tiên theo district, fallback global)
    - forced   : các giá trị bị ép
    - defaults : gợi ý mặc định
    - disabled : những biến nên khoá UI (do bị ép hoặc chỉ còn 1 lựa chọn)
    """

    schema = schema or default_schema_fallback()

    per = (schema.get("per_district", {}) or {}).get(district, {}) or {}
    per_cat = per.get("categorical", {}) or {}
    per_num = per.get("numeric", {}) or {}
    per_bin = per.get("binary", {}) or {}

    # ----------------------------
    # Base options (ưu tiên theo district)
    # ----------------------------
    allowed_cat: dict[str, list] = {}
    for c in CATEGORICAL_COLS:
        # nếu district có dữ liệu -> lấy unique theo district
        vals = list(per_cat.get(c, []))
        if not vals:
            vals = list((schema.get("categorical", {}) or {}).get(c, []))
        allowed_cat[c] = vals

    allowed_bin: dict[str, list[int]] = {}
    for c in BINARY_COLS:
        vals = per_bin.get(c, [])
        if isinstance(vals, list) and len(vals) > 0:
            allowed_bin[c] = [int(v) for v in vals]
        else:
            allowed_bin[c] = [0, 1]

    num_range: dict[str, dict[str, float]] = {}
    for c in NUMERIC_COLS:
        info = per_num.get(c)
        if isinstance(info, dict) and all(k in info for k in ["q05", "q95", "median"]):
            lo, hi, med = float(info["q05"]), float(info["q95"]), float(info["median"])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
                info = None
        if not info:
            g = (schema.get("numeric", {}) or {}).get(c, {}) or {}
            lo = float(g.get("min", 0.0))
            hi = float(g.get("max", 1.0))
            med = float(g.get("median", (lo + hi) / 2))
        num_range[c] = {"min": float(lo), "max": float(hi), "default": float(med)}

    forced: dict[str, object] = {}
    defaults: dict[str, object] = {}

    # Forced theo district (từ schema)
    if isinstance(per.get("force"), dict):
        for k, v in per["force"].items():
            forced[k] = v

    applied_rule = None

    # ----------------------------
    # Apply group rule (intersection – không ghi đè bừa)
    # ----------------------------
    for rule in GROUP_RULES:
        if district not in rule.get("districts", []):
            continue
        applied_rule = rule

        # Disallow values (lọc ra khỏi allowed list)
        for k, bad_set in (rule.get("disallow", {}) or {}).items():
            if k in allowed_cat and isinstance(bad_set, (set, list, tuple)):
                allowed_cat[k] = [x for x in allowed_cat[k] if x not in set(bad_set)]
            if k in allowed_bin and isinstance(bad_set, (set, list, tuple)):
                allowed_bin[k] = [x for x in allowed_bin[k] if x not in set(bad_set)]

        # Numeric range intersection
        for k, (lo2, hi2) in (rule.get("num_range", {}) or {}).items():
            if k in num_range:
                num_range[k]["min"] = max(num_range[k]["min"], float(lo2))
                num_range[k]["max"] = min(num_range[k]["max"], float(hi2))
                if num_range[k]["min"] > num_range[k]["max"]:
                    # nếu giao bị rỗng -> fallback về range global
                    g = (schema.get("numeric", {}) or {}).get(k, {}) or {}
                    num_range[k]["min"] = float(g.get("min", 0.0))
                    num_range[k]["max"] = float(g.get("max", 1.0))
                # kéo default về trong range
                num_range[k]["default"] = float(max(min(num_range[k]["default"], num_range[k]["max"]), num_range[k]["min"]))

        # Default suggestions
        for k, v in (rule.get("defaults", {}) or {}).items():
            defaults[k] = v

        break  # 1 district thuộc 1 nhóm

    # ----------------------------
    # Apply district overrides (nếu có)
    # ----------------------------
    override = DISTRICT_OVERRIDES.get(district)
    if override:
        for k, v in (override.get("force", {}) or {}).items():
            forced[k] = v
        for k, v in (override.get("num_min", {}) or {}).items():
            if k in num_range:
                num_range[k]["min"] = max(num_range[k]["min"], float(v))
                num_range[k]["default"] = max(num_range[k]["default"], num_range[k]["min"])

    # ----------------------------
    # Force -> overwrite allowed options + disable UI
    # ----------------------------
    disabled: set[str] = set()
    for k, v in forced.items():
        if k in allowed_cat:
            allowed_cat[k] = [v]
            disabled.add(k)
        if k in allowed_bin:
            allowed_bin[k] = [int(v)]
            disabled.add(k)
        if k in num_range:
            # nếu ép numeric (ví dụ center district)
            try:
                fv = float(v)
                num_range[k]["min"] = fv
                num_range[k]["max"] = fv
                num_range[k]["default"] = fv
                disabled.add(k)
            except Exception:
                pass

    # Nếu sau disallow mà rỗng thì fallback global
    for c in CATEGORICAL_COLS:
        if c in allowed_cat and len(allowed_cat[c]) == 0:
            allowed_cat[c] = list((schema.get("categorical", {}) or {}).get(c, []))

    for c in BINARY_COLS:
        if c in allowed_bin and len(allowed_bin[c]) == 0:
            allowed_bin[c] = [0, 1]

    # Tự disable khi chỉ còn 1 lựa chọn
    for c in CATEGORICAL_COLS:
        if c in allowed_cat and len(allowed_cat[c]) == 1:
            disabled.add(c)
    for c in BINARY_COLS:
        if c in allowed_bin and len(allowed_bin[c]) == 1:
            disabled.add(c)
    for c in NUMERIC_COLS:
        if c in num_range and float(num_range[c]["min"]) == float(num_range[c]["max"]):
            disabled.add(c)

    return {
        "allowed_cat": allowed_cat,
        "allowed_bin": allowed_bin,
        "num_range": num_range,
        "forced": forced,
        "defaults": defaults,
        "disabled": disabled,
        "applied_rule": applied_rule,
    }


def ensure_select(key: str, options: list, default=None):
    if not options:
        options = [default] if default is not None else [None]

    if key not in st.session_state:
        st.session_state[key] = default if default in options else options[0]
    else:
        if st.session_state[key] not in options:
            st.session_state[key] = default if default in options else options[0]


def ensure_number(key: str, minv: float, maxv: float, default: float):
    if key not in st.session_state:
        st.session_state[key] = default
    try:
        v = float(st.session_state[key])
    except Exception:
        v = default

    v = max(min(v, maxv), minv)
    st.session_state[key] = v


def encode_input(record: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Encode 1 record raw -> vector đúng thứ tự cột model_features.pkl"""
    df = pd.DataFrame([record])

    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        prefix=CATEGORICAL_COLS,
        prefix_sep="__",
    )

    # Bổ sung cột còn thiếu
    for c in feature_cols:
        if c not in df_encoded.columns:
            df_encoded[c] = 0

    # Loại bỏ cột thừa nếu có
    df_encoded = df_encoded[feature_cols]

    return df_encoded


def predict_price(model, feature_cols: list[str], record: dict) -> float:
    X = encode_input(record, feature_cols)
    pred_log = float(model.predict(X)[0])
    return float(np.expm1(pred_log))


def build_12m_forecast_series(base_price_ty: float) -> "pd.DataFrame":
    """Tạo chuỗi dự báo 12 tháng cho giá (tỷ VNĐ) theo 2 kịch bản vĩ mô.

    Ghi chú:
    - Đây là *điều chỉnh theo tốc độ tăng giá dự kiến của thị trường* (không phải
      mô hình time-series)
    - Dùng tỉ lệ tăng trưởng năm 2026 từ báo cáo CBRE:
        * Nhà đất (landed) - secondary: ~+3%/năm
        * Chung cư (condo) - secondary: ~+6%/năm
    """
    import pandas as pd

    base = float(base_price_ty)
    months = 12

    # Growth assumptions (annual)
    annual_landed = 0.03
    annual_condo = 0.06

    def comp_monthly(a: float) -> float:
        return (1.0 + a) ** (1.0 / 12.0) - 1.0

    m_landed = comp_monthly(annual_landed)
    m_condo = comp_monthly(annual_condo)

    start = pd.Timestamp.today().normalize().replace(day=1)
    idx = pd.date_range(start=start, periods=months + 1, freq="MS")

    landed = [base * ((1 + m_landed) ** i) for i in range(months + 1)]
    condo = [base * ((1 + m_condo) ** i) for i in range(months + 1)]

    out = pd.DataFrame(
        {
            "Nhà đất (CBRE secondary ~3%/năm)": landed,
            "Chung cư (CBRE secondary ~6%/năm)": condo,
        },
        index=idx,
    )
    return out


# ============================
# UI
# ============================
st.markdown(
    """
<style>
    .main-header{font-size:2.3rem;font-weight:800;text-align:center;color:#1E88E5;margin-bottom:0.2rem}
    .sub-header{font-size:1.0rem;text-align:center;color:#666;margin-bottom:1.2rem}
    .result-box{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1.4rem;border-radius:14px;text-align:center;margin:1.2rem 0}
    .result-price{font-size:2.6rem;font-weight:800;color:white;margin:0}
    .result-label{font-size:1.1rem;color:rgba(255,255,255,0.9);margin:0}
    .mini-note{color:#777;font-size:0.9rem}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🏠 Dự đoán Giá Nhà Hà Nội (Full thuộc tính)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hiển thị đầy đủ thuộc tính train + ràng buộc theo Quận/Huyện/Thị xã</div>', unsafe_allow_html=True)

model, feature_cols, schema = load_artifacts()

if model is None or feature_cols is None:
    st.error("⚠️ Chưa có model/feature list. Hãy chạy theo thứ tự:")
    st.code("python preprocessing.py\npython model_training.py --sample 15000", language="bash")
    st.stop()

trained_districts = (
    schema.get("districts_in_data", schema.get("districts", [])) if schema else []
)
all_units = schema.get("all_hanoi_units", ALL_DISTRICTS) if schema else ALL_DISTRICTS

if trained_districts:
    st.success(
        "✅ Model đã sẵn sàng. "
        f"(Giao diện hỗ trợ {len(all_units)} quận/huyện/thị xã; "
        f"dữ liệu train hiện có {len(trained_districts)}: {', '.join(trained_districts)})"
    )
else:
    st.success(f"✅ Model đã sẵn sàng. (Giao diện hỗ trợ {len(all_units)} quận/huyện/thị xã)")

# ----------------------------
# District selector
# ----------------------------
ensure_select("quan", ALL_DISTRICTS, default=(trained_districts[0] if trained_districts else "Cầu Giấy"))
quan = st.selectbox("🏙️ Quận/Huyện/Thị xã", options=ALL_DISTRICTS, key="quan")

if trained_districts and quan not in trained_districts:
    st.warning(
        "Khu vực bạn chọn **chưa có trong dữ liệu train**. "
        "Model vẫn chạy được (do đã tạo đủ 30 cột one-hot), nhưng ảnh hưởng riêng của "
        "quận/huyện này gần như **chưa được học** => dự đoán có thể kém chính xác."
    )

constraints = build_constraints(quan, schema)

# Summary
with st.expander("📌 Xem ràng buộc đang áp dụng", expanded=False):
    group_name = find_group(quan)
    st.markdown(f"**Nhóm đô thị:** {group_name}")

    # Show disallow per group
    applied_rule = next((r for r in GROUP_RULES if quan in r.get("districts", [])), None)
    if applied_rule and applied_rule.get("disallow"):
        st.markdown("**Giá trị bị chặn (❌):**")
        for k, bad in applied_rule["disallow"].items():
            st.write(f"- {k}: {', '.join(map(str, sorted(list(bad))))}")
    if constraints.get("forced"):
        st.markdown("**Giá trị bị ép (force):**")
        for k, v in constraints["forced"].items():
            st.write(f"- {k} = {v}")
    if applied_rule and applied_rule.get("num_range"):
        st.markdown("**Miền số khuyến nghị/áp dụng:**")
        for k, (lo, hi) in applied_rule["num_range"].items():
            st.write(f"- {k}: {lo} → {hi}")

st.markdown("---")

# ----------------------------
# Input widgets (FULL FEATURES)
# ----------------------------
colA, colB, colC = st.columns(3)

# ---- Column A: Vị trí & kích thước ----
with colA:
    st.markdown("### 📍 Vị trí & kích thước")

    center_district = (schema or {}).get("center_district", "Hai Bà Trưng")

    if quan == center_district:
        # Theo yêu cầu: Quận trung tâm không cần nhập "khoảng cách"
        Khoang_cach_TT_km = 0.0
        st.session_state["khoang_cach"] = 0.0
        st.markdown("📌 Khoảng cách tới trung tâm (km): **0.0** (tự động – khu vực trung tâm)")
    else:
        r = constraints["num_range"]["Khoang_cach_TT_km"]
        ensure_number(
            "khoang_cach",
            r["min"],
            r["max"],
            constraints["defaults"].get("Khoang_cach_TT_km", r["default"]),
        )
        Khoang_cach_TT_km = st.number_input(
            "📌 Khoảng cách tới trung tâm (km)",
            min_value=float(r["min"]),
            max_value=float(r["max"]),
            value=float(st.session_state["khoang_cach"]),
            step=0.1,
            key="khoang_cach",
        )

    r = constraints["num_range"]["Dien_tich_m2"]
    ensure_number("dien_tich", r["min"], r["max"], r["default"])
    Dien_tich_m2 = st.number_input(
        "📐 Diện tích (m²)",
        min_value=int(r["min"]),
        max_value=int(r["max"]),
        value=int(st.session_state["dien_tich"]),
        step=1,
        key="dien_tich",
    )

    r = constraints["num_range"]["Mat_tien_m"]
    ensure_number("mat_tien", r["min"], r["max"], r["default"])
    Mat_tien_m = st.number_input(
        "↔️ Mặt tiền (m)",
        min_value=float(r["min"]),
        max_value=float(r["max"]),
        value=float(st.session_state["mat_tien"]),
        step=0.1,
        key="mat_tien",
    )

    r = constraints["num_range"]["Do_rong_duong_m"]
    ensure_number("do_rong_duong", r["min"], r["max"], r["default"])
    Do_rong_duong_m = st.number_input(
        "🛣️ Độ rộng đường/ngõ (m)",
        min_value=float(r["min"]),
        max_value=float(r["max"]),
        value=float(st.session_state["do_rong_duong"]),
        step=0.1,
        key="do_rong_duong",
    )

    ensure_select(
        "loai_duong",
        constraints["allowed_cat"]["Loai_duong"],
        default=("Đường nhựa" if "Đường nhựa" in constraints["allowed_cat"]["Loai_duong"] else None),
    )
    Loai_duong = st.selectbox(
        "🛣️ Loại đường",
        options=constraints["allowed_cat"]["Loai_duong"],
        key="loai_duong",
        disabled=("Loai_duong" in constraints["disabled"]),
    )

# ---- Column B: Cấu trúc nhà ----
with colB:
    st.markdown("### 🏗️ Cấu trúc nhà")

    r = constraints["num_range"]["So_tang"]
    ensure_number("so_tang", r["min"], r["max"], r["default"])
    So_tang = st.number_input(
        "🏢 Số tầng",
        min_value=int(r["min"]),
        max_value=int(r["max"]),
        value=int(st.session_state["so_tang"]),
        step=1,
        key="so_tang",
    )

    r = constraints["num_range"]["So_phong_ngu"]
    ensure_number("so_phong_ngu", r["min"], r["max"], r["default"])
    So_phong_ngu = st.number_input(
        "🛏️ Số phòng ngủ",
        min_value=int(r["min"]),
        max_value=int(r["max"]),
        value=int(st.session_state["so_phong_ngu"]),
        step=1,
        key="so_phong_ngu",
    )

    r = constraints["num_range"]["So_phong_tam"]
    ensure_number("so_phong_tam", r["min"], r["max"], r["default"])
    So_phong_tam = st.number_input(
        "🛁 Số phòng tắm",
        min_value=int(r["min"]),
        max_value=int(r["max"]),
        value=int(st.session_state["so_phong_tam"]),
        step=1,
        key="so_phong_tam",
    )

    r = constraints["num_range"]["Tuoi_nha_nam"]
    ensure_number("tuoi_nha", r["min"], r["max"], r["default"])
    Tuoi_nha_nam = st.number_input(
        "🕰️ Tuổi nhà (năm)",
        min_value=int(r["min"]),
        max_value=int(r["max"]),
        value=int(st.session_state["tuoi_nha"]),
        step=1,
        key="tuoi_nha",
    )

    ensure_select(
        "noi_that",
        constraints["allowed_cat"]["Noi_that"],
        default=("Cơ bản" if "Cơ bản" in constraints["allowed_cat"]["Noi_that"] else None),
    )
    Noi_that = st.selectbox(
        "🛋️ Nội thất",
        options=constraints["allowed_cat"]["Noi_that"],
        key="noi_that",
        disabled=("Noi_that" in constraints["disabled"]),
    )

    ensure_select(
        "dien_nuoc",
        constraints["allowed_cat"]["Tinh_trang_Dien_Nuoc"],
        default=("Tốt" if "Tốt" in constraints["allowed_cat"]["Tinh_trang_Dien_Nuoc"] else None),
    )
    Tinh_trang_Dien_Nuoc = st.selectbox(
        "⚡🚰 Tình trạng điện nước",
        options=constraints["allowed_cat"]["Tinh_trang_Dien_Nuoc"],
        key="dien_nuoc",
        disabled=("Tinh_trang_Dien_Nuoc" in constraints["disabled"]),
    )

# ---- Column C: Tiện ích & môi trường ----
with colC:
    st.markdown("### 🧩 Tiện ích & môi trường")

    # Binary widgets helper
    def bin_select(label, key, feature_name, default=1):
        ensure_select(key, constraints["allowed_bin"][feature_name], default=constraints["defaults"].get(feature_name, default))
        return st.selectbox(
            label,
            options=constraints["allowed_bin"][feature_name],
            key=key,
            disabled=(feature_name in constraints["disabled"]),
            format_func=lambda x: "Có" if int(x) == 1 else "Không",
        )

    O_to_vao = bin_select("🚗 Ô tô vào", "o_to_vao", "O_to_vao", default=1)
    Co_Gara = bin_select("🅿️ Có gara", "co_gara", "Co_Gara", default=0)
    Co_San_thuong = bin_select("🌤️ Có sân thượng", "co_san_thuong", "Co_San_thuong", default=1)
    Gan_Metro_Bus = bin_select("🚇🚌 Gần metro/bus", "gan_metro", "Gan_Metro_Bus", default=0)
    Gan_nghia_trang_bai_rac = bin_select("⚠️ Gần nghĩa trang/bãi rác", "gan_bai_rac", "Gan_nghia_trang_bai_rac", default=0)

    ensure_select(
        "mat_do",
        constraints["allowed_cat"]["Mat_do_dan_cu"],
        default=("Trung bình" if "Trung bình" in constraints["allowed_cat"]["Mat_do_dan_cu"] else None),
    )
    Mat_do_dan_cu = st.selectbox(
        "👥 Mật độ dân cư",
        options=constraints["allowed_cat"]["Mat_do_dan_cu"],
        key="mat_do",
        disabled=("Mat_do_dan_cu" in constraints["disabled"]),
    )

    ensure_select(
        "an_ninh",
        constraints["allowed_cat"]["An_ninh"],
        default=("Tốt" if "Tốt" in constraints["allowed_cat"]["An_ninh"] else None),
    )
    An_ninh = st.selectbox(
        "🛡️ An ninh",
        options=constraints["allowed_cat"]["An_ninh"],
        key="an_ninh",
        disabled=("An_ninh" in constraints["disabled"]),
    )

    ensure_select(
        "ngap",
        constraints["allowed_cat"]["Tinh_trang_ngap"],
        default=("Không ngập" if "Không ngập" in constraints["allowed_cat"]["Tinh_trang_ngap"] else None),
    )
    Tinh_trang_ngap = st.selectbox(
        "🌧️ Tình trạng ngập",
        options=constraints["allowed_cat"]["Tinh_trang_ngap"],
        key="ngap",
        disabled=("Tinh_trang_ngap" in constraints["disabled"]),
    )

st.markdown("---")

# ----------------------------
# Predict
# ----------------------------
record = {
    "Quan": quan,
    "Khoang_cach_TT_km": float(Khoang_cach_TT_km),
    "Dien_tich_m2": int(Dien_tich_m2),
    "Mat_tien_m": float(Mat_tien_m),
    "So_tang": int(So_tang),
    "So_phong_ngu": int(So_phong_ngu),
    "So_phong_tam": int(So_phong_tam),
    "Do_rong_duong_m": float(Do_rong_duong_m),
    "Loai_duong": Loai_duong,
    "O_to_vao": int(O_to_vao),
    "Co_Gara": int(Co_Gara),
    "Co_San_thuong": int(Co_San_thuong),
    "Gan_Metro_Bus": int(Gan_Metro_Bus),
    "Mat_do_dan_cu": Mat_do_dan_cu,
    "An_ninh": An_ninh,
    "Gan_nghia_trang_bai_rac": int(Gan_nghia_trang_bai_rac),
    "Tinh_trang_ngap": Tinh_trang_ngap,
    "Noi_that": Noi_that,
    "Tinh_trang_Dien_Nuoc": Tinh_trang_Dien_Nuoc,
    "Tuoi_nha_nam": int(Tuoi_nha_nam),
}

btn_col1, btn_col2 = st.columns([1, 3])
with btn_col1:
    run = st.button("🔮 DỰ ĐOÁN GIÁ", type="primary", use_container_width=True)

if run:
    try:
        with st.spinner("Đang dự đoán..."):
            price_ty = predict_price(model, feature_cols, record)

        st.markdown(
            f"""
<div class="result-box">
  <p class="result-label">💰 Giá dự kiến</p>
  <p class="result-price">{price_ty:,.2f} tỷ VNĐ</p>
</div>
""",
            unsafe_allow_html=True,
        )

        price_trieu_m2 = (price_ty * 1000) / max(record["Dien_tich_m2"], 1)

        st.markdown("### 📊 Tóm tắt")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("📍 Khu vực", record["Quan"])
            st.metric("📌 Cách trung tâm", f"{record['Khoang_cach_TT_km']:.1f} km")
        with c2:
            st.metric("📐 Diện tích", f"{record['Dien_tich_m2']} m²")
            st.metric("💵 Giá / m²", f"{price_trieu_m2:,.0f} triệu")
        with c3:
            st.metric("🏢 Số tầng", f"{record['So_tang']}")
            st.metric("🛏️ Phòng ngủ", f"{record['So_phong_ngu']}")

        st.info("ℹ️ Giá dự đoán chỉ mang tính tham khảo. Thực tế còn phụ thuộc vị trí ngõ, pháp lý, quy hoạch, thời điểm thị trường...")

        # 12-month forecast chart (macro-based)
        st.markdown("### 📈 Dự báo 12 tháng tới (tham khảo)")
        fc = build_12m_forecast_series(price_ty)
        st.line_chart(fc)
        st.caption(
            "Dự báo này là điều chỉnh vĩ mô theo tỉ lệ tăng giá dự kiến của thị trường (không phải mô hình time-series). "
            "Bạn có thể dùng như một *kịch bản tham khảo* để so sánh."
        )

    except Exception as e:
        st.error(f"❌ Lỗi dự đoán: {e}")

st.markdown("---")
st.caption("Dữ liệu/feature được load từ pipeline mới (HN_Houseprice_Raw.csv → preprocessing.py → model_training.py).")
