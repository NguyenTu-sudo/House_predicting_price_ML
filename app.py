import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Optional


# =========================
#  Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "feature_schema.json"
DEFAULT_MODEL_PATH = BASE_DIR / "best_model.pkl"
MODELS_DIR = BASE_DIR / "models"  # (tuỳ chọn) chứa nhiều mô hình
MODEL_INFO_PATH = BASE_DIR / "model_info.json"


# =========================
#  UI helpers
# =========================
def inject_css() -> None:
    st.markdown(
        """
<style>
/* App background */
.stApp{
    background: radial-gradient(circle at 10% 20%, rgba(250,250,255,1) 0%, rgba(242,247,255,1) 35%, rgba(245,245,245,1) 100%);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(18, 38, 72, 1) 0%, rgba(22, 56, 104, 1) 45%, rgba(18, 38, 72, 1) 100%);
}
section[data-testid="stSidebar"] *{
    color: #ffffff !important;
}

/* Title */
h1{
    font-size: 2.1rem !important;
    letter-spacing: 0.2px;
}
h2,h3{
    letter-spacing: 0.2px;
}

/* Cards (metrics) */
div[data-testid="stMetric"]{
    background: rgba(255,255,255,0.80);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 10px 22px rgba(0,0,0,0.06);
}

/* Buttons */
.stButton>button{
    border-radius: 14px;
    padding: 0.65rem 1.1rem;
    border: none;
    background: linear-gradient(90deg, rgba(52,120,246,1) 0%, rgba(122,80,255,1) 55%, rgba(255,86,176,1) 100%);
    color: #fff;
    font-weight: 700;
    box-shadow: 0 12px 26px rgba(52,120,246,0.25);
}
.stButton>button:hover{
    transform: translateY(-1px);
    box-shadow: 0 16px 32px rgba(52,120,246,0.30);
}

/* Input widgets */
div[data-baseweb="select"]>div,
div[data-baseweb="input"]>div{
    border-radius: 12px !important;
}
</style>
<style>
/* Sửa màu input number trong sidebar ở light mode */
section[data-testid="stSidebar"] input[type="number"] {
    color: #222 !important;
    background: #fff !important;
    font-weight: 600;
}
</style>
""",
        unsafe_allow_html=True,
    )


def load_schema():
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy schema tại: {SCHEMA_PATH}. Hãy chạy: python preprocessing.py"
        )
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(path_str: str):
    """Load model (joblib) theo đường dẫn.

    Dùng cache để tránh load lại liên tục khi UI rerun.
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {p}. Hãy chạy: python model_training.py")
    return joblib.load(p)


def fmt_ty(x: float) -> str:
    return f"{x:,.3f} tỷ"


def fmt_vnd(x_ty: float) -> str:
    vnd = x_ty * 1_000_000_000
    return f"{vnd:,.0f} ₫"


def annual_to_monthly(r_annual: float) -> float:
    """Chuyển tăng trưởng theo năm -> theo tháng (lãi kép).

    Công thức:
        r_tháng = (1 + r_năm)^(1/12) - 1
    """
    return (1.0 + float(r_annual)) ** (1.0 / 12.0) - 1.0


def generate_forecast_12m(
    current_price_ty: float,
    annual_rates: dict,
    start_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Tạo chuỗi dự báo 12 tháng theo tăng trưởng năm -> tháng.

    - Dùng đúng công thức lãi kép theo tháng từ r_năm.
    - Nếu dự báo băng qua ranh giới năm, mỗi tháng sẽ dùng r_năm tương ứng của năm đó.
    - annual_rates: dict, ví dụ {2025: 0.17, 2026: 0.15, "default": 0.15}
    """
    if start_date is None:
        start_date = pd.Timestamp.today().normalize()

    months = pd.date_range(start=start_date, periods=12, freq="MS")

    prices = []
    monthly_returns = []
    annual_used = []

    p = float(current_price_ty)

    default_r = float(annual_rates.get("default", 0.0))
    for m in months:
        r_year = float(annual_rates.get(int(m.year), default_r))
        r_month = annual_to_monthly(r_year)

        p = p * (1.0 + r_month)

        prices.append(max(p, 0.0))
        monthly_returns.append(r_month)
        annual_used.append(r_year)

    out = pd.DataFrame(
        {
            "Thang": months,
            "Gia_du_bao_ty": prices,
            "Ty_suat_thang": monthly_returns,
            "Ty_suat_nam": annual_used,
        }
    )
    return out


def get_district_block(schema: dict, district: str) -> Optional[dict]:
    return schema.get("per_district", {}).get(district)


def ui_numeric(
    col: str,
    label: str,
    schema: dict,
    district_block: Optional[dict],
    use_p05_p95: bool,
    key: str,
):
    # pick stats
    stt = None
    if district_block and "numeric" in district_block and col in district_block["numeric"]:
        stt = district_block["numeric"][col]
    else:
        stt = schema.get("numeric", {}).get(col, {})

    if not stt:
        st.warning(f"Thiếu thống kê numeric cho cột: {col}")
        return 0.0

    is_int = bool(stt.get("is_int", False))
    step = stt.get("step", 1 if is_int else 0.1)

    if use_p05_p95:
        minv = stt.get("p05", stt.get("min", 0))
        maxv = stt.get("p95", stt.get("max", 0))
        range_note = "P05–P95"
    else:
        minv = stt.get("min", 0)
        maxv = stt.get("max", 0)
        range_note = "Min–Max"

    # safety
    if maxv < minv:
        minv, maxv = maxv, minv

    default = stt.get("median", (minv + maxv) / 2)
    default = float(default)
    default = max(float(minv), min(float(maxv), default))

    help_txt = f"Ràng buộc theo dữ liệu ({range_note}). Median={stt.get('median', '')} | Full min={stt.get('min', '')}, max={stt.get('max', '')}"

    # constant -> show text only
    if abs(float(maxv) - float(minv)) < 1e-12:
        if is_int:
            v = int(round(minv))
            st.write(f"**{label}:** {v} (cố định theo dữ liệu quận/huyện)")
            return v
        v = float(minv)
        st.write(f"**{label}:** {v:.3f} (cố định theo dữ liệu quận/huyện)")
        return v

    if is_int:
        v = st.number_input(
            label,
            min_value=int(round(minv)),
            max_value=int(round(maxv)),
            value=int(round(default)),
            step=1,
            key=key,
            help=help_txt,
        )
        return int(v)

    v = st.number_input(
        label,
        min_value=float(minv),
        max_value=float(maxv),
        value=float(default),
        step=float(step),
        key=key,
        help=help_txt,
    )
    return float(v)


def ui_categorical(
    col: str,
    label: str,
    schema: dict,
    district_block: Optional[dict],
    key: str,
):
    opts = []
    if district_block and "categorical" in district_block:
        opts = district_block["categorical"].get(col, [])
    if not opts:
        opts = schema.get("categorical", {}).get(col, [])

    if not opts:
        st.write(f"**{label}:** (không có dữ liệu)")
        return ""

    if len(opts) == 1:
        st.write(f"**{label}:** {opts[0]} (cố định theo dữ liệu quận/huyện)")
        return opts[0]

    return st.selectbox(label, opts, index=0, key=key)


def ui_binary(
    col: str,
    label: str,
    schema: dict,
    district_block: Optional[dict],
    key: str,
):
    allowed = None
    if district_block and "binary" in district_block and col in district_block["binary"]:
        allowed = district_block["binary"][col]
    else:
        allowed = schema.get("binary", {}).get(col, [0, 1])

    allowed = sorted(list(set([int(x) for x in allowed])))

    # Remove option if district never has it
    if allowed == [0]:
        st.write(f"**{label}:** Không (quận/huyện này không có lựa chọn 'Có' trong dữ liệu)")
        return 0
    if allowed == [1]:
        st.write(f"**{label}:** Có (quận/huyện này luôn là 'Có' trong dữ liệu)")
        return 1

    val = st.checkbox(label, value=False, key=key)
    return int(bool(val))


# =========================
#  App
# =========================
st.set_page_config(page_title="Hanoi House Price Forecast", page_icon="🏠", layout="wide")
inject_css()

schema = load_schema()

st.title("🏠 Dự đoán giá nhà Hà Nội (30 quận/huyện)")
st.caption("UI tự ràng buộc theo quận/huyện dựa trên dữ liệu thô sau làm sạch (lọc theo lựa chọn có thật & range theo từng quận/huyện).")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Thiết lập")
    use_p05_p95 = st.checkbox("Ràng buộc numeric theo P05–P95 (khuyến nghị)", value=True)
    st.divider()
    st.subheader("📈 Dự báo 12 tháng")
    st.latex(r"r_{tháng} = (1 + r_{năm})^{1/12} - 1")
    r_2025 = st.number_input("Tăng trưởng năm 2025 (%/năm)", value=17.0, step=0.1) / 100.0
    r_2026 = st.number_input("Tăng trưởng năm 2026 (%/năm)", value=15.0, step=0.1) / 100.0
    annual_rates = {2025: float(r_2025), 2026: float(r_2026), "default": float(r_2026)}
    st.caption("Mỗi tháng sẽ dùng r_năm của đúng năm đó (nếu chuỗi dự báo vượt qua ranh giới năm).")
    st.divider()

    # =========================
    #  Model selection (nếu có nhiều model)
    # =========================
    model_paths = []
    if DEFAULT_MODEL_PATH.exists():
        model_paths.append(DEFAULT_MODEL_PATH)
    if MODELS_DIR.exists():
        model_paths.extend(sorted(MODELS_DIR.glob("*.pkl")))

    if not model_paths:
        st.error("⚠️ Chưa có model. Hãy chạy: python model_training.py")
        st.stop()

    labels = []
    for p in model_paths:
        if p.name == DEFAULT_MODEL_PATH.name:
            labels.append("BEST (best_model.pkl)")
        else:
            labels.append(p.stem)

    chosen_label = st.selectbox("🧠 Chọn mô hình dự đoán", labels, index=0)
    chosen_path = model_paths[labels.index(chosen_label)]
    st.caption(f"Đang dùng model: **{chosen_path.name}**")

    # Hiện best model theo lần train gần nhất (nếu có)
    if MODEL_INFO_PATH.exists():
        try:
            info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
            bm = info.get("best_model")
            if bm:
                st.caption(f"Best (từ train): **{bm}**")
        except Exception:
            pass

    st.markdown("**Ghi chú:** Nếu quận/huyện không có một lựa chọn (vd: *gần bãi rác*), UI sẽ tự ẩn/khóa lựa chọn đó dựa theo dữ liệu.")

# Load chosen model
model = load_model(str(chosen_path))

# Show model class (runtime)
try:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        st.caption(f"Mô hình đang chạy: **{type(model.named_steps['model']).__name__}**")
except Exception:
    pass

districts = schema.get("categorical", {}).get("Quan_Huyen", [])
if not districts:
    st.error("Schema không có danh sách Quan_Huyen. Hãy chạy lại preprocessing.py để tạo schema.")
    st.stop()

# District selection first
quan = st.selectbox("Chọn Quận/Huyện", districts, index=0)
district_block = get_district_block(schema, quan)
n_samples = district_block.get("n") if district_block else None
if n_samples is not None:
    st.caption(f"Dữ liệu quận/huyện **{quan}**: **{n_samples:,}** mẫu (tập full sau làm sạch).")

# Optional: show constraints summary
with st.expander("Xem ràng buộc theo dữ liệu của quận/huyện đã chọn"):
    if not district_block:
        st.write("Không tìm thấy thống kê theo quận/huyện trong schema.")
    else:
        # Numeric summary table
        rows = []
        for c, stt in district_block.get("numeric", {}).items():
            rows.append(
                {
                    "Thuộc tính": c,
                    "P05": stt.get("p05"),
                    "Median": stt.get("median"),
                    "P95": stt.get("p95"),
                    "Min": stt.get("min"),
                    "Max": stt.get("max"),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows))

# =========================
#  Inputs
# =========================
c1, c2, c3 = st.columns([1.05, 1.0, 1.0], gap="large")

with c1:
    st.subheader("📍 Vị trí & khu vực")
    khoang_cach = ui_numeric(
        "Khoang_cach_TT_km",
        "Khoảng cách tới trung tâm (km)",
        schema,
        district_block,
        use_p05_p95=use_p05_p95,
        key=f"Khoang_cach_TT_km_{quan}",
    )
    dac_diem = ui_categorical(
        "Dac_diem_khu_vuc",
        "Đặc điểm khu vực",
        schema,
        district_block,
        key=f"Dac_diem_khu_vuc_{quan}",
    )
    loai_dat = ui_categorical(
        "Loai_dat",
        "Loại đất",
        schema,
        district_block,
        key=f"Loai_dat_{quan}",
    )
    mat_do = ui_categorical(
        "Mat_do_dan_cu",
        "Mật độ dân cư",
        schema,
        district_block,
        key=f"Mat_do_dan_cu_{quan}",
    )

with c2:
    st.subheader("🏗️ Quy mô & pháp lý")
    dien_tich = ui_numeric(
        "Dien_tich_m2",
        "Diện tích (m²)",
        schema,
        district_block,
        use_p05_p95=use_p05_p95,
        key=f"Dien_tich_m2_{quan}",
    )
    mat_tien = ui_numeric(
        "Mat_tien_m",
        "Mặt tiền (m)",
        schema,
        district_block,
        use_p05_p95=use_p05_p95,
        key=f"Mat_tien_m_{quan}",
    )
    so_tang = ui_numeric(
        "So_tang",
        "Số tầng",
        schema,
        district_block,
        use_p05_p95=use_p05_p95,
        key=f"So_tang_{quan}",
    )
    so_phong_ngu = ui_numeric(
        "So_phong_ngu",
        "Số phòng ngủ",
        schema,
        district_block,
        use_p05_p95=use_p05_p95,
        key=f"So_phong_ngu_{quan}",
    )
    so_phong_tam = ui_numeric(
        "So_phong_tam",
        "Số phòng tắm",
        schema,
        district_block,
        use_p05_p95=use_p05_p95,
        key=f"So_phong_tam_{quan}",
    )
    tuoi_nha = ui_numeric(
        "Tuoi_nha_nam",
        "Tuổi nhà (năm)",
        schema,
        district_block,
        use_p05_p95=use_p05_p95,
        key=f"Tuoi_nha_nam_{quan}",
    )
    phap_ly = ui_categorical(
        "Phap_ly",
        "Pháp lý",
        schema,
        district_block,
        key=f"Phap_ly_{quan}",
    )
    xuong_cap = ui_categorical(
        "Muc_do_xuong_cap",
        "Mức độ xuống cấp",
        schema,
        district_block,
        key=f"Muc_do_xuong_cap_{quan}",
    )

with c3:
    st.subheader("🛣️ Đường xá & tiện ích")
    loai_duong = ui_categorical(
        "Loai_duong",
        "Loại đường",
        schema,
        district_block,
        key=f"Loai_duong_{quan}",
    )
    do_rong_duong = ui_numeric(
        "Do_rong_duong_m",
        "Độ rộng đường (m)",
        schema,
        district_block,
        use_p05_p95=use_p05_p95,
        key=f"Do_rong_duong_m_{quan}",
    )

    o_to = ui_binary("O_to_vao", "Ô tô vào được", schema, district_block, key=f"O_to_vao_{quan}")
    gara = ui_binary("Co_Gara", "Có gara", schema, district_block, key=f"Co_Gara_{quan}")
    san_thuong = ui_binary("Co_San_thuong", "Có sân thượng", schema, district_block, key=f"Co_San_thuong_{quan}")

    huong = ui_categorical("Huong_nha", "Hướng nhà", schema, district_block, key=f"Huong_nha_{quan}")
    an_ninh = ui_categorical("An_ninh", "An ninh", schema, district_block, key=f"An_ninh_{quan}")

    nghia_trang = ui_binary(
        "Gan_nghia_trang_bai_rac",
        "Gần nghĩa trang/bãi rác",
        schema,
        district_block,
        key=f"Gan_nghia_trang_bai_rac_{quan}",
    )
    ngap = ui_binary(
        "Co_bi_ngap",
        "Có bị ngập",
        schema,
        district_block,
        key=f"Co_bi_ngap_{quan}",
    )

    gan_tien_ich = ui_categorical(
        "Gan_Tien_ich",
        "Gần tiện ích",
        schema,
        district_block,
        key=f"Gan_Tien_ich_{quan}",
    )
    gan_giao_thong = ui_categorical(
        "Gan_Giao_thong",
        "Gần giao thông công cộng",
        schema,
        district_block,
        key=f"Gan_Giao_thong_{quan}",
    )
    noi_that = ui_categorical("Noi_that", "Nội thất", schema, district_block, key=f"Noi_that_{quan}")
    dien_nuoc = ui_categorical(
        "Tinh_trang_Dien_Nuoc",
        "Tình trạng điện/nước",
        schema,
        district_block,
        key=f"Tinh_trang_Dien_Nuoc_{quan}",
    )

st.divider()

# Predict
left, right = st.columns([1, 1])
with left:
    do_predict = st.button("🚀 Dự đoán giá")

if do_predict:
    record = {
        "Quan_Huyen": quan,
        "Khoang_cach_TT_km": khoang_cach,
        "Dac_diem_khu_vuc": dac_diem,
        "Loai_dat": loai_dat,
        "Dien_tich_m2": dien_tich,
        "Mat_tien_m": mat_tien,
        "So_tang": so_tang,
        "So_phong_ngu": so_phong_ngu,
        "So_phong_tam": so_phong_tam,
        "Do_rong_duong_m": do_rong_duong,
        "Loai_duong": loai_duong,
        "O_to_vao": int(o_to),
        "Co_Gara": int(gara),
        "Co_San_thuong": int(san_thuong),
        "Huong_nha": huong,
        "Phap_ly": phap_ly,
        "Mat_do_dan_cu": mat_do,
        "An_ninh": an_ninh,
        "Gan_nghia_trang_bai_rac": int(nghia_trang),
        "Co_bi_ngap": int(ngap),
        "Gan_Tien_ich": gan_tien_ich,
        "Gan_Giao_thong": gan_giao_thong,
        "Noi_that": noi_that,
        "Tinh_trang_Dien_Nuoc": dien_nuoc,
        "Muc_do_xuong_cap": xuong_cap,
        "Tuoi_nha_nam": tuoi_nha,
    }

    X_in = pd.DataFrame([record])

    # Model predicts log-price (trained on log1p), so reverse with expm1
    pred_log = float(model.predict(X_in)[0])
    pred_ty = float(np.expm1(pred_log))
    pred_ty = max(pred_ty, 0.0)

    # Output metrics
    m1, m2, m3 = st.columns([1.0, 1.0, 1.0])
    with m1:
        st.metric("Giá dự đoán", fmt_ty(pred_ty))
    with m2:
        st.metric("Quy đổi VND", fmt_vnd(pred_ty))
    with m3:
        unit_trieu = (pred_ty * 1000) / float(dien_tich) if float(dien_tich) > 0 else np.nan
        st.metric("Giá / m² (ước tính)", f"{unit_trieu:,.2f} triệu/m²" if np.isfinite(unit_trieu) else "-")

    st.markdown("""<div style="height:10px"></div>""", unsafe_allow_html=True)

    st.subheader("📈 Dự báo giá trong 12 tháng tới (lãi kép)")
    st.caption(f"Giả định: 2025 = {r_2025*100:.1f}%/năm, 2026 = {r_2026*100:.1f}%/năm")
    st.latex(r"r_{tháng} = (1 + r_{năm})^{1/12} - 1")
    fc = generate_forecast_12m(pred_ty, annual_rates=annual_rates)

    fig = plt.figure()
    plt.plot(fc["Thang"], fc["Gia_du_bao_ty"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Giá (tỷ VNĐ)")
    plt.xlabel("Tháng")
    plt.title("Dự báo giá 12 tháng tới (lãi kép theo r_năm)")
    st.pyplot(fig, clear_figure=True)

    tbl = fc.copy()
    tbl["Thang"] = tbl["Thang"].dt.strftime("%Y-%m")
    tbl["Ty_suat_nam"] = tbl["Ty_suat_nam"] * 100
    tbl["Ty_suat_thang"] = tbl["Ty_suat_thang"] * 100
    tbl = tbl[["Thang", "Gia_du_bao_ty", "Ty_suat_nam", "Ty_suat_thang"]]
    st.dataframe(
        tbl.rename(
            columns={
                "Thang": "Tháng",
                "Gia_du_bao_ty": "Giá dự báo (tỷ)",
                "Ty_suat_nam": "Tăng trưởng năm dùng (%)",
                "Ty_suat_thang": "Tăng trưởng tháng (%)",
            }
        )
    )

    with st.expander("Xem input đã gửi vào mô hình"):
        st.json(record)
