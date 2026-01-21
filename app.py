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
MODEL_PATH = BASE_DIR / "best_model.pkl"


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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại: {MODEL_PATH}. Hãy chạy: python model_training.py"
        )
    return joblib.load(MODEL_PATH)


def fmt_ty(x: float) -> str:
    return f"{x:,.3f} tỷ"


def fmt_vnd(x_ty: float) -> str:
    vnd = x_ty * 1_000_000_000
    return f"{vnd:,.0f} ₫"


def generate_forecast_12m(
    current_price_ty: float,
    scenario: str = "Cơ sở",
    seed: int = 42,
) -> pd.DataFrame:
    """Tạo chuỗi dự báo 12 tháng (mang tính mô phỏng).

    - Có 'biến đổi bất thường' (shock) để phản ánh độ biến động của thị trường 2026.
    - Đây là mô phỏng theo kịch bản, KHÔNG phải khuyến nghị đầu tư.
    """
    rng = np.random.default_rng(seed)

    if scenario == "Thận trọng":
        annual_growth = 0.02  # 2%/năm
        sigma = 0.010
    elif scenario == "Tăng nhanh":
        annual_growth = 0.10  # 10%/năm
        sigma = 0.015
    else:
        annual_growth = 0.05  # 5%/năm
        sigma = 0.012

    base_monthly = (1 + annual_growth) ** (1 / 12) - 1

    # noise ngẫu nhiên
    noise = rng.normal(0, sigma, size=12)

    # shock theo "sự kiện" (tháng index 0..11)
    shocks = {
        2: -0.025,  # Q1: siết tín dụng (giả lập)
        5: -0.015,  # giữa năm: chính sách/thuế (giả lập)
        8: +0.020,  # Q3: hạ tầng/TOD tạo nhịp tăng (giả lập)
    }

    monthly_returns = np.full(12, base_monthly) + noise
    for i, v in shocks.items():
        monthly_returns[i] += v

    prices = []
    p = float(current_price_ty)
    for r in monthly_returns:
        p = p * (1 + float(r))
        prices.append(max(p, 0.0))

    months = pd.date_range(start=pd.Timestamp.today().normalize(), periods=12, freq="MS")
    out = pd.DataFrame({"Thang": months, "Gia_du_bao_ty": prices, "Ty_suat_thang": monthly_returns})
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
model = load_model()

st.title("🏠 Dự đoán giá nhà Hà Nội (30 quận/huyện)")
st.caption("UI tự ràng buộc theo quận/huyện dựa trên dữ liệu thô sau làm sạch (lọc theo lựa chọn có thật & range theo từng quận/huyện).")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Thiết lập")
    use_p05_p95 = st.checkbox("Ràng buộc numeric theo P05–P95 (khuyến nghị)", value=True)
    st.divider()
    scenario = st.selectbox("Kịch bản 2026 (mô phỏng 12 tháng)", ["Cơ sở", "Thận trọng", "Tăng nhanh"], index=0)
    st.divider()
    st.markdown("**Ghi chú:** Nếu quận/huyện không có một lựa chọn (vd: *gần bãi rác*), UI sẽ tự ẩn/khóa lựa chọn đó dựa theo dữ liệu.")

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
        unit = pred_ty / float(dien_tich) if float(dien_tich) > 0 else np.nan
        st.metric("Giá / m² (ước tính)", f"{unit:,.3f} tỷ/m²" if np.isfinite(unit) else "-")

    st.markdown("""<div style="height:10px"></div>""", unsafe_allow_html=True)

    st.subheader("📈 Dự báo 12 tháng tới (mô phỏng theo kịch bản 2026)")
    fc = generate_forecast_12m(pred_ty, scenario=scenario, seed=42)

    fig = plt.figure()
    plt.plot(fc["Thang"], fc["Gia_du_bao_ty"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Giá (tỷ VNĐ)")
    plt.xlabel("Tháng")
    plt.title("Dự báo giá 12 tháng tới (mô phỏng)")
    st.pyplot(fig, clear_figure=True)

    st.dataframe(
        fc.assign(Thang=fc["Thang"].dt.strftime("%Y-%m")).rename(
            columns={"Thang": "Tháng", "Gia_du_bao_ty": "Giá dự báo (tỷ)", "Ty_suat_thang": "Tỷ suất tháng"}
        )
    )

    with st.expander("Xem input đã gửi vào mô hình"):
        st.json(record)