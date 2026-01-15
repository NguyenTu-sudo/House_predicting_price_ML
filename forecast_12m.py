"""forecast_12m.py

Sinh dự báo giá 12 tháng (mô phỏng) cho năm 2026 theo kịch bản có biến động bất thường.

Lưu ý: Đây là mô phỏng theo kịch bản (scenario simulation) – KHÔNG phải dự báo chính thức.

Chạy:
    python forecast_12m.py

Kết quả:
    - forecast_12m.csv
    - forecast_12m.png
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_forecast(current_price_ty: float, annual_growth: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_monthly = (1 + annual_growth) ** (1 / 12) - 1

    # biến động ngẫu nhiên + shock (bất thường)
    sigma = 0.012
    noise = rng.normal(0, sigma, size=12)

    shocks = {
        2: -0.025,  # Q1: siết tín dụng (giả lập)
        5: -0.015,  # giữa năm: chính sách/thuế/đầu cơ (giả lập)
        8: 0.020,   # Q3: hạ tầng / TOD / kỳ vọng (giả lập)
        10: -0.010, # cuối năm: hấp thụ chậm/điều chỉnh (giả lập)
    }

    monthly = base_monthly + noise
    for k, v in shocks.items():
        monthly[k] += v
    monthly = np.clip(monthly, -0.08, 0.08)

    start = pd.Timestamp.today().normalize().replace(day=1)
    months = pd.date_range(start=start + pd.offsets.MonthBegin(1), periods=12, freq="MS")

    prices = []
    p = float(current_price_ty)
    for i in range(12):
        p = p * (1 + float(monthly[i]))
        prices.append(p)

    return pd.DataFrame(
        {
            "Thang": [m.strftime("%Y-%m") for m in months],
            "Gia_du_bao_ty": prices,
            "Bien_dong_%": (monthly * 100).round(2),
        }
    )


def main():
    # Ví dụ: giá hiện tại 8 tỷ, tăng 5%/năm
    current_price = 8.0
    df = generate_forecast(current_price, annual_growth=0.05, seed=42)

    df.to_csv("forecast_12m.csv", index=False)

    plt.figure()
    plt.plot(df["Thang"], df["Gia_du_bao_ty"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Tháng")
    plt.ylabel("Giá dự báo (tỷ VNĐ)")
    plt.title("Dự báo 12 tháng (mô phỏng, có biến động bất thường)")
    plt.tight_layout()
    plt.savefig("forecast_12m.png", dpi=160)
    print("✅ Saved forecast_12m.csv and forecast_12m.png")


if __name__ == "__main__":
    main()
