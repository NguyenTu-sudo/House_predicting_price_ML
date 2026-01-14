"""forecast_12m.py

Tạo đồ thị dự báo giá trung vị (median) trong 12 tháng tới dựa trên
tốc độ tăng giá dự kiến theo báo cáo thị trường.

Ý tưởng (đơn giản, dễ thuyết trình):
  - Lấy giá hiện tại = median(Gia_ban_ty) trong dữ liệu cleaned
  - Chiết xuất kịch bản tăng giá năm 2026 từ CBRE (Hanoi):
      * Nhà đất (landed) secondary: ~3%/năm
      * Chung cư (condo) secondary: ~6%/năm
  - Quy đổi sang tăng trưởng theo tháng và vẽ 2 đường.

Chạy:
  python forecast_12m.py --input HN_Houseprice_Cleaned.csv

Output:
  - forecast_12m.csv
  - forecast_12m.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def comp_monthly(annual: float) -> float:
    return (1.0 + annual) ** (1.0 / 12.0) - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="HN_Houseprice_Cleaned.csv")
    parser.add_argument("--out_csv", default="forecast_12m.csv")
    parser.add_argument("--out_png", default="forecast_12m.png")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {in_path.resolve()}")

    df = pd.read_csv(in_path)
    if "Gia_ban_ty" not in df.columns:
        raise ValueError("Thiếu cột Gia_ban_ty trong dữ liệu cleaned")

    base = float(df["Gia_ban_ty"].median())

    annual_landed = 0.03
    annual_condo = 0.06
    m_landed = comp_monthly(annual_landed)
    m_condo = comp_monthly(annual_condo)

    start = pd.Timestamp.today().normalize().replace(day=1)
    idx = pd.date_range(start=start, periods=13, freq="MS")

    landed = [base * ((1 + m_landed) ** i) for i in range(13)]
    condo = [base * ((1 + m_condo) ** i) for i in range(13)]

    out = pd.DataFrame(
        {
            "Thang": idx,
            "Gia_trung_vi_ty": [base] * len(idx),
            "Du_bao_nha_dat_ty": landed,
            "Du_bao_chung_cu_ty": condo,
        }
    )
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.plot(idx, landed, label="Nhà đất (CBRE secondary ~3%/năm)")
    plt.plot(idx, condo, label="Chung cư (CBRE secondary ~6%/năm)")
    plt.title("Dự báo 12 tháng (tham khảo) từ mức giá trung vị hiện tại")
    plt.xlabel("Tháng")
    plt.ylabel("Giá (tỷ VNĐ)")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)

    print(f"✅ Đã lưu: {args.out_csv}")
    print(f"✅ Đã lưu: {args.out_png}")


if __name__ == "__main__":
    main()
