# Dự đoán Giá Nhà Hà Nội 🏠 (30 quận/huyện) – Bản đã chỉnh theo yêu cầu

Bản này đã được chỉnh lại để:

- ✅ **Bỏ thuộc tính `Nhom_Khu_vuc`** ("Nhóm khu vực") khỏi dữ liệu *và* giao diện
- ✅ **Không còn ràng buộc theo quận/huyện** trong UI (không còn clamp/ép giá trị theo quận)
- ✅ Sửa cách xử lý **biến phân loại** bằng sklearn **Pipeline + OneHotEncoder**  
  → đổi các thuộc tính phân loại sẽ **làm giá dự đoán thay đổi** (không còn bị “đứng giá” do lệch schema)
- ✅ Thêm **sơ đồ dự báo 12 tháng** (mô phỏng theo kịch bản 2026, có “biến động bất thường”)

---

## 1) Dữ liệu & các file đầu ra

Nguồn dữ liệu đầu vào: `HaNoi_Housing_Ultimate_Full.csv` (≈ 50.000 dòng)

Sau khi chạy `preprocessing.py`, repo sẽ có:

- `HN_Houseprice.csv` : dữ liệu gốc (đã loại `Nhom_Khu_vuc`)
- `HN_Houseprice_Cleaned.csv` : dữ liệu làm sạch (≤ 20.000 dòng theo yêu cầu)
- `HN_Houseprice_Encoded.csv` : dữ liệu one-hot để EDA
- `HN_Houseprice_Processed.csv` : thêm cột `Gia_ban_ty_log` để hỗ trợ phân tích
- `feature_schema.json` : schema cho Streamlit UI (danh sách category + min/max/median)
- `cleaning_report.json` : log tóm tắt làm sạch

> Lưu ý: Dữ liệu làm sạch được lấy mẫu **có stratify theo `Quan_Huyen`** để đảm bảo đủ **30 quận/huyện**.

---

## 2) Train model

Model được train theo log-target (`log1p(Gia_ban_ty)`) bằng sklearn Pipeline:

- Numeric + binary: `StandardScaler`
- Categorical: `OneHotEncoder(handle_unknown='ignore')`
- Model: `Ridge`

Chạy:

```bash
python model_training.py --data HN_Houseprice_Cleaned.csv
```

Kết quả:
- `best_model.pkl` (pipeline model)
- `model_features.pkl` (tùy chọn, để debug)

---

## 3) Chạy app Streamlit

```bash
streamlit run app.py
```

Trong app:
- Nhập thuộc tính
- Bấm **Dự đoán giá**
- App sẽ hiển thị:
  - Giá dự đoán (tỷ VNĐ)
  - Quy đổi VND
  - Giá/m² ước tính
  - **Biểu đồ dự báo 12 tháng** theo kịch bản (Cơ sở / Thận trọng / Tăng nhanh)

---

## 4) Dự báo 12 tháng (file mẫu)

Có sẵn script để sinh 1 biểu đồ mẫu:

```bash
python forecast_12m.py
```

Tạo ra:
- `forecast_12m.csv`
- `forecast_12m.png`

---

## 5) Ghi chú quan trọng

- Kết quả chỉ mang tính tham khảo (mô phỏng), không phải khuyến nghị đầu tư.
- Biểu đồ 12 tháng là **kịch bản mô phỏng** (có shock dương/âm để phản ánh biến động thị trường).
