# Dự đoán Giá Nhà Hà Nội 🏠 (Full thuộc tính + Ràng buộc theo Quận/Huyện)

Project này đã được **làm lại từ đầu** để phù hợp với bộ dữ liệu thô bạn gửi (~50,000 dòng).

Điểm khác biệt so với bản cũ:
- Dataset mới có **đầy đủ 20 thuộc tính đầu vào** (khoảng cách, hạ tầng, môi trường, nội thất...)
- Có **3 file dữ liệu đầu ra**: *Cleaned* → *Encoded* → *Processed (train-ready)*
- `app.py` đã được sửa để:
  - hiển thị **đầy đủ toàn bộ thuộc tính** dùng để train
  - áp dụng **ràng buộc theo quận/huyện/thị xã** dựa trên file docx (`urban_constraints.docx`)
  - hỗ trợ chọn **đủ 30 quận/huyện/thị xã Hà Nội** (12 quận + 17 huyện + 1 thị xã)

---

## 1) Bộ dữ liệu & các file đầu ra

### ✅ Input (raw)
- `HN_Houseprice_Raw.csv` (đã copy từ file bạn gửi)
- `HN_Houseprice.csv` (được đồng bộ giống raw để tránh nhầm)

### ✅ Output tạo tự động
Chạy `python preprocessing.py` sẽ sinh:
- `HN_Houseprice_Cleaned.csv`: dữ liệu đã làm sạch + có thêm cột `Gia_trieu_m2` (chỉ để EDA)
- `HN_Houseprice_Encoded.csv`: dữ liệu đã one-hot encode các biến phân loại
- `HN_Houseprice_Processed.csv`: dữ liệu cuối để train (Encoded + `Gia_ban_ty_log`)
- `feature_schema.json`: schema cho app (danh sách category + min/max/median)

---

## 2) Các thuộc tính đầu vào (20 features)

| Nhóm | Thuộc tính |
|---|---|
| Vị trí | `Quan`, `Khoang_cach_TT_km` |
| Kích thước | `Dien_tich_m2`, `Mat_tien_m`, `Do_rong_duong_m` |
| Cấu trúc | `So_tang`, `So_phong_ngu`, `So_phong_tam`, `Tuoi_nha_nam` |
| Hạ tầng | `Loai_duong`, `O_to_vao`, `Gan_Metro_Bus` |
| Tiện ích | `Co_Gara`, `Co_San_thuong` |
| Môi trường | `Mat_do_dan_cu`, `An_ninh`, `Tinh_trang_ngap`, `Gan_nghia_trang_bai_rac` |
| Nội thất/tiện nghi | `Noi_that`, `Tinh_trang_Dien_Nuoc` |

Target:
- `Gia_ban_ty` (tỷ VNĐ)

---

## 3) Cách chạy pipeline

### Bước 1 — Tiền xử lý (raw → cleaned → encoded → processed)
```bash
python preprocessing.py
```

Mặc định `preprocessing.py` sẽ:
- lọc outlier theo quantile (hai phía)
- và giới hạn số dòng sau làm sạch về khoảng **~15,000** (stratified theo Quận) để train nhanh.

Bạn có thể đổi tham số:
```bash
python preprocessing.py --max_rows 0 --outlier_q 0.05
```

### Bước 2 — Train model
Mặc định model train trên `log(1 + Gia_ban_ty)`.

Script sẽ so sánh nhiều thuật toán (Linear/Ridge, RandomForest, ExtraTrees,
GradientBoosting, HistGradientBoosting, KNN, và XGBoost nếu có).

Nếu bạn muốn dùng XGBoost, cài thêm:
```bash
pip install xgboost
```

Train (mặc định):
```bash
python model_training.py
```

Tuỳ chọn: bật tuning nhanh để giảm sai số hơn nữa:
```bash
python model_training.py --tune
```

Sau bước này sẽ có:
- `best_model.pkl`
- `model_features.pkl`
- `model_comparison.csv`

### Bước 3 — Chạy giao diện
```bash
streamlit run app.py
```

### (Tuỳ chọn) Tuning để giảm sai số
```bash
python model_training.py --tune
```

### (Tuỳ chọn) Vẽ đồ thị dự báo 12 tháng
```bash
python forecast_12m.py
```

---

## 4) Ràng buộc theo Quận/Huyện/Thị xã

- Tài liệu gốc: `urban_constraints.docx`
- `app.py` sẽ tự động:
  - chặn các giá trị ❌ (không cho chọn)
  - ép một số trường hợp đặc biệt (ví dụ: Sơn Tây ép `Gan_Metro_Bus=0`, `Mat_do_dan_cu=Trung bình`)
  - clamp một số range số (ví dụ: quận lõi trung tâm giới hạn `Khoang_cach_TT_km`)

Bạn có thể xem phần **“📌 Xem ràng buộc đang áp dụng”** ngay trong app.

---

## 5) Ghi chú
- Nếu chọn quận/huyện **không có trong dữ liệu train**, app sẽ cảnh báo vì mô hình chưa học được pattern của khu vực đó.
- Giá dự đoán chỉ mang tính tham khảo; thực tế phụ thuộc pháp lý, quy hoạch, vị trí ngõ, thời điểm thị trường...

