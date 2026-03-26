# 🚀 TITAN V27 - NVIDIA AI Lottery Predictor

> Hệ thống dự đoán lô đề "3 số 5 tinh" tích hợp AI NVIDIA Llama-3.1 & Google Gemini, xây dựng với Streamlit cho deployment dễ dàng.

## 🎮 Luật chơi "3 số 5 tinh"

- Người chơi chọn **3 số từ 0-9** để cược
- Kết quả là **số 5 chữ số** (hàng: Chục ngàn → Ngàn → Trăm → Chục → Đơn vị)
- ✅ **THẮNG** nếu 3 số bạn chọn **xuất hiện** trong 5 chữ số kết quả (bất kỳ vị trí, không quan tâm thứ tự)
- ❌ **THUA** nếu thiếu ít nhất 1 số trong 3 số đã cược
- 🔄 Dù số xuất hiện bao nhiêu lần trong kết quả, vẫn chỉ tính 1 lần

### Ví dụ:
| Cược | Kết quả | Kết quả | Giải thích |
|------|---------|---------|------------|
| `1,2,6` | `12864` | 🎉 THẮNG | Có đủ 1, 2, 6 trong kết quả |
| `1,3,6` | `12662` | ❌ THUA | Thiếu số 3 |

---

## 🚀 Deployment

### 1. Chạy local
```bash
git clone https://github.com/yourusername/titan-v27.git
cd titan-v27
pip install -r requirements.txt
streamlit run app.py