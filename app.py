import streamlit as st
from collections import Counter

# Cấu hình trang
st.set_page_config(page_title="TITAN v30.1 - XIÊN 2 PRO", layout="wide")

def analyze_logic(data_input):
    # Lọc lấy các dòng có đúng 5 chữ số
    history = [str(line).strip() for line in data_input if len(str(line).strip()) == 5]
    if len(history) < 5:
        return None

    # Tách dữ liệu 2 hàng mục tiêu: Hàng Chục (vị trí -2) và Hàng Đơn Vị (vị trí -1)
    h_chuc = [int(line[-2]) for line in history]
    h_donvi = [int(line[-1]) for line in history]

    def get_binary_prediction(digits):
        # Đếm 5 kỳ gần nhất
        last_5 = digits[:5]
        tai_count = sum(1 for d in last_5 if d >= 5)
        # Nếu đang bệt Tài (4/5 kỳ), dự đoán bẻ sang Xỉu hoặc ngược lại
        if tai_count >= 4: return "XỈU"
        if tai_count <= 1: return "TÀI"
        # Nếu cầu nhảy, đánh theo số vừa về (bám bệt)
        return "TÀI" if digits[0] >= 5 else "XỈU"

    res_chuc = get_binary_prediction(h_chuc)
    res_donvi = get_binary_prediction(h_donvi)
    
    return res_chuc, res_donvi

# --- GIAO DIỆN ---
st.title("🎯 TITAN v30.1 - KHAI THÁC XIÊN 2")
st.markdown("---")

raw_data = st.text_area("📥 Dán kết quả 5D (Ví dụ: 80673):", height=150, placeholder="Dán dãy số vào đây...")

if raw_data:
    lines = raw_data.split('\n')
    results = analyze_logic(lines)
    
    if results:
        trend_c, trend_dv = results
        
        # Hiển thị khu vực XIÊN 2
        st.subheader("🔥 GỢI Ý XIÊN 2 (Vốn ít - Ăn đậm)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📍 HÀNG CHỤC: **{trend_c}**")
        with col2:
            st.info(f"📍 ĐƠN VỊ: **{trend_dv}**")
            
        st.warning(f"🚀 CƯỢC XIÊN gợi ý: **H.Chục {trend_c} & Đơn vị {trend_dv}**")
        st.caption("Tỷ lệ lợi nhuận: ~3.9 lần vốn. Chỉ cần thắng 1 kỳ gỡ lại 3 kỳ thua.")

        # Quản lý vốn Xiên 2
        with st.expander("💰 Công thức vào tiền Xiên 2"):
            st.write("""
            | Kỳ | Mức cược | Tổng vốn | Nếu thắng nhận | Lợi nhuận |
            | :--- | :--- | :--- | :--- | :--- |
            | 1 | 10k | 10k | 39k | +29k |
            | 2 | 10k | 20k | 39k | +19k |
            | 3 | 20k | 40k | 78k | +38k |
            """)
    else:
        st.error("Cần tối thiểu 5 dòng dữ liệu để phân tích!")

st.markdown("---")
st.write("🛠 **Mẹo:** Nếu anh thấy 2 hàng báo cùng 1 loại (ví dụ cùng Tài), xác suất nổ Xiên 2 cực cao.")
