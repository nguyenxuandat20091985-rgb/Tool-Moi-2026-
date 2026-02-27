import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# Cấu hình trang tối ưu cho Mobile
st.set_page_config(page_title="TITAN v28.0 - SPEED", layout="centered")

def analyze_logic(data_input):
    # Tách dữ liệu hàng đơn vị (số cuối cùng)
    digits = [int(str(line).strip()[-1]) for line in data_input if str(line).strip()]
    if not digits: return None
    
    # 1. Dự đoán Kèo Đôi (Tài/Xỉu) dựa trên xác suất 50/50
    last_digit = digits[0]
    tx_status = "TÀI (5-9)" if last_digit < 5 else "XỈU (0-4)" # Logic đánh đảo cầu
    
    # 2. Tạo Dàn 7 số "Tĩnh" dựa trên tần suất xuất hiện
    counts = Counter(digits)
    # Lấy 7 số xuất hiện nhiều nhất trong 50 kỳ gần nhất
    most_common = [str(num) for num, count in counts.most_common(7)]
    dan_7 = " ".join(sorted(most_common))
    
    return tx_status, dan_7

# --- GIAO DIỆN CHÍNH ---
st.title("🚀 TITAN v28.0 - 5D KU")
st.markdown("---")

# Ô nhập liệu siêu tốc
raw_data = st.text_area("📥 Dán 10-20 kết quả gần nhất (Ví dụ: 80673):", height=150)

if raw_data:
    lines = raw_data.split('\n')
    result = analyze_logic(lines)
    
    if result:
        tx, d7 = result
        
        # Hiển thị kết quả Kèo Đôi
        st.subheader("🎯 KÈO ĐÔI (Xác suất 50/50)")
        st.error(f"GỢI Ý: {tx}")
        st.caption("Chiến thuật: Đánh đều tay hoặc Fibonacci")
        
        st.markdown("---")
        
        # Hiển thị Dàn 7 số cho 1 hàng duy nhất
        st.subheader("🔢 DÀN 7 SỐ (Hàng Đơn Vị)")
        st.success(d7)
        st.info("💡 Cách chơi: Nhập dàn này vào 'Hàng đơn vị', chọn 'Kỳ liên tiếp: 5' để rảnh tay.")
        
        # Bảng quản lý vốn gợi ý
        with st.expander("💰 Quản lý vốn (Gợi ý)"):
            st.write("""
            | Kỳ | Vốn (10k/số) | Tổng cược | Lợi nhuận |
            | :--- | :--- | :--- | :--- |
            | 1 | 70 | 70 | +29 |
            | 2 (Gấp) | 140 | 210 | +38 |
            """)

st.markdown("---")
st.warning("⚠️ Cảnh báo: AI chỉ tính toán dựa trên xác suất. Anh nên test nhẹ tay để quen nhịp 1 phút trước.")
