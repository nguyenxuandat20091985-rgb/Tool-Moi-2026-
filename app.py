import streamlit as st
import collections

# Cấu hình giao diện
st.set_page_config(page_title="TOOL LOTO 2026", layout="wide")
st.title("🎰 TOOL LOTO PHIÊN BẢN MỚI 2026")
st.write("---")

# Ô nhập liệu
txt = st.text_area("👇 Nhập kết quả (5 số mỗi kỳ, mỗi dòng 1 kỳ):", "12345\n67890\n55555")

if st.button("🚀 BẮT ĐẦU SOI CẦU"):
    lines = [l.strip() for l in txt.split('\n') if len(l.strip()) == 5]
    if lines:
        st.subheader("🎯 Kết quả dự đoán Nhất Tinh:")
        cols = st.columns(5)
        titles = ["Vạn", "Nghìn", "Trăm", "Chục", "Đơn vị"]
        
        for i in range(5):
            digits = [line[i] for line in lines]
            # Lấy số về nhiều nhất
            num, count = collections.Counter(digits).most_common(1)[0]
            percent = int((count / len(lines)) * 100)
            
            with cols[i]:
                st.metric(label=titles[i], value=f"SỐ {num}", delta=f"{percent}%")
        st.success("💡 Mẹo: Hàng nào có % càng cao thì cầu càng chắc anh nhé!")
    else:
        st.error("Anh nhập ít nhất 1 dòng có 5 số nhé (Ví dụ: 12345)")
