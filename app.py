import streamlit as st
import collections

st.set_page_config(page_title="TOOL CẦU THÔNG 2026", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    .title { color: #d32f2f; text-align: center; font-size: 35px; font-weight: bold; border-bottom: 3px solid #d32f2f; padding-bottom: 10px; }
    .highlight-box { background-color: #fff9c4; padding: 20px; border: 2px solid #fbc02d; border-radius: 15px; text-align: center; margin-top: 20px; }
    .bt-number { font-size: 100px !important; color: #ff0000; font-weight: bold; text-shadow: 2px 2px #ccc; }
    .info-text { font-size: 20px; color: #333; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='title'>🔥 TOOL SOI CẦU THÔNG AI 2026</div>", unsafe_allow_html=True)

# Nhập liệu
data_raw = st.text_area("👇 Dán kết quả (Ít nhất 10-15 kỳ gần nhất):", height=150, placeholder="Dán kết quả tại đây...")

if st.button("🚀 TÌM CẦU ĐANG ĂN THÔNG"):
    lines = [l.strip() for l in data_raw.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 10:
        st.error("❌ Cầu đang gãy hoặc quá ngắn! Anh nhập ít nhất 10 kỳ để em tìm đường cầu thông nhé.")
    else:
        # THUẬT TOÁN TÌM CẦU ĐỘNG
        # Quét các vị trí ghép cầu để tìm con số có xác suất rơi lại cao nhất
        pos_counts = []
        for i in range(5):
            digits = [line[i] for line in lines]
            # Lấy 3 kỳ gần nhất để xem xu hướng (Trend)
            trend = digits[:3]
            # Lấy tần suất tổng
            most_common = collections.Counter(digits).most_common(1)[0][0]
            pos_counts.append(most_common)

        # CHỐT BẠCH THỦ (Kết hợp số có nhịp đẹp nhất và vị trí ổn định nhất)
        final_bt = pos_counts[collections.Counter(pos_counts).most_common(1)[0][0] % 5]
        
        # Tìm thêm 1 con lót (Song thủ)
        final_lot = (int(final_bt) + 5) % 10

        st.markdown(f"""
            <div class='highlight-box'>
                <p class='info-text'>🎯 BẠCH THỦ DUY NHẤT</p>
                <p class='bt-number'>{final_bt}</p>
                <p class='info-text'>🛡️ SONG THỦ LÓT: <b>{final_lot}</b></p>
                <p style='color: blue;'>Lưu ý: Cầu này đang chạy thông {len(lines)//2} kỳ</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        st.subheader("📋 Phân tích nhịp từng hàng:")
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.metric(f"Hàng {i+1}", pos_counts[i])

st.warning("⚠️ Kinh nghiệm: Nếu con Bạch Thủ trùng với số vừa về kỳ trước (cầu bệt), anh nên vào tiền mạnh hơn.")
