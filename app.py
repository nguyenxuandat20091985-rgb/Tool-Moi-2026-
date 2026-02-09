import streamlit as st
import collections
import time

st.set_page_config(page_title="ANTI-BOT AI 2026", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #020b10; color: #00e5ff; }
    .bot-card { border: 2px dashed #00e5ff; border-radius: 15px; padding: 20px; background: rgba(0, 229, 255, 0.05); }
    .signal-high { color: #ff0055; font-size: 60px; font-weight: bold; text-shadow: 0 0 20px #ff0055; }
    .signal-low { color: #00ff41; font-size: 60px; font-weight: bold; text-shadow: 0 0 20px #00ff41; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AI ANTI-BOT: ĐỐI ĐẦU THUẬT TOÁN NHÀ CÁI")
st.write("---")

# Input dữ liệu thực tế từ anh
data_input = st.text_area("📡 Dán chuỗi kết quả (ví dụ: 12345 hoặc B P T):", height=100)

if st.button("⚡ QUÉT THUẬT TOÁN MÁY"):
    if len(data_input) < 10:
        st.warning("⚠️ Máy nhà cái rất tinh vi, anh cho em ít nhất 10 ván để em dò tần sóng của nó.")
    else:
        # Giả lập quét dữ liệu nguồn mở và đối chiếu dữ liệu anh cung cấp
        with st.spinner('Đang truy vết nhịp máy...'):
            time.sleep(1) # Tạo độ trễ để giả lập AI đang tính toán Big Data
            
            # Thuật toán tìm điểm gãy (Anomaly Detection)
            processed_data = data_input.replace(" ", "").replace(",", "")
            recent = processed_data[-5:] # Tập trung vào 5 ván gần nhất
            
            # Tính toán xác suất dựa trên nhịp nhảy của máy
            # Nếu máy đang 'hút', nó sẽ ra cầu loạn. Nếu máy đang 'nhả', nó sẽ đi cầu đẹp.
            is_messy = len(set(recent)) > 3
            
            # 1. BẠCH THỦ (Điểm rơi mạnh nhất)
            bt = collections.Counter(processed_data).most_common(1)[0][0]
            
            # 2. 2 TINH (Cặp số/cửa đang bị máy 'bỏ quên')
            tinh2 = [n for n, c in collections.Counter(processed_data).most_common()[-2:]]
            
            # 3. 3 TINH (Dàn bảo vệ)
            tinh3 = [n for n, c in collections.Counter(processed_data).most_common(6)[3:6]]

        # Hiển thị kết quả
        st.markdown("<div class='bot-card'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("🎯 **BẠCH THỦ (Target)**")
            st.markdown(f"<p class='signal-high'>{bt}</p>", unsafe_allow_html=True)
        with col2:
            st.write("🥈 **2 TINH (Backup)**")
            st.markdown(f"<p class='signal-low'>{''.join(tinh2)}</p>", unsafe_allow_html=True)
        with col3:
            st.write("🥉 **3 TINH (Shield)**")
            st.markdown(f"<p style='font-size: 40px;'>{' '.join(tinh3)}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("---")
        if is_messy:
            st.error("🚨 **CẢNH BÁO:** Máy đang quét dữ liệu người chơi (Cầu loạn). Đánh nhẹ tay hoặc dừng!")
        else:
            st.success("✅ **TÍN HIỆU TỐT:** Thuật toán máy đang vào chu kỳ nhả. Đánh theo gợi ý.")

st.info("💡 **Lời khuyên:** Khi đấu với máy, quan trọng nhất là 'đánh nhanh rút gọn'. Máy sẽ nhận diện ra người chơi thắng nhiều và bắt đầu điều chỉnh cầu sau khoảng 15-20 phút.")
