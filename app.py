import streamlit as st
import collections

st.set_page_config(page_title="HỆ THỐNG PHÂN TÍCH GAME v19.0", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #060d0d; color: #00ffcc; }
    .box-pro { border: 2px solid #ffcc00; border-radius: 15px; padding: 25px; background: #111; text-align: center; box-shadow: 0 0 20px #ffcc00; }
    .num-pro { font-size: 80px !important; color: #ffffff; font-weight: bold; text-shadow: 0 0 10px #00ffcc; }
    .status-on { color: #00ff00; font-weight: bold; animation: blinker 1s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ CHIẾN THUẬT PHÒNG THỦ & PHẢN CÔNG 2026")

# Input dữ liệu
data = st.text_area("📡 Dán kết quả ván chơi của anh vào đây:", height=150)

if st.button("🔍 PHÂN TÍCH NHỊP CẦU"):
    lines = [l.strip() for l in data.split('\n') if len(l.strip()) > 0]
    
    if len(lines) < 8:
        st.error("❌ Anh ơi, cho em xin ít nhất 8-10 kỳ để em 'đọc vị' thuật toán ván này!")
    else:
        # Thuật toán bắt nhịp nhảy
        last_nums = "".join(lines[:3]) # 3 ván gần nhất
        all_nums = "".join(lines)
        freq = collections.Counter(all_nums)
        
        # Sắp xếp số theo lực đẩy
        sorted_nums = [n for n, c in freq.most_common(10)]
        
        # 1. Bạch thủ (Số có nhịp rơi trùng khớp cao nhất)
        bt = sorted_nums[0]
        # 2. 2 Tinh (Cặp đôi đang có xu hướng đi cùng nhau)
        tinh2 = sorted_nums[1:3]
        # 3. 3 Tinh (Dàn số lót vùng an toàn)
        tinh3 = sorted_nums[3:6]

        # Hiển thị kết quả thực chiến
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='box-pro'><h3>🎯 BẠCH THỦ</h3><p class='num-pro'>{bt}</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='box-pro'><h3>💎 2 TINH</h3><p class='num-pro'>{''.join(tinh2)}</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='box-pro'><h3>⚔️ 3 TINH</h3><p class='num-pro'>{''.join(tinh3)}</p></div>", unsafe_allow_html=True)

        st.write("---")
        # Phân tích trạng thái bàn chơi
        if lines[0] == lines[1]:
            st.markdown("⚠️ **TRẠNG THÁI:** Bàn đang đi cầu Bệt cực nặng. Đánh bám cầu, không bẻ!")
        else:
            st.markdown("🔄 **TRẠNG THÁI:** Cầu đang nhảy nhịp 1-1 hoặc Đảo. Tool đã cập nhật số theo nhịp nhảy.")

st.info("💡 **Ghi nhớ:** Máy móc là công cụ, anh mới là người ra quyết định. Nếu Tool báo số mà anh thấy cầu đang 'gãy', hãy dừng lại 2 ván để nạp dữ liệu mới cho Tool học lại nhịp.")
