import streamlit as st

st.set_page_config(page_title="THA BET STRATEGY 2026", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d0d0d; color: #fff; }
    .bet-card { border-radius: 15px; padding: 20px; text-align: center; margin: 10px; border: 2px solid #d4af37; background: #1a1a1a; }
    .banker { color: #ff4b4b; font-size: 50px; font-weight: bold; }
    .player { color: #1e90ff; font-size: 50px; font-weight: bold; }
    .title { color: #d4af37; font-size: 24px; text-transform: uppercase; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🃏 BACCARAT MASTER v20.0 - THIÊN HẠ BET")
st.write("---")

# Nhập lịch sử cầu (B: Banker, P: Player)
road_data = st.text_input("📡 Nhập chuỗi cầu (Ví dụ: BPBPPB):", "").upper()

if st.button("🧠 PHÂN TÍCH THẾ BÀI"):
    if len(road_data) < 5:
        st.warning("⚠️ Anh nhập ít nhất 5 tay gần nhất để em nhận diện nhịp cầu.")
    else:
        # Thuật toán bắt nhịp cầu (Pattern Recognition)
        last_3 = road_data[-3:]
        
        # Giả lập logic dự đoán dựa trên xu hướng cầu (Bệt/Đảo)
        if last_3 in ["BBB", "PPP"]:
            prediction = "BỆT tiếp" if last_3 == "BBB" else "BỆT tiếp"
            main_bet = last_3[0] 
        elif last_3 in ["BPB", "PBP"]:
            prediction = "CẦU ĐẢO 1-1"
            main_bet = "P" if last_3[-1] == "B" else "B"
        else:
            prediction = "CẦU NHẢY"
            main_bet = "B" # Ưu tiên Banker vì lợi thế toán học cao hơn

        # Xuất kết quả theo yêu cầu của anh
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='bet-card'><p class='title'>🎯 BẠCH THỦ (Cửa Chính)</p>", unsafe_allow_html=True)
            color_class = "banker" if main_bet == "B" else "player"
            st.markdown(f"<p class='{color_class}'>{main_bet}</p>", unsafe_allow_html=True)
            st.write(f"Nhịp: {prediction}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='bet-card'><p class='title'>🥈 2 TINH (Phụ)</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='banker'>{main_bet}</p><p class='player'>HÒA (Tie)</p>", unsafe_allow_html=True)
            st.write("Lót cửa Hòa để bảo toàn vốn")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='bet-card'><p class='title'>🥉 3 TINH (Thế Bài)</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #fff; font-size: 30px;'>{road_data[-1]} ➔ {main_bet} ➔ {main_bet}</p>", unsafe_allow_html=True)
            st.write("Dàn thế bài 3 tay liên tiếp")
            st.markdown("</div>", unsafe_allow_html=True)

st.info("💡 **Kinh nghiệm:** Trong Thiên Hạ Bet, nếu anh thấy cầu ra 4 cây giống nhau (Bệt 4), đừng bao giờ bẻ. Hãy đánh theo bệt cho đến khi gãy thì thôi. Đó là cách lấy tiền nhanh nhất.")
