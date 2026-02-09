import streamlit as st
import collections

st.set_page_config(page_title="TOOL BẠCH THỦ 2026", layout="wide")

st.markdown("""
    <style>
    .bach-thu-box { background: linear-gradient(135deg, #ff4b4b 0%, #ff8000 100%); padding: 30px; border-radius: 20px; text-align: center; color: white; margin-bottom: 30px; border: 5px solid #fff; box-shadow: 0px 10px 20px rgba(0,0,0,0.3); }
    .number-vip { font-size: 120px !important; font-weight: bold; line-height: 1; text-shadow: 2px 2px 10px #000; }
    .stButton>button { background-color: #28a745; color: white; font-size: 25px; height: 3em; border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏆 CHỐT SỐ BẠCH THỦ - ĐỘC THỦ LÔ AI")

data_input = st.text_area("👉 Nhập kết quả (Càng nhiều càng chuẩn):", height=150)

if st.button("🔥 CHỐT SỐ BẠCH THỦ TẬN TÂY"):
    lines = [l.strip() for l in data_input.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 10:
        st.error("⚠️ Anh ơi, nhập ít nhất 10 kỳ thì AI mới lọc được con Bạch Thủ 'xịn' nhé!")
    else:
        # Lấy tất cả các số từ tất cả các vị trí để phân tích nhịp chung
        all_numbers = []
        for line in lines:
            all_numbers.extend([int(d) for d in line])
        
        # Tìm con số "vua" (về đều và đang trong nhịp rơi)
        counts = collections.Counter(all_numbers)
        bach_thu = counts.most_common(1)[0][0]
        
        # Tính toán tỉ lệ tin cậy dựa trên mật độ xuất hiện
        confidence = min(99.8, (counts[bach_thu] / len(all_numbers)) * 500)

        # HIỂN THỊ BẠCH THỦ TO TRÀN MÀN HÌNH
        st.markdown(f"""
            <div class="bach-thu-box">
                <span style="font-size: 30px; font-weight: bold;">🌟 BẠCH THỦ KIM CƯƠNG 🌟</span><br>
                <span class="number-vip">{bach_thu}</span><br>
                <span style="font-size: 25px;">Độ tin cậy: {confidence:.1f}%</span>
            </div>
        """, unsafe_allow_html=True)

        # Gợi ý thêm dàn phụ
        st.subheader("📋 Dàn dự phòng (Nếu anh muốn đánh bao lô)")
        top_3 = counts.most_common(4)[1:] # Lấy 3 số tiếp theo
        cols = st.columns(3)
        for idx, (num, freq) in enumerate(top_3):
            cols[idx].metric(label=f"SỐ PHỤ {idx+1}", value=num, delta=f"Nhịp {freq}")

st.info("💡 Mẹo: Con Bạch Thủ trên là con số có 'từ trường' mạnh nhất trong bảng kết quả anh vừa nhập.")
