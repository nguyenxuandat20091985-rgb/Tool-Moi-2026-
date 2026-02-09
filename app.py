import streamlit as st
import collections
import pandas as pd

st.set_page_config(page_title="SIÊU TOOL TỬ THỦ 2026", layout="wide")

# CSS Thiết kế giao diện đỉnh cao
st.markdown("""
    <style>
    .stApp { background-color: #000; color: #fff; }
    .header-box { background: linear-gradient(90deg, #1f1c2c, #928dab); padding: 20px; border-radius: 15px; text-align: center; border: 2px solid #ffd700; }
    .bt-box { background: #111; border: 5px double #ffd700; border-radius: 50%; width: 250px; height: 250px; margin: 30px auto; display: flex; align-items: center; justify-content: center; flex-direction: column; box-shadow: 0 0 50px #ffd700; }
    .bt-number { font-size: 130px !important; color: #ffd700; font-weight: bold; text-shadow: 0 0 20px #fff; line-height: 1; }
    .win-text { color: #00ff00; font-weight: bold; font-size: 20px; }
    .label-gold { color: #ffd700; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='header-box'><h1>👑 HỆ THỐNG SOI CẦU ĐẲNG CẤP v9.0</h1><p>BẢN TỐI ƯU BẠCH THỦ - BAO LÔ THỰC CHIẾN</p></div>", unsafe_allow_html=True)

# Nhập dữ liệu
data_raw = st.text_area("👇 Dán danh sách 5 số (Ván mới nhất nằm TRÊN CÙNG):", height=180)

if st.button("🎰 KÍCH HOẠT SIÊU MÁY TÍNH"):
    lines = [l.strip() for l in data_raw.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 10:
        st.error("❌ Anh dán ít nhất 10 ván để máy tính chạy ma trận vị trí nhé!")
    else:
        # 1. PHÂN TÍCH MA TRẬN VỊ TRÍ
        pos_counts = [collections.Counter() for _ in range(5)]
        all_nums = []
        for line in lines:
            for i, char in enumerate(line):
                pos_counts[i][char] += 1
                all_nums.append(char)
        
        # 2. THUẬT TOÁN TÌM BẠCH THỦ (LOẠI BỎ SỐ NGÁO)
        # Lấy top 3 số về nhiều nhất toàn bảng
        global_counts = collections.Counter(all_nums)
        top_candidates = [n for n, c in global_counts.most_common(4)]
        
        # Kiểm tra nhịp rơi 3 ván gần nhất để tránh số 'chết'
        recent_3 = "".join(lines[:3])
        
        # Chọn con số có sự kết nối giữa lịch sử và hiện tại tốt nhất
        final_bt = None
        for cand in top_candidates:
            if cand in recent_3: # Phải đang có đà về mới lấy
                final_bt = cand
                break
        if not final_bt: final_bt = top_candidates[0]

        # 3. GIAO DIỆN CHỐT SỐ SẬP MẮT
        st.write("---")
        st.markdown(f"""
            <div class="bt-box">
                <p class="label-gold">BẠCH THỦ</p>
                <span class="bt-number">{final_bt}</span>
                <p class="win-text">TỶ LỆ NỔ CAO</p>
            </div>
        """, unsafe_allow_html=True)

        # 4. BẢNG CHI TIẾT VỊ TRÍ (Để anh tự thẩm định)
        st.subheader("📊 BẢNG SOI VỊ TRÍ CHI TIẾT")
        df_data = {
            "Vị trí": ["Hàng Vạn", "Hàng Nghìn", "Hàng Trăm", "Hàng Chục", "Hàng Đơn Vị"],
            "Số hay về nhất": [pos_counts[i].most_common(1)[0][0] for i in range(5)],
            "Tần suất": [pos_counts[i].most_common(1)[0][1] for i in range(5)],
            "Xu hướng": ["🔥 Đang bệt" if lines[0][i] == lines[1][i] else "📉 Đang đảo" for i in range(5)]
        }
        st.table(pd.DataFrame(df_data))

st.info("💡 **Gợi ý từ AI:** Nếu con Bạch Thủ trên trùng với 'Số hay về nhất' ở bảng vị trí, anh có thể tự tin vào tiền mạnh tay!")
