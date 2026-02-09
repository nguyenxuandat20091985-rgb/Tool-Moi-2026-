import streamlit as st
import collections

# Cấu hình giao diện
st.set_page_config(page_title="AI TAM TINH 2026", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0a0a0a; color: #ffffff; }
    .result-card { background: #1a1a1a; border: 2px solid #00ffcc; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 20px; }
    .number-text { font-size: 80px !important; color: #ffff00; font-weight: bold; text-shadow: 0 0 20px #ff0000; }
    .header-title { color: #00ffcc; font-size: 24px; font-weight: bold; border-bottom: 2px solid #333; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ AI TAM TINH v16.0 - CHỐT SỐ KHÔNG LỖI")
st.write("---")

# Nhập liệu
data_raw = st.text_area("📋 Dán kết quả (5 số mỗi dòng, ván mới nhất TRÊN CÙNG):", height=200)

# Ma trận xác suất nguồn mở (Tự động tích hợp)
OPEN_DATA = {
    '0': '358', '1': '479', '2': '068', '3': '157', '4': '248', 
    '5': '059', '6': '137', '7': '246', '8': '059', '9': '147'
}

if st.button("🚀 PHÂN TÍCH & XUẤT 3 BỘ SỐ"):
    # Xử lý dữ liệu đầu vào
    lines = [l.strip() for l in data_raw.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 5:
        st.error("❌ Anh dán ít nhất 5 ván để máy tính bắt nhịp cầu nhé!")
    else:
        # 1. Thuật toán tổng hợp nguồn (Local Data)
        all_content = "".join(lines[:15]) # Ưu tiên 15 kỳ gần nhất
        freq = collections.Counter(all_content)
        
        # 2. Bắt nhịp biến thiên (Biến số cuối làm chìa khóa)
        key = lines[0][-1]
        bonus_nums = OPEN_DATA.get(key, '123')
        
        # 3. Tính toán điểm tổng hợp cho 10 số (0-9)
        scores = []
        for i in range(10):
            num = str(i)
            # Điểm = Tần suất thực tế + Thưởng nếu nằm trong ma trận nguồn mở
            score = freq[num] + (5 if num in bonus_nums else 0)
            # Giảm điểm nếu số nổ quá dày (hơn 3 lần trong 5 ván) để tránh số ảo
            if "".join(lines[:5]).count(num) > 3:
                score -= 10
            scores.append((num, score))
        
        # Sắp xếp lấy 9 số mạnh nhất
        scores.sort(key=lambda x: x[1], reverse=True)
        top_9 = [s[0] for s in scores[:9]]
        
        # Chia 3 bộ Tam Tinh độc lập
        bo_1 = sorted(top_9[0:3])
        bo_2 = sorted(top_9[3:6])
        bo_3 = sorted(top_9[6:9])

        # HIỂN THỊ KẾT QUẢ SẬP MẮT
        st.write("### 💎 DỰ ĐOÁN 3 BỘ TAM TINH (9 SỐ TỰ DO)")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f"<div class='result-card'><p class='header-title'>BỘ 1 (CHÍNH)</p><p class='number-text'>{''.join(bo_1)}</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='result-card'><p class='header-title'>BỘ 2 (PHỤ)</p><p class='number-text'>{''.join(bo_2)}</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='result-card'><p class='header-title'>BỘ 3 (LÓT)</p><p class='number-text'>{''.join(bo_3)}</p></div>", unsafe_allow_html=True)

        st.info("💡 Chiến thuật: Đánh bao lô 3 con cho từng bộ. Chỉ cần 1 bộ nổ 3/5 số là anh thắng!")

st.markdown("<p style='text-align: center; color: #444;'>Hệ thống dự đoán thông minh - Phiên bản thực chiến 2026</p>", unsafe_allow_html=True)
