import streamlit as st
import collections
import numpy as np

st.set_page_config(page_title="AI DYNAMIC 2026 - CHỐT SỐ BIẾN THIÊN", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #00ff00; }
    .status-card { background: #111; border-left: 5px solid #ff0000; padding: 15px; margin-bottom: 20px; }
    .bo-so-vip { font-size: 80px !important; color: #ffff00; font-weight: bold; text-shadow: 3px 3px #ff0000; line-height: 1.2; }
    .highlight { color: #ff00ff; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ AI DYNAMIC v14.0: HỆ THỐNG TỔNG HỢP NGUỒN TỐI TÂN")
st.markdown("<p class='highlight'>Cảnh báo: Dữ liệu biến thiên theo từng kỳ - Cập nhật liên tục</p>", unsafe_allow_html=True)

# Nhập dữ liệu
data_raw = st.text_area("👇 Dán danh sách 5 số (Ván mới nhất TRÊN CÙNG):", height=200)

if st.button("🔄 PHÂN TÍCH BIẾN THIÊN & CHỐT BỘ 9 SỐ"):
    lines = [l.strip() for l in data_raw.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 5:
        st.error("❌ Cần ít nhất 5 kỳ gần nhất để kích hoạt chế độ Biến Thiên!")
    else:
        # --- THUẬT TOÁN TỔNG HỢP NGUỒN THÔNG MINH ---
        
        # 1. Trọng số thời gian: Kỳ càng mới điểm càng cao
        weighted_counts = collections.Counter()
        for i, line in enumerate(lines[:15]): # Chỉ tập trung 15 kỳ gần nhất
            weight = 15 - i # Kỳ mới nhất (i=0) có điểm là 15, kỳ cũ giảm dần
            for char in line:
                weighted_counts[char] += weight

        # 2. Xử lý "Số ngáo" (Số nổ quá dày trong 3 kỳ gần nhất sẽ bị giảm ưu tiên)
        recent_3 = "".join(lines[:3])
        recent_counts = collections.Counter(recent_3)
        
        final_scores = []
        for num in "0123456789":
            score = weighted_counts[num]
            if recent_counts[num] >= 3: score *= 0.5 # Giảm nhiệt nếu nổ quá 'điên'
            final_scores.append((num, score))
        
        # Sắp xếp theo điểm số thực tế
        final_scores.sort(key=lambda x: x[1], reverse=True)
        top_9 = [x[0] for x in final_scores[:9]]

        # 3. Chia thành 3 bộ Tam Tinh độc lập
        bo_1 = top_9[0:3]
        bo_2 = top_9[3:6]
        bo_3 = top_9[6:9]

        # HIỂN THỊ KẾT QUẢ SẬP MẮT
        st.write("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"<div class='status-card'><h3>BỘ 1: ƯU TIÊN 1</h3><p class='bo-so-vip'>{''.join(bo_1)}</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='status-card'><h3>BỘ 2: ƯU TIÊN 2</h3><p class='bo-so-vip'>{''.join(bo_2)}</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='status-card'><h3>BỘ 3: DỰ PHÒNG</h3><p class='bo-so-vip'>{''.join(bo_3)}</p></div>", unsafe_allow_html=True)

        # PHẦN CHIẾN THUẬT
        st.write("---")
        st.subheader("🎯 CHIẾN THUẬT ĐẦU TƯ (AI ADVICE)")
        
        # Phân tích xem cầu đang Bệt hay Đảo
        is_bet = any(lines[0][i] == lines[1][i] for i in range(5))
        if is_bet:
            st.warning("⚠️ PHÁT HIỆN CẦU BỆT: Giữ nguyên bộ số cũ và vào tiền đều tay.")
        else:
            st.success("🔄 CẦU ĐẢO NHỊP: AI đã cập nhật bộ số mới theo dòng chảy.")

st.info("💡 **Gợi ý:** Anh hãy nhập thêm 1 kỳ mới nhất vừa ra và bấm nút lần nữa, anh sẽ thấy 3 bộ số này thay đổi ngay lập tức để bám đuổi kết quả!")
