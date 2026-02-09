import streamlit as st
import collections
import random

st.set_page_config(page_title="HỆ THỐNG TAM TINH 9 SỐ", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #00050a; color: #ffffff; }
    .main-card { background: #111; border: 2px solid #ffd700; border-radius: 15px; padding: 20px; margin: 10px; text-align: center; }
    .bo-so { font-size: 60px !important; color: #ffd700; font-weight: bold; text-shadow: 0 0 10px #ffd700; }
    .tieude-bo { color: #00ffcc; font-size: 20px; font-weight: bold; border-bottom: 1px solid #333; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏆 SIÊU TỔ HỢP TAM TINH 2026")
st.write("---")

data_raw = st.text_area("👇 Dán kết quả (Ván mới nhất TRÊN CÙNG):", height=150)

if st.button("🚀 XUẤT 3 CẶP TAM TINH CHÍNH XÁC"):
    lines = [l.strip() for l in data_raw.split('\n') if len(l.strip()) == 5]
    
    if len(lines) < 10:
        st.error("❌ Anh dán ít nhất 10 kỳ để em tính toán 3 bộ số khác nhau cho chuẩn!")
    else:
        # Lấy toàn bộ số và phân tích tần suất
        full_pool = "".join(lines)
        counts = collections.Counter(full_pool)
        
        # Sắp xếp số theo độ mạnh giảm dần
        sorted_nums = [n for n, c in counts.most_common(10)]
        
        # Thuật toán chia 3 Bộ khác nhau:
        # Bộ 1: Ưu tiên Cầu Bệt (những số vừa nổ ở kỳ gần nhất)
        bo_1 = list(lines[0][:3]) 
        if len(set(bo_1)) < 3: # Nếu trùng thì lấy thêm số mạnh
            for n in sorted_nums:
                if n not in bo_1: bo_1.append(n)
                if len(bo_1) == 3: break
        
        # Bộ 2: Ưu tiên Nhịp Rơi (những số có tần suất ổn định nhất)
        bo_2 = []
        for n in sorted_nums:
            if n not in bo_1:
                bo_2.append(n)
            if len(bo_2) == 3: break
            
        # Bộ 3: Ưu tiên Cầu Đảo (những số gan hoặc số bóng)
        bo_3 = []
        reversed_nums = sorted_nums[::-1]
        for n in reversed_nums:
            if n not in bo_1 and n not in bo_2:
                bo_3.append(n)
            if len(bo_3) == 3: break
        if len(bo_3) < 3: bo_3 = ["1", "0", "5"] # Dự phòng nếu thiếu số

        # HIỂN THỊ 3 CẶP
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""<div class="main-card">
                <p class="tieude-bo">BỘ 1: CẦU BỆT 🔥</p>
                <p class="bo-so">{''.join(bo_1)}</p>
            </div>""", unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""<div class="main-card">
                <p class="tieude-bo">BỘ 2: NHỊP RƠI 📈</p>
                <p class="bo-so">{''.join(bo_2)}</p>
            </div>""", unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""<div class="main-card">
                <p class="tieude-bo">BỘ 3: CẦU ĐẢO 🌀</p>
                <p class="bo-so">{''.join(bo_3)}</p>
            </div>""", unsafe_allow_html=True)

        st.success(f"✅ Tổng hợp 9 số: {', '.join(bo_1 + bo_2 + bo_3)}")
        st.info("💡 **Cách chơi:** Anh có thể đánh 3 bộ này riêng biệt. Chỉ cần 1 trong 3 bộ nổ chính xác 3 con trong giải là anh thắng đậm!")

st.markdown("<p style='text-align: center; color: #444;'>Phiên bản Tam Tinh Tổ Hợp - Anti Ngáo v12.0</p>", unsafe_allow_html=True)
