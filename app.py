import streamlit as st
import re
from collections import Counter
import math

# --- PHẦN 1: BỘ NÃO AI TÍNH TOÁN TRỰC TIẾP (KHÔNG LƯU TRỮ TRUNG GIAN) ---
class TitanNeuralAI:
    def __init__(self):
        self.shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

    def analyze_now(self, raw_text):
        # 1. Bóc tách số ngay lập tức từ ô nhập liệu
        found = re.findall(r'\d{5}', str(raw_text))
        if not found: return ["-"] * 5, 0, []
        
        data = [[int(d) for d in item] for item in found]
        # Đảo ngược để số mới nhất nằm trên cùng
        data.reverse() 
        
        # 2. Thuật toán xử lý xác suất nhịp ngắn (Focus 15 kỳ gần nhất)
        recent = data[:15]
        all_nums = [d for row in recent for d in row]
        freq = Counter(all_nums)
        
        gaps = {i: 20 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row: gaps[i] = idx; break

        scores = {}
        for i in range(10):
            f_score = freq.get(i, 0) * 8 # Tăng trọng số tần suất
            g_score = 60 if gaps[i] == 1 else 30 if gaps[i] == 0 else 0
            s_score = 40 if gaps[self.shadow_map[i]] == 0 else 0
            scores[str(i)] = f_score + g_score + s_score
            
        # Lấy 5 số có điểm cao nhất
        top_5 = [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:5]
        
        # 3. Tính độ tin cậy dựa trên Entropy thực tế của dữ liệu vừa nhập
        try:
            counts = [all_nums.count(i) for i in set(all_nums)]
            total = sum(counts)
            entropy = -sum((c/total)*math.log2(c/total) for c in counts if c > 0)
            acc = int(max(25, min(99, (3.32 - entropy) * 450)))
        except: acc = 25
        
        return top_5, acc, data

# --- PHẦN 2: GIAO DIỆN PHẢN HỒI THỜI GIAN THỰC ---
st.set_page_config(page_title="TITAN v8.6 NEURAL", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; }
    .gold-container { display: flex; justify-content: space-between; margin: 15px 0; }
    .gold-item {
        background: linear-gradient(135deg, #ffd700, #b8860b);
        border-radius: 10px; width: 18%; aspect-ratio: 1/1;
        display: flex; align-items: center; justify-content: center;
        font-size: 28px; font-weight: bold; color: #000;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        border: 2px solid #fff;
    }
    .stButton button { 
        background: linear-gradient(to bottom, #ffd700, #daa520) !important; 
        color: black !important; font-weight: bold !important; 
        border-radius: 30px !important; height: 50px; font-size: 18px !important;
    }
    .stTextArea textarea { border: 2px solid #ffd700 !important; background: #1a1c23 !important; color: white !important; }
    .result-box { background: #ffd700; color: black; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 22px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

ai = TitanNeuralAI()

st.markdown("<h2 style='text-align: center; color: #ffd700;'>⚡ TITAN NEURAL v8.6</h2>", unsafe_allow_html=True)

# Ô nhập liệu luôn hiển thị phía trên
raw_input = st.text_area("Dán dữ liệu kỳ quay:", placeholder="Dán dãy số tại đây (Ví dụ: 24373)...", height=120)

if st.button("🚀 PHÂN TÍCH NGAY LẬP TỨC", use_container_width=True):
    if raw_input:
        top_5, acc, history = ai.analyze_now(raw_input)
        
        if history:
            st.markdown(f"<p style='text-align: center; color: #00ff00; font-weight: bold; font-size: 20px;'>Độ tin cậy AI: {acc}%</p>", unsafe_allow_html=True)
            
            # Hiển thị 5 số
            res_html = f"""
            <div class="gold-container">
                <div class="gold-item">{top_5[0]}</div>
                <div class="gold-item">{top_5[1]}</div>
                <div class="gold-item">{top_5[2]}</div>
                <div class="gold-item">{top_5[3]}</div>
                <div class="gold-item">{top_5[4]}</div>
            </div>
            """
            st.markdown(res_html, unsafe_allow_html=True)
            
            st.markdown(f"<div class='result-box'>🎯 CHỐT 3 TINH: {top_5[0]} - {top_5[1]} - {top_5[2]}</div>", unsafe_allow_html=True)
            
            with st.expander("📊 Kiểm tra dữ liệu AI đã đọc"):
                for idx, row in enumerate(history[:10]):
                    st.write(f"Kỳ {idx+1}: {''.join(map(str, row))}")
        else:
            st.error("Không tìm thấy dãy 5 số trong nội dung anh dán!")
    else:
        st.warning("Anh chưa dán số vào kìa!")
else:
    st.info("💡 Hãy dán dữ liệu và ấn nút để AI bắt đầu tính toán.")
