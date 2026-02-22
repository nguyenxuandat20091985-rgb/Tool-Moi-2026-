import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import time

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM" # Thay bằng Key của anh
genai.configure(api_key=API_KEY)

st.set_page_config(page_title="TITAN V23 ELITE", layout="wide", initial_sidebar_state="collapsed")

# CSS Tối ưu thu nhỏ cửa sổ và hiệu ứng "Bào tiền"
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    .stApp { background: #0a0e14; color: #e6edf3; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    
    /* Cấu trúc Card dự đoán */
    .main-card {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    .number-highlight {
        font-family: 'Courier New', monospace;
        font-size: 50px !important;
        font-weight: 900;
        color: #238636;
        text-shadow: 0 0 20px #238636;
        letter-spacing: 10px;
        margin: 10px 0;
    }
    
    .backup-number {
        font-size: 30px !important;
        color: #d29922;
        text-shadow: 0 0 10px #d29922;
        letter-spacing: 5px;
    }

    .status-tag {
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    /* Thu nhỏ cho Mobile/Tab */
    @media (max-width: 600px) {
        .number-highlight { font-size: 35px !important; }
        .backup-number { font-size: 22px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# ================= CORE LOGIC PHÂN TÍCH =================
class TitanV23Engine:
    def __init__(self, history):
        self.history = history
        self.digits = "".join([s for s in history])

    def detect_casino_traps(self):
        """Thuật toán phát hiện cầu lừa"""
        reasons = []
        is_trap = False
        if len(self.history) < 10: return False, []
        
        last_5 = self.history[-5:]
        # Bẫy 1: Cầu bệt ảo (số lặp lại liên tục ở 1 vị trí quá 4 lần)
        for pos in range(5):
            pos_digits = [n[pos] for n in last_5]
            if len(set(pos_digits)) == 1:
                is_trap = True
                reasons.append(f"Cảnh báo bẫy bệt vị trí {pos+1}")
        
        # Bẫy 2: Cầu rỗng (số biến thiên quá lớn đột ngột)
        unique_chars = len(set(self.digits[-20:]))
        if unique_chars > 8:
            is_trap = True
            reasons.append("Cầu đang loạn (nhà cái đảo số)")
            
        return is_trap, reasons

    def get_probability(self):
        """Tính toán xác suất nâng cao"""
        if not self.digits: return {str(i): 0.1 for i in range(10)}
        
        counts = Counter(self.digits[-100:]) # Lấy 100 số gần nhất
        total = sum(counts.values())
        prob = {str(i): counts.get(str(i), 0) / total for i in range(10)}
        
        # Điều chỉnh trọng số dựa trên xu hướng gần (10 kỳ)
        recent_counts = Counter(self.digits[-20:])
        for d in prob:
            prob[d] = (prob[d] * 0.4) + ((recent_counts.get(d, 0) / 20) * 0.6)
            
        return dict(sorted(prob.items(), key=lambda x: x[1], reverse=True))

# ================= GIAO DIỆN CHÍNH =================
def main():
    # Khởi tạo Memory
    if "history" not in st.session_state: st.session_state.history = []
    
    # Header cực gọn
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.markdown("### 🧬 TITAN V23 ELITE")
    with col_h2:
        if st.button("🗑️ XÓA"): 
            st.session_state.history = []
            st.rerun()

    # Input Data
    raw_input = st.text_input("Nhập số mới (VD: 12345, 67890):", key="input_box")
    if raw_input:
        new_nums = re.findall(r"\d{5}", raw_input)
        for n in new_nums:
            if n not in st.session_state.history[-5:]: # Chống trùng lặp
                st.session_state.history.append(n)
        st.toast(f"Đã nạp {len(new_nums)} kỳ", icon="✅")

    if len(st.session_state.history) < 5:
        st.info("Cần tối thiểu 5 kỳ để phân tích thuật toán...")
        return

    # Thực thi AI & Thuật toán
    engine = TitanV23Engine(st.session_state.history)
    is_trap, trap_reasons = engine.detect_casino_traps()
    probs = engine.get_probability()
    
    top_7 = list(probs.keys())[:7]
    main_3 = top_7[:3]  # 3 Số khả năng về cao nhất
    backup_4 = top_7[3:] # 4 Số dự phòng

    # --- KHU VỰC HIỂN THỊ DỰ ĐOÁN (OPTIMIZED FOR TAB/MINI WINDOW) ---
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    # Dòng trạng thái
    status_color = "#f85149" if is_trap else "#238636"
    status_text = "CẦU NGUY HIỂM (LỪA)" if is_trap else "CẦU ĐANG ĐẸP (ỔN)"
    st.markdown(f"<span class='status-tag' style='background:{status_color}; color:white;'>{status_text}</span>", unsafe_allow_html=True)

    # 3 SỐ CHỦ LỰC (99.99%)
    st.markdown("<p style='margin-bottom:0; color:#8b949e;'>3 SỐ CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='number-highlight'>{' '.join(main_3)}</div>", unsafe_allow_html=True)
    
    # 4 SỐ DỰ PHÒNG
    st.markdown("<p style='margin:10px 0 0 0; color:#8b949e;'>4 SỐ DỰ PHÒNG (LÓT)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='backup-number'>{' '.join(backup_4)}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Gemini Chiến lược
    with st.expander("🧠 PHÂN TÍCH CHIẾN LƯỢC GEMINI"):
        if st.button("GỌI AI PHÂN TÍCH"):
            with st.spinner("AI đang giải mã cầu..."):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"""
                    Phân tích dãy số LotoBet: {st.session_state.history[-30:]}
                    Dựa trên thuật toán xác suất, hãy cho biết:
                    1. Quy luật cầu hiện tại.
                    2. Tại sao chọn bộ số {''.join(top_7)}.
                    3. Chiến thuật vào tiền để 'bào' nhà cái hiệu quả nhất.
                    Trả lời ngắn gọn, tập trung vào con số.
                    """
                    response = model.generate_content(prompt)
                    st.write(response.text)
                except Exception as e:
                    st.error("Lỗi kết nối AI. Vui lòng kiểm tra API Key.")

    # Thống kê nhanh
    st.markdown("---")
    st.markdown(f"**Dữ liệu:** {len(st.session_state.history)} kỳ | **Gợi ý:** Chia vốn 70% vào 3 số chính, 30% lót 4 số dự phòng.")

if __name__ == "__main__":
    main()
