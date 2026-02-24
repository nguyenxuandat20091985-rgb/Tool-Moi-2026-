import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG TITAN v22.1 =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_neural_memory_v22.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU & BỘ NHỚ =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_memory(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-2000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()

# ================= GIAO DIỆN DARK MODE PRO =================
st.set_page_config(page_title="TITAN v22.1 OMNI - KUBET SPECIAL", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .prediction-card {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 2px solid #58a6ff; border-radius: 15px; padding: 30px;
        box-shadow: 0 0 40px rgba(88, 166, 255, 0.15);
    }
    .main-number { font-size: 100px; font-weight: 900; color: #ff5858; text-shadow: 0 0 30px #ff5858; text-align: center; letter-spacing: 15px; }
    .secondary-number { font-size: 55px; font-weight: 700; color: #58a6ff; text-align: center; opacity: 0.8; letter-spacing: 10px; }
    .warning-box { background: #331010; color: #ff7b72; padding: 20px; border-radius: 10px; border: 2px solid #6e2121; text-align: center; font-size: 18px; margin-bottom: 15px; }
    .logic-box { background: #161b22; border-left: 4px solid #58a6ff; padding: 15px; margin: 15px 0; font-style: italic; color: #8b949e; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🧬 TITAN v22.1 PRO OMNI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Hệ thống Giải mã 3 Càng Không Cố Định - Kubet Special Edition</p>", unsafe_allow_html=True)

# ================= LOGIC XỬ LÝ CHÍNH =================
raw_input = st.text_area("📥 NẠP DỮ LIỆU (Copy từ bảng lịch sử hoặc dán dãy 5 số):", height=150, placeholder="Ví dụ: 78931\n88657\n...")

c_btn1, c_btn2 = st.columns(2)
with c_btn1:
    if st.button("🚀 KÍCH HOẠT GIẢI MÃ BẺ CẦU"):
        # Lọc sạch dữ liệu bẩn: lấy đúng các cụm 5 chữ số
        clean_data = re.findall(r"\d{5}", raw_input)
        if clean_data:
            st.session_state.history.extend(clean_data)
            save_memory(st.session_state.history)
            
            # PROMPT TITAN v22.1 - SIÊU PHÂN TÍCH
            prompt = f"""
            Hệ thống: TITAN v22.1. Chuyên gia bẻ cầu Kubet/Lotobet 3D Không cố định.
            Dữ liệu (100 kỳ): {st.session_state.history[-100:]}
            
            YÊU CẦU PHÂN TÍCH:
            1. PHẢN XẠ CẦU: Tìm số bệt, số bóng kỳ trước (0-5, 1-6, 2-7, 3-8, 4-9).
            2. VỊ TRÍ VÀNG: Phân tích tần suất 10 số tại các hàng Chục nghìn, Ngàn, Trăm, Chục, Đơn vị.
            3. CHỐT 3 SỐ CHỦ LỰC: Phải xuất hiện trong 5 số của kết quả (Xác suất > 95%).
            4. DÀN 7 SỐ KUBET: Gồm 3 số chủ lực + 4 số lót (không trùng).
            5. CẢNH BÁO: Nếu cầu đang chạy ảo, lặp vô nghĩa hoặc dấu hiệu 'kìm' số => warning: true.

            TRẢ VỀ JSON:
            {{
                "main_3": "ABC", 
                "support_4": "DEFG", 
                "logic": "Giải thích sắc bén dựa trên bóng số và nhịp cầu", 
                "warning": false, 
                "confidence": 98
            }}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    st.session_state.last_prediction = json.loads(json_match.group())
                else:
                    st.error("AI trả về định dạng sai - Thử lại!")
            except Exception as e:
                # Fallback Statisics
                all_nums = "".join(st.session_state.history[-50:])
                top_7 = [x[0] for x in Counter(all_nums).most_common(7)]
                st.session_state.last_prediction = {
                    "main_3": "".join(top_7[:3]),
                    "support_4": "".join(top_7[3:]),
                    "logic": "Dữ liệu ngoại tuyến: Sử dụng thuật toán tần suất nhịp rơi kỳ gần nhất.",
                    "warning": False,
                    "confidence": 70
                }
            st.rerun()

with c_btn2:
    if st.button("🗑️ RESET DỮ LIỆU"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ ĐẲNG CẤP =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    if res.get('warning') or res.get('confidence', 0) < 75:
        st.markdown(f"<div class='warning-box'>⚠️ CẢNH BÁO: NHÀ CÁI ĐANG ĐIỀU TIẾT CẦU ẢO - KHÔNG NÊN VÀO TIỀN LỚN!</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='logic-box'><b>🧬 GIẢI MÃ:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; color:#ff7b72; font-weight:bold; margin-bottom:0;'>🎯 3 SỐ CHỦ LỰC (VÀO TIỀN CHÍNH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='main-number'>{res['main_3']}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; color:#58a6ff; font-weight:bold; margin-top:20px; margin-bottom:0;'>🛡️ DÀN 4 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='secondary-number'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    # Khu vực sao chép nhanh cho Kubet
    full_7 = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ ĐỂ CHỌN TRÊN KUBET:", full_7)
    
    st.progress(res.get('confidence', 50) / 100)
    st.markdown(f"<p style='text-align:right; font-size:12px; color:#58a6ff;'>Hệ thống tự tin: {res.get('confidence')}%</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer thống kê
if st.session_state.history:
    with st.expander("📊 Xem nhịp cầu thực tế (Data Insight)"):
        all_digits = "".join(st.session_state.history[-50:])
        st.bar_chart(pd.Series(Counter(all_digits)).sort_index())
