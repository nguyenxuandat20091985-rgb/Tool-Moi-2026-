import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG V25.0 =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_permanent_v25.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    # Loại bỏ trùng lặp và lọc đúng 5 chữ số trước khi lưu
    clean_list = [str(x) for x in data if re.fullmatch(r'\d{5}', str(x))]
    unique_data = list(dict.fromkeys(clean_list))
    with open(DB_FILE, "w") as f:
        json.dump(unique_data[-3000:], f)
    return unique_data

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= THIẾT KẾ UI V22 (GIỮ NGUYÊN CẤU TRÚC) =================
st.set_page_config(page_title="TITAN v25.0 OMNI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 15px;
    }
    .main-num-box {
        font-size: 65px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 8px; border: 1px dashed #444;
        margin: 5px; border-radius: 10px; background: #1c1c1c;
    }
    .lot-box {
        font-size: 45px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px;
    }
    .status-bar { padding: 12px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v25.0 OMNI - SIÊU TRÍ TUỆ Kép</h1>", unsafe_allow_html=True)

# ================= NHẬP LIỆU & XỬ LÝ =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 NẠP DỮ LIỆU (Tự động lọc trùng & sai):", height=120, placeholder="Dán dãy số tại đây...")
    with col_st:
        st.write(f"📊 Dataset sạch: **{len(st.session_state.history)} kỳ**")
        if st.button("🚀 GIẢI MÃ ĐA TẦNG"):
            new_data = re.findall(r"\b\d{5}\b", raw_input)
            if new_data:
                # Gộp và lưu sạch
                st.session_state.history.extend(new_data)
                st.session_state.history = save_db(st.session_state.history)
                
                # PROMPT SIÊU CẤP CHO GEMINI
                prompt = f"""
                Bạn là AI TITAN v25.0, chuyên gia bẻ cầu nhà cái.
                Dữ liệu 100 kỳ: {st.session_state.history[-100:]}
                
                NHIỆM VỤ KHẮT KHE:
                1. Nhận diện cầu Bệt (số rơi liên tục) và cầu Đảo (nhà cái đổi nhịp).
                2. Phân tích ma trận số để chọn ra 2 DÀN CHỦ LỰC (mỗi dàn 3 số).
                3. Loại bỏ 5 số có xác suất trượt cao nhất, chỉ tập trung vào 5 số tiềm năng cho kỳ sau.
                4. Nếu cầu đang quá loạn, đặt 'decision' là 'DỪNG CƯỢC'.

                TRẢ VỀ JSON:
                {{
                  "core_1": "3 số", 
                  "core_2": "3 số", 
                  "support_4": "4 số lót", 
                  "decision": "ĐÁNH/DỪNG", 
                  "logic": "Giải thích ngắn gọn nhịp cầu",
                  "warning": "Cảnh báo bệt/đảo",
                  "conf": 99
                }}
                """
                try:
                    response = neural_engine.generate_content(prompt)
                    res_text = re.search(r'\{.*\}', response.text, re.DOTALL).group()
                    st.session_state.v25_res = json.loads(res_text)
                except:
                    # Thuật toán dự phòng nếu lỗi API
                    st.session_state.v25_res = {
                        "core_1": "123", "core_2": "456", "support_4": "7890",
                        "decision": "CẦN THÊM DỮ LIỆU", "logic": "Lỗi kết nối AI", "warning": "N/A", "conf": 0
                    }
                st.rerun()
        
        if st.button("🗑️ RESET DỮ LIỆU"):
            st.session_state.history = []
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            st.rerun()

# ================= HIỂN THỊ KẾT QUẢ DÀN HÀNG NGANG =================
if "v25_res" in st.session_state:
    res = st.session_state.v25_res
    
    # Cảnh báo bệt/đảo
    st.warning(f"⚠️ **CẢNH BÁO HỆ THỐNG:** {res['warning']}")
    
    # Thanh trạng thái
    color = "#238636" if "ĐÁNH" in res['decision'] else "#da3633"
    st.markdown(f"<div class='status-bar' style='background: {color};'>LỆNH: {res['decision']} ({res['conf']}%)</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # 2 Dàn chủ lực kép
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("<p style='text-align:center; color:#ff5858;'>🎯 CHỦ LỰC 1</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num-box'>{res['core_1']}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='text-align:center; color:#ff5858;'>🎯 CHỦ LỰC 2</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num-box'>{res['core_2']}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<p style='text-align:center; color:#58a6ff;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box' style='margin-top:15px;'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div style='background:#161b22; padding:10px; border-radius:5px;'><b>🔍 PHÂN TÍCH:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    # Tổng hợp dàn 7-8 số để copy
    all_nums = "".join(sorted(set(res['core_1'] + res['core_2'] + res['support_4'])))
    st.text_input("📋 DÀN TỔNG HỢP (KUBET):", all_nums)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê nhịp rơi
if st.session_state.history:
    with st.expander("📊 Phân tích ma trận tần suất (Hot/Cold Numbers)"):
        
        all_digits = "".join(st.session_state.history[-100:])
        st.bar_chart(pd.Series(Counter(all_digits)).sort_index())
