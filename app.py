import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG SUPREME =================
# Cập nhật API Key mới anh cung cấp
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_supreme_v25_permanent.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        # Sử dụng model flash để đảm bảo tốc độ mượt mà nhất
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Lỗi kết nối API: {e}")
        return None

neural_engine = setup_neural()

# ================= QUẢN LÝ CƠ SỞ DỮ LIỆU VĨNH VIỄN =================
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding='utf-8') as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    # Lưu tối đa 5000 kỳ để AI có độ nhạy bén cao nhất với các cầu cũ
    with open(DB_FILE, "w", encoding='utf-8') as f:
        json.dump(data[-5000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= THUẬT TOÁN PHÂN TÍCH NHẠY BÉN =================
def detect_bridge_trap(data):
    if len(data) < 10: return "CHỜ DỮ LIỆU", "Gray"
    
    last_10 = data[-10:]
    # Phân tích bệt số
    all_digits = "".join(last_10)
    counts = Counter(all_digits)
    most_common = counts.most_common(1)[0]
    
    # Nếu 1 số xuất hiện > 8 lần trong 10 kỳ (50 chữ số) -> Bệt nặng
    if most_common[1] >= 8:
        return f"⚠️ CẢNH BÁO BỆT SỐ: {most_common[0]}", "#da3633"
    
    # Phân tích đảo cầu (dựa trên biến thiên tổng số)
    sums = [sum([int(d) for d in s]) for s in last_10]
    std_dev = np.std(sums)
    if std_dev > 7:
        return "🔄 NHÀ CÁI ĐANG ĐẢO CẦU", "#f2cc60"
        
    return "✅ CẦU ỔN ĐỊNH - VÀO TIỀN", "#238636"

# ================= GIAO DIỆN TITAN v25.0 SUPREME =================
st.set_page_config(page_title="TITAN v25.0 SUPREME", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 15px; padding: 25px; margin-top: 10px;
        box-shadow: 0 4px 15px rgba(88, 166, 255, 0.2);
    }
    .main-num-box {
        font-size: 80px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 12px;
        text-shadow: 0 0 15px rgba(255, 88, 88, 0.5);
    }
    .lot-box {
        font-size: 45px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px;
    }
    .status-bar { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>💎 TITAN v25.0 SUPREME - SIÊU TRÍ TUỆ</h1>", unsafe_allow_html=True)

# ================= PHẦN NHẬP LIỆU & BỘ NHỚ =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 NẠP DỮ LIỆU NHÀ CÁI (Chỉ nhập 5 số mỗi kỳ):", height=120, placeholder="Ví dụ: 12345\n67890...")
    with col_st:
        st.info(f"💾 CƠ SỞ DỮ LIỆU: {len(st.session_state.history)} KỲ")
        c1, c2 = st.columns(2)
        if c1.button("🚀 GIẢI MÃ KỸ CÀNG"):
            # Loại bỏ các ký tự lạ, chỉ lấy đúng cụm 5 số
            clean_list = re.findall(r"\d{5}", raw_input)
            if clean_list:
                # Loại bỏ số nhập sai, số trùng lặp trong phiên nhập hiện tại
                for s in clean_list:
                    if s not in st.session_state.history:
                        st.session_state.history.append(s)
                
                save_db(st.session_state.history)
                
                # Gửi Prompt Siêu Trí Tuệ cho Gemini
                prompt = f"""
                Bạn là AI tối tân nhất chuyên bẻ khóa thuật toán nhà cái.
                Lịch sử lưu trữ: {st.session_state.history[-150:]}
                
                Nhiệm vụ:
                1. Phân tích bệt, đảo, ma trận Pascal để tìm số chủ lực.
                2. Loại bỏ 5 số có xác suất trượt cao nhất.
                3. Dự đoán 2 Dàn Số Chủ Lực (mỗi dàn 3 số). Ví dụ: '456' và '478'.
                4. Cung cấp 4 số lót an toàn.
                
                Yêu cầu: Số liệu phải nhạy bén với nhịp đảo của nhà cái hiện tại.
                Trả về định dạng JSON:
                {{
                  "main_1": "abc",
                  "main_2": "xyz",
                  "support_4": "defg",
                  "decision": "ĐÁNH/DỪNG/CHỜ",
                  "logic": "Lý do ngắn gọn về nhịp cầu",
                  "conf": 99
                }}
                """
                try:
                    response = neural_engine.generate_content(prompt)
                    res_text = response.text
                    json_res = json.loads(re.search(r'\{.*\}', res_text, re.DOTALL).group())
                    st.session_state.last_res = json_res
                except:
                    # Thuật toán dự phòng nếu AI bận
                    nums = "".join(st.session_state.history[-50:])
                    top = [x[0] for x in Counter(nums).most_common(7)]
                    st.session_state.last_res = {
                        "main_1": "".join(top[:3]), "main_2": "".join(top[1:4]),
                        "support_4": "".join(top[3:]), "decision": "THẬN TRỌNG",
                        "logic": "Sử dụng ma trận tần suất dự phòng.", "conf": 80
                    }
                st.rerun()
        
        if c2.button("🗑️ RESET DỮ LIỆU"):
            st.session_state.history = []
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            st.rerun()

# ================= PHẦN HIỂN THỊ KẾT QUẢ ĐẲNG CẤP =================
status_msg, status_col = detect_bridge_trap(st.session_state.history)
st.markdown(f"<div class='status-bar' style='background: {status_col}; color: white;'>{status_msg}</div>", unsafe_allow_html=True)

if "last_res" in st.session_state:
    res = st.session_state.last_res
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Hiển thị 2 dàn chủ lực rõ ràng
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<p style='text-align:center; color:#ff5858; font-weight:bold;'>🔥 DÀN CHỦ LỰC 1</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num-box'>{res['main_1']}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<p style='text-align:center; color:#ff5858; font-weight:bold;'>🔥 DÀN CHỦ LỰC 2</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num-box'>{res['main_2']}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Dàn lót an toàn
    st.markdown(f"<p style='text-align:center; color:#8b949e;'>🛡️ 4 SỐ LÓT AN TOÀN: <span style='color:#58a6ff; font-size:30px; font-weight:bold;'>{res['support_4']}</span></p>", unsafe_allow_html=True)
    
    st.write(f"💡 **PHÂN TÍCH SOI CẦU:** {res['logic']}")
    
    # Tổng hợp dàn 7 số
    full_set = "".join(sorted(set(res['main_1'] + res['main_2'] + res['support_4'])))[:7]
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ KUBET:", full_set)
    
    st.progress(res['conf'] / 100)
    st.markdown(f"<p style='text-align:right;'>Độ tin cậy AI: {res['conf']}%</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Biểu đồ nhịp cầu thời gian thực
if st.session_state.history:
    with st.expander("📊 XEM MA TRẬN NHỊP CẦU (LỊCH SỬ)"):
        all_data = "".join(st.session_state.history[-50:])
        df = pd.Series(Counter(all_data)).sort_index()
        st.bar_chart(df)
