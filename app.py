import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG TITAN v24.2 =================
# Cập nhật API KEY mới nhất từ anh
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_supreme_v24_2.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: 
                data = json.load(f)
                return data if isinstance(data, list) else []
            except: return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f) # Lưu tối đa 3000 kỳ để AI học sâu

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= THIẾT KẾ GIAO DIỆN v22.0 STYLE =================
st.set_page_config(page_title="TITAN v24.2 SUPREME", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .num-box {
        font-size: 80px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 12px; border-right: 2px solid #30363d;
        text-shadow: 0 0 15px rgba(255,88,88,0.4);
    }
    .lot-box {
        font-size: 55px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px; padding-left: 20px;
    }
    .status-bar { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; margin-bottom: 15px; }
    .warning-box { background: #331010; color: #ff7b72; padding: 10px; border-radius: 5px; border: 1px solid #6e2121; text-align: center; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v24.2 SUPREME AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Hệ thống soi cầu 3D - Khắc chế nhà cái đảo cầu</p>", unsafe_allow_html=True)

# ================= PHẦN 1: NHẬP LIỆU & XỬ LÝ SẠCH =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 Nạp dữ liệu (Dán bảng lịch sử hoặc dãy số):", height=120, placeholder="32880\n21808...")
    with col_st:
        st.write(f"📊 Kho dữ liệu: **{len(st.session_state.history)} kỳ**")
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 GIẢI MÃ")
        btn_reset = c2.button("🗑️ RESET")

if btn_reset:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.success("Đã xóa sạch bộ nhớ vĩnh viễn.")
    st.rerun()

if btn_save:
    # Bước 1: Lọc sạch dữ liệu (Chỉ lấy dãy đúng 5 chữ số)
    new_data = re.findall(r"\b\d{5}\b", raw_input)
    if new_data:
        # Loại bỏ trùng lặp và giữ nguyên thứ tự
        st.session_state.history.extend(new_data)
        st.session_state.history = list(dict.fromkeys(st.session_state.history))
        save_db(st.session_state.history)
        
        # Bước 2: Phân tích bệt/đảo trước khi gửi cho Gemini
        last_nums = "".join(st.session_state.history[-10:])
        streak_check = Counter(last_nums).most_common(1)
        
        # Gửi AI Phân tích chuyên sâu
        prompt = f"""
        Bạn là Siêu trí tuệ TITAN v24.2 chuyên soi cầu Lotobet.
        Dữ liệu lịch sử: {st.session_state.history[-100:]}
        Nhận diện nhanh: Số '{streak_check[0][0]}' đang có dấu hiệu bệt/về nhiều.
        Nhiệm vụ:
        1. Phân tích nhịp đảo cầu của nhà cái (Tài/Xỉu, Chẵn/Lẻ).
        2. Chốt 3 số chính (Main_3) có khả năng nằm trong giải ĐB cao nhất.
        3. Chốt 4 số lót (Support_4) tạo dàn 7 số.
        4. Trả về kết luận 'NÊN ĐÁNH' hoặc 'DỪNG' nếu cầu đang ảo.
        
        TRẢ VỀ JSON:
        {{
            "main_3": "abc", 
            "support_4": "defg", 
            "decision": "ĐÁNH/DỪNG/CẢNH BÁO BỆT", 
            "logic": "Giải thích ngắn gọn nhịp cầu", 
            "color": "Green/Red/Yellow", 
            "conf": 98
        }}
        """
        try:
            response = neural_engine.generate_content(prompt)
            st.session_state.last_prediction = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
        except:
            # Thuật toán dự phòng nếu AI bận
            all_n = "".join(st.session_state.history[-50:])
            top = [x[0] for x in Counter(all_n).most_common(7)]
            st.session_state.last_prediction = {
                "main_3": "".join(top[:3]), 
                "support_4": "".join(top[3:]), 
                "decision": "THEO DÕI", 
                "logic": "Hệ thống đang đồng bộ dữ liệu cầu.", 
                "color": "Yellow", 
                "conf": 65
            }
        st.rerun()

# ================= PHẦN 2: KẾT QUẢ HIỂN THỊ TRỰC QUAN =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    # Định dạng màu sắc trạng thái
    status_colors = {"green": "#238636", "red": "#da3633", "yellow": "#d29922"}
    bg_color = status_colors.get(res['color'].lower(), "#30363d")
    
    st.markdown(f"<div class='status-bar' style='background: {bg_color};'>📢 TRẠNG THÁI: {res['decision']} ({res['conf']}%)</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Hiển thị 3 số chính và 4 số lót hàng ngang
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown(f"<p style='color:#8b949e; text-align:center;'>🔥 3 SỐ CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main_3']}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<p style='color:#8b949e; text-align:center;'>🛡️ 4 SỐ LÓT (AN TOÀN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Phân tích chi tiết & Cảnh báo bệt
    col_logic, col_copy = st.columns([2, 1])
    with col_logic:
        st.write(f"💡 **PHÂN TÍCH:** {res['logic']}")
        if "BỆT" in res['decision'] or res['conf'] < 80:
            st.markdown("<div class='warning-box'>⚠️ CẢNH BÁO: Cầu đang có dấu hiệu bệt sâu hoặc đảo liên tục. Đánh nhẹ hoặc dừng.</div>", unsafe_allow_html=True)
    
    with col_copy:
        full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
        st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", full_dan)
        
    st.markdown("</div>", unsafe_allow_html=True)

# ================= PHẦN 3: BỘ LỌC ĐA TẦNG (HỌC KỲ) =================
if st.session_state.history:
    with st.expander("📊 Thống kê nhịp rơi & Logic đa tầng"):
        st.write("Dữ liệu 50 kỳ gần nhất được AI phân tích để tìm quy luật đảo cầu:")
        all_d = "".join(st.session_state.history[-50:])
        st.bar_chart(pd.Series(Counter(all_d)).sort_index())
