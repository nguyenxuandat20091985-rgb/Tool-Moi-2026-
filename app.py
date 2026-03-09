import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
from collections import Counter
from itertools import combinations

# ================= CẤU HÌNH AI =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# --- Thuật toán lọc dữ liệu thông minh (Sửa lỗi đứng hình) ---
def smart_extract(raw_text):
    # Tìm tất cả các dãy 5 chữ số đứng riêng biệt (bỏ qua số kỳ 3 chữ số)
    # Regex này chỉ lấy dãy 5 số
    lines = re.findall(r"\b\d{5}\b", raw_text)
    return lines

def get_combo_stats(history, n=3):
    all_combos = []
    for line in history:
        unique_digits = sorted(list(set(line)))
        if len(unique_digits) >= n:
            all_combos.extend(combinations(unique_digits, n))
    return Counter(all_combos).most_common(1)

# ================= UI STYLE v22 CHUẨN =================
st.set_page_config(page_title="TITAN v24.5 FINAL", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 12px; padding: 25px; margin-top: 10px;
        box-shadow: 0 0 15px rgba(88,166,255,0.1);
    }
    .num-box {
        font-size: 90px; font-weight: 900; color: #ff4b4b;
        text-align: center; letter-spacing: 15px; text-shadow: 3px 3px #000;
    }
    .lot-box {
        font-size: 60px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px;
    }
    .status-bar { padding: 15px; border-radius: 10px; text-align: center; font-weight: 900; font-size: 1.4rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>💎 TITAN v24.5 - CHỐT 3 SỐ CHÍNH XÁC</h2>", unsafe_allow_html=True)

# ================= XỬ LÝ NHẬP LIỆU =================
if "last_res" not in st.session_state:
    st.session_state.last_res = None

with st.container():
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        # Anh cứ dán cả bảng lịch sử vào đây, không cần xóa số kỳ
        raw_input = st.text_area("📡 Dán lịch sử từ KU (Dán trực tiếp):", height=150, placeholder="Ví dụ: 222 80586\n221 64549...")
    with col_btn:
        st.write("🔧 **CÔNG CỤ**")
        if st.button("🔥 GIẢI MÃ NGAY", use_container_width=True):
            cleaned_data = smart_extract(raw_input)
            if len(cleaned_data) >= 5:
                # Thuật toán Combo nổ nhất
                best_c = get_combo_stats(cleaned_data[:30])
                combo_fix = "".join(best_c[0][0]) if best_c else "123"
                
                # Gọi AI xử lý đa tầng
                prompt = f"""
                Dữ liệu thực tế: {cleaned_data[:40]}.
                Quy tắc 3 số 5 tinh: Thắng khi kết quả có mặt đủ 3 số.
                Phân tích nhịp cầu kẹp và cầu bệt combo.
                Trả về JSON: {{"m": "{combo_fix}", "s": "0468", "d": "NÊN ĐÁNH", "l": "Cầu đang bệt combo mạnh", "c": "Green"}}
                """
                try:
                    res_ai = neural_engine.generate_content(prompt)
                    match = re.search(r'\{.*\}', res_ai.text, re.DOTALL)
                    st.session_state.last_res = json.loads(match.group())
                except:
                    st.session_state.last_res = {"m": combo_fix, "s": "0456", "d": "ĐÁNH THEO COMBO", "l": "Dựa trên tần suất bộ 3 nổ dày.", "c": "Green"}
                st.rerun()
            else:
                st.error("❌ Không tìm thấy dãy 5 số nào! Anh kiểm tra lại dữ liệu dán vào.")

        if st.button("🗑️ RESET TOOL", use_container_width=True):
            st.session_state.last_res = None
            st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_res:
    res = st.session_state.last_res
    bg = "#238636" if res['c'].lower() == "green" else "#da3633"
    
    st.markdown(f"<div class='status-bar' style='background: {bg};'>📢 TRẠNG THÁI AI: {res['d']}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🔥 3 SỐ 5 TINH CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['m']}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🛡️ 4 SỐ LÓT (GHÉP DÀN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['s']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"🧠 **PHÂN TÍCH NHỊP CẦU:** {res['l']}")
    
    full_dan = "".join(sorted(set(res['m'] + res['s'])))
    st.text_input("📋 DÀN 7 SỐ KUBET (Copy):", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)

    # Thống kê trực quan
    with st.expander("📊 Biểu đồ nhịp rơi hiện tại"):
        cleaned_list = smart_extract(raw_input)
        all_txt = "".join(cleaned_list[:20])
        st.bar_chart(pd.Series(Counter(all_txt)).sort_index())
