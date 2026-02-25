import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

# ================= CONFIG & API =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_permanent_data.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= BẢO LƯU DỮ LIỆU VĨNH VIỄN =================
def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_data(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-2000:], f) # Lưu tối đa 2000 kỳ gần nhất

if "history" not in st.session_state:
    st.session_state.history = load_data()

# ================= TRÍ TUỆ NHẬN DIỆN CẦU (v24 Inside) =================
def get_bridge_status(data):
    if len(data) < 10: return "Cần thêm dữ liệu", "#888", 50
    last_5 = data[-5:]
    all_str = "".join(last_5)
    counts = Counter(all_str)
    
    # Kiểm tra bệt
    max_freq = counts.most_common(1)[0][1]
    if max_freq >= 7: 
        return "⚠️ CẦU BỆT - DỪNG CƯỢC", "#ff4b4b", 30
    
    # Kiểm tra đảo cầu (biến động lớn)
    sums = [sum(int(d) for d in s) for s in last_5]
    if np.std(sums) > 12:
        return "🟡 CẦU ĐẢO - ĐÁNH NHỎ", "#f2cc60", 60
        
    return "✅ CẦU ĐẸP - VÀO TIỀN", "#39d353", 95

# ================= GIAO DIỆN CHUẨN v22.0 =================
st.set_page_config(page_title="AI LOTOBET PRO v22.1", layout="wide")

st.markdown(f"""
    <style>
    .stApp {{ background-color: #f8f9fa; color: #1f1f1f; }}
    .stMetric {{ background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    .prediction-box {{ background-color: white; padding: 25px; border-radius: 15px; border-left: 8px solid #ff4b4b; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
    .num-large {{ font-size: 80px; font-weight: 900; color: #ff4b4b; text-align: center; letter-spacing: 15px; }}
    </style>
""", unsafe_allow_html=True)

st.title("🎯 AI LOTOBET 2-TINH / 3D PRO v22.1")

# Tab phân chia rõ ràng như anh thích
tab1, tab2 = st.tabs(["📊 Dự đoán & Thống kê", "📥 Nhập liệu hệ thống"])

with tab2:
    st.subheader("📥 Cập nhật dữ liệu sạch")
    st.info(f"Dữ liệu đã bảo lưu: {len(st.session_state.history)} kỳ")
    raw_input = st.text_area("Nhập 5 số viết liền (Mỗi kỳ 1 dòng):", height=200, placeholder="12345\n67890...")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Lưu & Đồng bộ dữ liệu"):
            new_data = re.findall(r"\d{5}", raw_input)
            if new_data:
                st.session_state.history.extend(new_data)
                # Loại bỏ trùng và giữ thứ tự
                st.session_state.history = list(dict.fromkeys(st.session_state.history))
                save_data(st.session_state.history)
                st.success(f"Đã lưu thành công {len(new_data)} kỳ mới!")
                st.rerun()
    with c2:
        if st.button("🗑️ Xóa toàn bộ bộ nhớ"):
            st.session_state.history = []
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            st.rerun()

with tab1:
    if len(st.session_state.history) < 15:
        st.warning("Vui lòng nhập tối thiểu 15 kỳ ở tab 'Nhập liệu' để AI bắt đầu soi cầu.")
    else:
        # 1. Trạng thái cầu (Bộ não v24)
        status_text, status_color, confidence = get_bridge_status(st.session_state.history)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng số kỳ", len(st.session_state.history))
        c2.markdown(f"<div style='text-align:center; padding:10px; border-radius:5px; background:{status_color}; color:white; font-weight:bold;'>{status_text}</div>", unsafe_allow_html=True)
        c3.metric("Độ tin cậy", f"{confidence}%")

        st.divider()

        # 2. Dự đoán chính
        if st.button("🔮 KÍCH HOẠT GEMINI & AI SOI CẦU"):
            with st.spinner("Đang giải mã nhịp cầu..."):
                prompt = f"""
                Hệ thống: Chuyên gia 3D Lotobet. 
                Dữ liệu: {st.session_state.history[-60:]}.
                Nhiệm vụ: 
                - Phân tích nhịp cầu bệt và đảo.
                - Chốt 3 số (Main_3) và 4 số lót (Support_4).
                Trả về JSON: {{"main_3": "ABC", "support_4": "DEFG", "logic": "Ngắn gọn", "confidence": 98}}
                """
                try:
                    response = neural_engine.generate_content(prompt)
                    st.session_state.last_res = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                except:
                    # Fallback nếu lỗi mạng
                    top = [x[0] for x in Counter("".join(st.session_state.history[-30:])).most_common(7)]
                    st.session_state.last_res = {"main_3": "".join(top[:3]), "support_4": "".join(top[3:]), "logic": "Thống kê tần suất kỳ gần.", "confidence": 70}
            st.rerun()

        if "last_res" in st.session_state:
            res = st.session_state.last_res
            st.markdown(f"<div class='prediction-box' style='border-left-color: {status_color};'>", unsafe_allow_html=True)
            st.write(f"🔍 **PHÂN TÍCH:** {res['logic']}")
            
            st.markdown(f"<div class='num-large' style='color:{status_color};'>{res['main_3']}</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#888;'>🎯 3 SỐ CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
            
            st.divider()
            st.write(f"🛡️ **DÀN LÓT (GIỮ VỐN):** {res['support_4']}")
            st.text_input("📋 SAO CHÉP DÀN 7 SỐ KUBET:", res['main_3'] + res['support_4'])
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Biểu đồ tần suất (Cho anh dễ nhìn như v22.0)
        st.subheader("📈 Tần suất số đơn (30 kỳ gần nhất)")
        all_nums = "".join(st.session_state.history[-30:])
        df_chart = pd.DataFrame(pd.Series(list(all_nums)).value_counts().sort_index(), columns=['Tần suất'])
        st.bar_chart(df_chart)

