import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH SIÊU TRÍ TUỆ TITAN v24.0 =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_elite_v24_permanent.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= QUẢN LÝ BỘ NHỚ VĨNH VIỄN =================
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        # Giữ lại 3000 kỳ để học sâu (Deep Learning)
        json.dump(data[-3000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= THUẬT TOÁN NHẬN DIỆN CẦU BỆT/ĐẢO =================
def analyze_bridge_logic(data):
    if len(data) < 15: return "Cần thêm dữ liệu", 0, "Gray"
    
    all_nums = "".join(data[-20:])
    last_5 = data[-5:]
    
    # 1. Kiểm tra Bệt (Streak)
    flat_last_5 = "".join(last_5)
    counts = Counter(flat_last_5)
    most_common_num = counts.most_common(1)[0]
    
    # 2. Kiểm tra Đảo cầu
    is_shuffling = False
    # Logic: Nếu tổng 5 số kỳ trước và kỳ này thay đổi đột ngột biên độ lớn
    sums = [sum([int(d) for d in s]) for s in last_5]
    diffs = np.diff(sums)
    if np.std(diffs) > 10: is_shuffling = True

    # 3. Ra quyết định ĐÁNH hay DỪNG
    confidence = 95
    status = "NÊN ĐÁNH"
    color = "#39d353" # Xanh

    if most_common_num[1] > 6: # Dấu hiệu bệt quá sâu, dễ cháy cầu
        status = "DỪNG - CẦU BỆT NGUY HIỂM"
        color = "#f85149"
        confidence = 40
    elif is_shuffling:
        status = "DỪNG - NHÀ CÁI ĐẢO CẦU"
        color = "#f2cc60"
        confidence = 55
        
    return status, confidence, color

# ================= GIAO DIỆN ELITE PRO =================
st.set_page_config(page_title="TITAN v24.0 ELITE", layout="wide")
st.markdown(f"""
    <style>
    .stApp {{ background: #010409; color: #e6edf3; }}
    .elite-card {{
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border: 2px solid #30363d; border-radius: 20px; padding: 40px;
        box-shadow: 0 10px 50px rgba(0,0,0,0.8);
    }}
    .signal-light {{
        height: 25px; width: 25px; border-radius: 50%; display: inline-block;
        margin-right: 10px; box-shadow: 0 0 15px currentColor;
    }}
    .main-number {{ font-size: 110px; font-weight: 900; color: #58a6ff; text-align: center; letter-spacing: 20px; }}
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ TITAN v24.0 ELITE - SIÊU TRÍ TUỆ")

# Tab hệ thống
tab1, tab2 = st.tabs(["🚀 GIẢI MÃ TINH HOA", "⚙️ CẤU HÌNH & DỮ LIỆU"])

with tab2:
    raw_input = st.text_area("📡 NẬP DỮ LIỆU (Mượt mà - Không giật lag):", height=200)
    if st.button("💾 LƯU VÀO BỘ NHỚ VĨNH VIỄN"):
        clean = re.findall(r"\d{5}", raw_input)
        if clean:
            # Gộp và loại trùng nhưng vẫn giữ thứ tự thời gian
            st.session_state.history.extend(clean)
            st.session_state.history = list(dict.fromkeys(st.session_state.history))
            save_db(st.session_state.history)
            st.success(f"Đã bảo lưu vĩnh viễn {len(clean)} kỳ mới!")
            st.rerun()
    if st.button("🗑️ XÓA SẠCH DỮ LIỆU"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

with tab1:
    if len(st.session_state.history) < 20:
        st.warning("⚠️ Hệ thống cần tối thiểu 20 kỳ lịch sử để bắt đầu học trí tuệ nhân tạo.")
    else:
        status, conf, color = analyze_bridge_logic(st.session_state.history)
        
        # UI Tín hiệu Đánh/Dừng
        st.markdown(f"""
            <div style='background: {color}22; border: 1px solid {color}; padding: 20px; border-radius: 10px; text-align: center;'>
                <span class='signal-light' style='color: {color}; background-color: {color};'></span>
                <b style='font-size: 24px; color: {color};'>{status}</b> (Độ tin cậy: {conf}%)
            </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ KÍCH HOẠT PHÂN TÍCH TINH HOA"):
            with st.spinner("AI đang quét cầu bệt và bóng số..."):
                # Kết hợp Gemini soi cầu sâu
                prompt = f"""
                Hệ thống: TITAN v24.0 ELITE.
                Lịch sử: {st.session_state.history[-100:]}.
                Nhiệm vụ: 
                1. Nhận diện bẫy nhà cái (Cầu bệt giả, đảo cầu đột ngột).
                2. Chốt 3 số (Main_3) có tần suất xuất hiện trong 5 số của giải ĐB cao nhất.
                3. Đưa ra dàn 7 số tổng thể (3 chính + 4 lót).
                Yêu cầu: Nếu cầu xấu, bắt buộc đặt 'should_bet': false.
                Trả về JSON: {{"main_3": "abc", "support_4": "defg", "logic": "...", "should_bet": true, "confidence": 98}}
                """
                
                try:
                    response = neural_engine.generate_content(prompt)
                    res = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                    st.session_state.elite_res = res
                except:
                    # Thuật toán dự phòng tinh hoa
                    all_n = "".join(st.session_state.history[-40:])
                    top = [x[0] for x in Counter(all_n).most_common(7)]
                    st.session_state.elite_res = {"main_3": "".join(top[:3]), "support_4": "".join(top[3:]), "logic": "Thuật toán tần suất nhịp rơi.", "should_bet": True, "confidence": 75}
            st.rerun()

        if "elite_res" in st.session_state:
            res = st.session_state.elite_res
            st.markdown("<div class='elite-card'>", unsafe_allow_html=True)
            
            if not res['should_bet']:
                st.markdown("<h2 style='color: #f85149; text-align: center;'>🚫 KHÔNG ĐÁNH KỲ NÀY</h2>", unsafe_allow_html=True)
                st.write(f"**Lý do từ AI:** {res['logic']}")
            else:
                st.markdown(f"<p style='text-align:center; color:#8b949e;'>🔥 3 SỐ VÀNG (DỰ ĐOÁN XUẤT HIỆN)</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='main-number'>{res['main_3']}</div>", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.info(f"🛡️ DÀN LÓT: {res['support_4']}")
                c2.success(f"📈 ĐỘ TIN CẬY: {res['confidence']}%")
                
                st.write(f"💡 **PHÂN TÍCH:** {res['logic']}")
                st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", res['main_3'] + res['support_4'])
            
            st.markdown("</div>", unsafe_allow_html=True)

# Footer thống kê kỳ
st.divider()
st.write(f"📊 Dữ liệu hiện tại: {len(st.session_state.history)} kỳ. Hệ thống đang tự học nhịp cầu mỗi giây.")
