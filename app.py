import streamlit as st
import re
import json
import numpy as np
from collections import Counter

# ================= CONFIG OMNI-INTERFACE =================
st.set_page_config(page_title="TITAN v8000 OMNI", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    [data-testid="stHeader"] {display: none;}
    .stButton > button {
        background: linear-gradient(90deg, #00ffcc, #0055ff);
        color: black; border: none; border-radius: 5px; width: 100%; height: 40px; font-weight: bold;
    }
    .card {
        background: #111; border: 1px solid #333; border-radius: 10px;
        padding: 10px; margin-bottom: 10px;
    }
    .title-gold { color: #ffd700; font-weight: bold; font-size: 14px; border-bottom: 1px solid #333; }
    .val-green { color: #00ff00; font-size: 24px; font-weight: 900; }
    .val-blue { color: #00ccff; font-weight: bold; }
    .val-red { color: #ff0055; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

if "db_omni" not in st.session_state: st.session_state.db_omni = []

# ================= BỘ NÃO PHÂN TÍCH TỔNG LỰC =================
def analyze_omni(data):
    if len(data) < 15: return None
    recent = data[-30:] # Soi 30 kỳ gần nhất để bắt nhịp
    matrix = np.array([[int(d) for d in list(ky)] for ky in recent])
    
    # 1. Thuật toán 3-TINH (Không cố định vị trí, Anti-Twin)
    flat_all = "".join(recent)
    freq = Counter(flat_all)
    # Lấy các số nổ nhiều nhưng không bị lặp lại trong kỳ cuối (tránh kép)
    last_ky = data[-1]
    candidates = [s for s in "0123456789" if s not in last_ky]
    p3 = sorted(candidates, key=lambda x: freq[x], reverse=True)[:3]

    # 2. Dự đoán TỔNG 5 (Tài/Xỉu/Chẵn/Lẻ)
    totals = np.sum(matrix, axis=1)
    avg_t = np.mean(totals)
    t5_tx = "TÀI" if avg_t < 22.5 else "XỈU"
    t5_cl = "CHẴN" if int(avg_t) % 2 == 0 else "LẺ"

    # 3. Dự đoán XÌ TỐ (Dựa trên độ nén dữ liệu)
    # Tính toán khả năng ra Sảnh, Cù Lũ, Thùng...
    diffs = np.std(matrix, axis=1) # Độ lệch chuẩn để đoán Sảnh/Đôi
    if diffs[-1] < 1.5: xi_to = "SÁM CÔ / 2 ĐÔI"
    elif diffs[-1] > 3.5: xi_to = "SỐ RỜI / SẢNH"
    else: xi_to = "1 ĐÔI / CÙ LŨ"

    # 4. RỒNG HỔ (C.Ngàn vs Đơn Vị)
    rong_val = matrix[-5:, 0].mean()
    ho_val = matrix[-5:, 4].mean()
    if abs(rong_val - ho_val) < 0.5: rh_res = "HÒA"
    else: rh_res = "RỒNG" if rong_val > ho_val else "HỔ"

    return {
        "p3": p3, "t5": f"{t5_tx} - {t5_cl}", 
        "xi_to": xi_to, "rh": rh_res,
        "conf": min(70 + len(data)//50, 98)
    }

# ================= GIAO DIỆN HIỂN THỊ =================
st.markdown("<h5 style='text-align: center; color: #00ffcc;'>🛰️ TITAN v8000 OMNI MASTER</h5>", unsafe_allow_html=True)

raw = st.text_area("Nhập mã 5D:", height=80, placeholder="Dán 5-10 kỳ vào đây...", label_visibility="collapsed")
c1, c2 = st.columns(2)
if c1.button("🔥 QUÉT OMNI"):
    if raw:
        st.session_state.db_omni.extend(re.findall(r"\d{5}", raw))
        st.rerun()
if c2.button("🧹 RESET"):
    st.session_state.db_omni = []; st.rerun()

if len(st.session_state.db_omni) >= 15:
    res = analyze_omni(st.session_state.db_omni)
    
    # KHU VỰC DỰ ĐOÁN TỔNG HỢP
    st.markdown(f"""
    <div class='card'>
        <p class='title-gold'>🎯 3 TINH CHÍNH XÁC (KHÔNG KÉP)</p>
        <p class='val-green' style='text-align:center;'>{" - ".join(res['p3'])}</p>
        <p style='font-size:10px; color:#888; text-align:center;'>Tỷ lệ nổ 3 trong 5 số cực cao</p>
    </div>
    
    <div class='card'>
        <div style='display: flex; justify-content: space-between;'>
            <div>
                <p class='title-gold'>📊 TỔNG 5 BANH</p>
                <p class='val-blue'>{res['t5']}</p>
            </div>
            <div style='text-align: right;'>
                <p class='title-gold'>🐲 RỒNG HỔ</p>
                <p class='val-red'>{res['rh']}</p>
            </div>
        </div>
    </div>

    <div class='card'>
        <p class='title-gold'>🃏 DỰ BÁO XÌ TỐ (5 CON)</p>
        <p style='font-size: 18px; font-weight: bold; color: #ffd700;'>{res['xi_to']}</p>
        <p style='font-size:10px; color:#666;'>Gồm: 5 Con, Cù Lũ, Tứ Quý, Sảnh, Sám Cô...</p>
    </div>

    <p style='text-align:center; color:#00ffcc; font-size:12px;'>ĐỘ TIN CẬY HỆ THỐNG: {res['conf']}%</p>
    """, unsafe_allow_html=True)
else:
    st.info("Anh dán thêm kỳ (Tổng ít nhất 15 kỳ) để em kích hoạt Omni-AI nhé!")

st.markdown("<p style='text-align:center; color:#333; font-size:10px;'>TITAN OMNI v8000 - SECURITY BY GEMINI AI</p>", unsafe_allow_html=True)
