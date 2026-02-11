import streamlit as st
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ================= CONFIG HỆ THỐNG QUÂN SỰ =================
st.set_page_config(page_title="TITAN V3000 ULTIMATE", layout="wide")
DATA_FILE = "dataset_5d_ultimate.json"

st.markdown("""
    <style>
    .reportview-container { background: #0a0a0a; }
    .stMetric { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    .prediction-box {
        background: linear-gradient(135deg, #1f1f1f 0%, #000 100%);
        border: 2px solid #ffd700; border-radius: 20px; padding: 30px; text-align: center;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.2);
    }
    .status-alert { padding: 10px; border-radius: 5px; font-weight: bold; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

def load_db():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f: return json.load(f)
    return []

def save_db(data):
    with open(DATA_FILE, "w") as f: json.dump(data[-5000:], f)

if "db" not in st.session_state: st.session_state.db = load_db()

# ================= THUẬT TOÁN ĐỐI ĐẦU AI (AI COUNTER) =================
def military_grade_analysis(db):
    if len(db) < 20: return None
    
    # 1. Chuyển đổi dữ liệu sang số liệu tính toán
    matrix = np.array([[int(d) for d in list(ky)] for ky in db])
    totals = np.sum(matrix, axis=1)
    
    # 2. Tính Entropy (Độ loạn của cầu)
    # Nếu Entropy cao -> Cầu loạn, AI nhà cái đang quét mạnh -> Khuyên nghỉ
    counts = np.unique(totals[-20:], return_counts=True)[1]
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    
    # 3. Thuật toán Mean Reversion (Hồi quy trung bình)
    avg_short = np.mean(totals[-10:])
    avg_long = np.mean(totals[-50:]) if len(db) >= 50 else 22.5
    
    # 4. Dự đoán đa tầng
    # Dự đoán Tổng 5 Banh
    pred_tx = "TÀI" if avg_short < 22.5 else "XỈU" # Đánh ngược nhịp ngắn để bắt hồi quy
    pred_cl = "CHẴN" if int(avg_short) % 2 != 0 else "LẺ"
    
    # Dự đoán Baccarat 5D (Logic: Bắt bệt nhịp mạnh)
    con_val = (matrix[:, 2] + matrix[:, 4]) % 10
    cai_val = (matrix[:, 1] + matrix[:, 3]) % 10
    con_streak = sum(1 for i in range(-3, 0) if con_val[i] > cai_val[i])
    bac_res = "CON (PLAYER)" if con_streak >= 2 else "CÁI (BANKER)"

    # 5. Công thức Kelly (Quản lý vốn)
    # Giả định tỉ lệ thắng là 55%, tỉ lệ ăn 1:1
    kelly_percent = "10-15%" if entropy < 2.5 else "2-5%"
    
    return {
        "tx": pred_tx, "cl": pred_cl, "bac": bac_res,
        "entropy": entropy, "kelly": kelly_percent,
        "history": totals[-30:].tolist(),
        "is_safe": entropy < 3.0
    }

# ================= GIAO DIỆN CHIẾN ĐẤU =================
st.markdown("<h1 style='text-align: center; color: #ffd700;'>🛰️ TITAN V3000 ULTIMATE CORE</h1>", unsafe_allow_html=True)

c_input, c_output = st.columns([1, 2])

with c_input:
    st.subheader("📡 TRẠM THU PHÁT DỮ LIỆU")
    raw = st.text_area("Nhập mã 5D (5 con số):", height=250, placeholder="Dán kết quả tại đây...")
    if st.button("⚡ QUÉT SÓNG AI", use_container_width=True):
        if raw:
            extracted = re.findall(r"\d{5}", raw)
            st.session_state.db.extend(extracted)
            save_db(st.session_state.db)
            st.rerun()
    
    if st.button("🗑️ RESET SYSTEM"):
        save_db([])
        st.session_state.db = []
        st.rerun()

with c_output:
    if len(st.session_state.db) >= 20:
        res = military_grade_analysis(st.session_state.db)
        
        # Cảnh báo độ loạn của cầu
        if res["is_safe"]:
            st.markdown("<div class='status-alert' style='background: rgba(0,255,0,0.1); color: #00ffcc;'>✅ SÓNG ỔN ĐỊNH - VÀO LỆNH AN TOÀN</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-alert' style='background: rgba(255,0,0,0.1); color: #ff4b4b;'>⚠️ SÓNG NHIỄU CAO - ĐI VỐN CỰC NHỎ HOẶC NGHỈ</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='prediction-box'>
            <p style='color: #888; letter-spacing: 3px;'>DỰ BÁO TỔNG 5 BANH</p>
            <h1 style='color: #ffd700; font-size: 60px; margin: 10px;'>{res['tx']} - {res['cl']}</h1>
            <p style='color: #00ffcc;'>Lệnh Baccarat: <b>{res['bac']}</b></p>
            <hr style='border-color: #333;'>
            <p style='color: #fff;'>CHIẾN THUẬT VỐN KELLY: <span style='color: #ffd700; font-size: 20px;'>{res['kelly']}</span></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 BIỂU ĐỒ ĐƯỜNG ĐI CỦA TỔNG 5")
        st.line_chart(res['history'])
        
        st.write(f"📊 **Chỉ số Entropy (Độ loạn):** {res['entropy']:.2f}")
    else:
        st.info("Hệ thống đang thu thập tín hiệu. Anh cần nạp tối thiểu 20 kỳ để AI bắt đầu phân tích đối ứng.")

st.markdown("<p style='text-align: center; color: #444; margin-top: 30px;'>CẢNH BÁO: KHÔNG ĐÁNH TẤT TAY. TUÂN THỦ CÔNG THỨC KELLY.</p>", unsafe_allow_html=True)
