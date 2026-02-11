import streamlit as st
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ================= CONFIG HỆ THỐNG QUÂN SỰ =================
st.set_page_config(page_title="TITAN V3000 ULTIMATE", layout="wide", initial_sidebar_state="collapsed")

# CSS Cao cấp - Biến giao diện thành trạm chỉ huy
st.markdown("""
    <style>
    .main { background-color: #05070a; }
    .stMetric { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }
    .prediction-card {
        background: linear-gradient(135deg, #1a1c23 0%, #000 100%);
        border: 2px solid #ffd700; border-radius: 25px; padding: 40px; text-align: center;
        box-shadow: 0 0 40px rgba(255, 215, 0, 0.15);
    }
    .number-main {
        font-size: 90px; font-weight: 900; color: #ffd700;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.6); margin: 15px 0;
    }
    .status-badge {
        padding: 8px 20px; border-radius: 50px; font-size: 16px; font-weight: bold;
        text-transform: uppercase; margin-bottom: 25px; display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "titan_v3000_db.json"

def get_db():
    if Path(DB_FILE).exists():
        with open(DB_FILE, "r") as f: return json.load(f)
    return []

def save_db(data):
    # AI nhà cái thường đổi thuật toán sau 3000-5000 kỳ, nên ta chỉ lưu đủ dùng
    with open(DB_FILE, "w") as f: json.dump(data[-5000:], f)

if "db" not in st.session_state: st.session_state.db = get_db()

# ================= THUẬT TOÁN ĐỐI ĐẦU AI (AI-COUNTER ENGINE) =================
def military_analysis(db):
    if len(db) < 25: return None
    
    # Chuyển đổi dữ liệu sang ma trận số
    matrix = np.array([[int(d) for d in list(ky)] for ky in db])
    totals = np.sum(matrix, axis=1)
    
    # 1. Đo độ hỗn loạn (Entropy) - Mắt thần né cầu cháy
    counts = np.unique(totals[-25:], return_counts=True)[1]
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    
    # 2. Logic Tổng 5 Banh (Hồi quy trung bình)
    avg_short = np.mean(totals[-12:])
    pred_t5_tx = "TÀI" if avg_short < 22.5 else "XỈU"
    pred_t5_cl = "CHẴN" if int(avg_short) % 2 != 0 else "LẺ"

    # 3. Logic Baccarat 5D (Trăm+Đơn vị vs Ngàn+Chục)
    con_scores = (matrix[:, 2] + matrix[:, 4]) % 10
    cai_scores = (matrix[:, 1] + matrix[:, 3]) % 10
    con_wins = sum(1 for i in range(-5, 0) if con_scores[i] > cai_scores[i])
    pred_bac = "CON (PLAYER)" if con_wins >= 3 else "CÁI (BANKER)"

    # 4. Quản lý vốn Kelly - Đối đầu sự tham lam
    kelly = "2-5% VỐN" if entropy > 2.8 else "10-15% VỐN"
    
    return {
        "t5": f"{pred_t5_tx} - {pred_t5_cl}",
        "bac": pred_bac,
        "entropy": entropy,
        "kelly": kelly,
        "is_safe": entropy < 3.0,
        "history": totals[-30:].tolist()
    }

# ================= GIAO DIỆN CHIẾN ĐẤU =================
st.markdown("<h1 style='text-align: center; color: #ffd700;'>🛰️ TITAN V3000 ULTIMATE CORE</h1>", unsafe_allow_html=True)

col_in, col_out = st.columns([1, 2])

with col_in:
    st.markdown("<div class='stMetric'>", unsafe_allow_html=True)
    raw = st.text_area("📡 TRẠM NHẬN DỮ LIỆU", height=250, placeholder="Dán dãy 5 số mở thưởng...")
    if st.button("⚡ QUÉT SÓNG NHÀ CÁI", use_container_width=True):
        if raw:
            extracted = re.findall(r"\d{5}", raw)
            st.session_state.db.extend(extracted)
            save_db(st.session_state.db)
            st.rerun()
    if st.button("🧹 RESET"):
        save_db([])
        st.session_state.db = []
        st.rerun()
    st.write(f"Dữ liệu tích lũy: **{len(st.session_state.db)} kỳ**")
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    if len(st.session_state.db) >= 25:
        res = military_analysis(st.session_state.db)
        
        # Hiển thị trạng thái sóng
        if res["is_safe"]:
            st.markdown("<div class='status-badge' style='background: rgba(0,255,100,0.2); color: #00ffcc;'>✅ Sóng ổn định - Vào lệnh</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-badge' style='background: rgba(255,50,50,0.2); color: #ff4b4b;'>⚠️ Sóng nhiễu cao - Cẩn thận AI</div>", unsafe_allow_html=True)

        # Dashboard Dự đoán chính
        st.markdown(f"""
            <div class='prediction-card'>
                <p style='color: #888; letter-spacing: 5px;'>DỰ BÁO TỔNG 5 BANH</p>
                <div class='number-main'>{res['t5']}</div>
                <p style='color: #00ffcc; font-size: 20px;'>Baccarat: <b>{res['bac']}</b></p>
                <hr style='border-color: #333;'>
                <p style='color: #888;'>CHIẾN THUẬT VỐN: <span style='color: #ffd700;'>{res['kelly']}</span></p>
            </div>
        """, unsafe_allow_html=True)
        
        # Biểu đồ nhịp cầu thực tế
        st.subheader("📊 Sóng nhịp Tổng 5 (30 kỳ gần nhất)")
        st.line_chart(res['history'])
        st.write(f"Chỉ số hỗn loạn Entropy: `{res['entropy']:.2f}` (Dưới 3.0 là an toàn)")
    else:
        st.warning("Hệ thống cần tối thiểu 25 kỳ để bắt đầu tính toán độ lệch chuẩn và Entropy đối ứng với AI nhà cái.")

st.markdown("<p style='text-align: center; color: #444; margin-top: 50px;'>HỆ THỐNG ĐÃ ĐƯỢC VŨ TRANG TỐT NHẤT - V3000 FINAL</p>", unsafe_allow_html=True)
