import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
import requests
from io import StringIO
from datetime import datetime

# ================= CẤU HÌNH HỆ THỐNG =================
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
genai.configure(api_key=GEMINI_API_KEY) 

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA/export?format=csv"

st.set_page_config(page_title="TITAN MASTER HUB V6.2", layout="wide") 

# Giao diện VIP
st.markdown("""
    <style>
    .stApp { background-color: #0a0a0a; color: #ffffff; }
    .metric-card {
        background: #161b22;
        padding: 15px;
        border-radius: 10px;
        border-top: 4px solid #00ff00;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(90deg, #1e1e1e 0%, #000 100%);
        padding: 25px;
        border: 2px solid #00ff00;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0,255,0,0.3);
    }
    .stButton>button { background: #00ff00 !important; color: #000 !important; font-weight: 900; height: 50px; }
    </style>
    """, unsafe_allow_html=True)

# Hàm load dữ liệu thông minh
def load_data_v6():
    try:
        response = requests.get(SHEET_CSV_URL, timeout=15)
        df = pd.read_csv(StringIO(response.text), header=None)
        all_data = []
        for col in df.columns:
            valid = df[col].astype(str).str.strip().str.extract(r'(\d{5})')[0].dropna()
            if not valid.empty: all_data.extend(valid.tolist())
        
        if not all_data: return pd.DataFrame()
        return pd.DataFrame(all_data, columns=['numbers'])
    except:
        return pd.DataFrame()

# ================= THUẬT TOÁN TITAN UPGRADE =================
class TitanIntelligence:
    def __init__(self, data):
        self.df = data
        self.matrix = np.array([[int(d) for d in str(n)] for n in data['numbers'].tail(200)])

    def analyze_advanced(self):
        # 1. Tính toán xác suất xuất hiện tiếp theo (Next-Digit Probability)
        last_digit = self.matrix[-1, -1] # Lấy số cuối của kỳ vừa rồi
        next_candidates = {}
        for i in range(len(self.matrix)-1):
            if self.matrix[i, -1] == last_digit:
                followed_by = self.matrix[i+1, -1]
                next_candidates[followed_by] = next_candidates.get(followed_by, 0) + 1
        
        # 2. Kiểm tra trạng thái "Chập"
        last_5_kỳ = self.df['numbers'].tail(5).tolist()
        is_shuffled = any(len(set(str(n))) < 4 for n in last_5_kỳ) # Nếu ít hơn 4 số khác nhau là chập

        # 3. Thống kê tần suất & Gan
        stats = {}
        for i in range(10):
            freq = np.sum(self.matrix[-25:] == i)
            last_seen = np.where(np.any(self.matrix == i, axis=1))[0]
            gap = (len(self.matrix) - 1 - last_seen[-1]) if len(last_seen) > 0 else 99
            stats[i] = {"freq": int(freq), "gap": int(gap)}
            
        return stats, next_candidates, is_shuffled

# ================= GIAO DIỆN CHÍNH =================
def main():
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>⚡ TITAN AI MASTER HUB V6.2 ⚡</h1>", unsafe_allow_html=True)
    
    # Nút bấm chính
    if st.button("🚀 KÍCH HOẠT QUÉT DỮ LIỆU REAL-TIME"):
        st.cache_data.clear()
        st.rerun()

    df = load_data_v6()

    if not df.empty:
        intel = TitanIntelligence(df)
        stats, next_prob, is_shuffled = intel.analyze_advanced()
        
        # 1. Dashboard chỉ số nhanh
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card">Dữ liệu<br><b>{len(df)} Kỳ</b></div>', unsafe_allow_html=True)
        with c2: 
            status = "⚠️ NHIỄU" if is_shuffled else "✅ ỔN ĐỊNH"
            st.markdown(f'<div class="metric-card">Thị trường<br><b>{status}</b></div>', unsafe_allow_html=True)
        with c3:
            best_num = max(stats, key=lambda x: stats[x]['freq'])
            st.markdown(f'<div class="metric-card">Số Nóng<br><b>{best_num}</b></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card">Cập nhật<br><b>{datetime.now().strftime("%H:%M")}</b></div>', unsafe_allow_html=True)

        # 2. AI Chốt số & Nhận định
        st.divider()
        st.subheader("🤖 TRÍ TUỆ NHÂN TẠO GEMINI CHỐT SỐ")
        
        # Gửi dữ liệu chi tiết cho Gemini
        prompt_data = {
            "history": df['numbers'].tail(10).tolist(),
            "stats": stats,
            "patterns": next_prob,
            "market_status": "Nhiễu" if is_shuffled else "Sạch"
        }

        with st.spinner('Gemini đang giải mã nhịp cầu...'):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"Bạn là TITAN AI. Dựa vào dữ liệu này: {prompt_data}. Hãy chốt 1 cặp số duy nhất (2D) theo đặc tả: Loại bỏ số chập, ưu tiên cầu bệt ổn định. Trả lời cực ngắn gọn: 'NHẬN ĐỊNH: ...' và 'CHỐT SỐ: XX-YY'.")
                prediction = response.text
            except:
                prediction = "Hệ thống đang bận. Gợi ý kỹ thuật: 05 - 50"

        st.markdown(f'<div class="prediction-box">{prediction}</div>', unsafe_allow_html=True)

        # 3. Phân tích chi tiết bằng biểu đồ
        st.divider()
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("📊 Phân tích chu kỳ 25 kỳ")
            chart_df = pd.DataFrame([{"Số": i, "Tần suất": stats[i]['freq'], "Độ Gan": stats[i]['gap']} for i in range(10)])
            fig = px.bar(chart_df, x="Số", y="Tần suất", color="Độ Gan", title="Mối quan hệ Tần suất & Độ Gan",
                         color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("💰 QUẢN LÝ VỐN (Dự kiến)")
            vốn = st.number_input("Nhập vốn (VNĐ):", value=1000000, step=100000)
            target = vốn * 0.2
            st.write(f"🎯 Mục tiêu lãi 20%: **{target:,.0f} VNĐ**")
            st.write(f"🛑 Cắt lỗ (15%): **{vốn * 0.15:,.0f} VNĐ**")
            
            st.divider()
            st.subheader("📜 Nhật ký số mới nhất")
            st.dataframe(df.tail(15), use_container_width=True)

    else:
        st.error("❌ Không lấy được dữ liệu. Anh Đạt hãy kiểm tra lại file Sheets (Cột A phải có số)!")

if __name__ == "__main__":
    main()
