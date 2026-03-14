import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import StringIO
from datetime import datetime

# ================= CẤU HÌNH HỆ THỐNG =================
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
genai.configure(api_key=GEMINI_API_KEY) 

# Link CSV từ Google Sheets của anh Đạt
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA/export?format=csv"

st.set_page_config(page_title="TITAN MASTER HUB V7", layout="wide") 

# Giao diện Dark VIP cho điện thoại
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #ffffff; }
    .prediction-card { background: #111; padding: 20px; border-radius: 15px; border-left: 8px solid #00ff00; margin-bottom: 20px; }
    .stButton>button { background: linear-gradient(45deg, #00ff00, #008000) !important; color: black !important; font-weight: bold; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- HÀM LẤY DỮ LIỆU SIÊU CẤP ---
def load_data_clean():
    try:
        response = requests.get(SHEET_CSV_URL, timeout=15)
        # Đọc không tiêu đề để không sót dòng đầu tiên (10004)
        df = pd.read_csv(StringIO(response.text), header=None)
        
        all_numbers = []
        for col in df.columns:
            # Chỉ lấy những ô có đúng 5 chữ số
            vals = df[col].astype(str).str.strip().str.extract(r'^(\d{5})$')[0].dropna()
            all_numbers.extend(vals.tolist())
        
        if not all_numbers: return pd.DataFrame()
        return pd.DataFrame(all_numbers, columns=['numbers'])
    except Exception as e:
        st.error(f"Lỗi kết nối: {e}")
        return pd.DataFrame()

# --- THUẬT TOÁN PHÂN TÍCH ---
class TitanCore:
    def __init__(self, df):
        self.df = df
        self.matrix = np.array([[int(d) for d in str(n)] for n in df['numbers'].tail(200)])

    def get_summary(self):
        last_num = self.df['numbers'].iloc[-1]
        freq_map = {}
        for i in range(10):
            freq_map[i] = np.sum(self.matrix[-30:] == i)
        return last_num, freq_map

# --- GIAO DIỆN CHÍNH ---
def main():
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>⚡ TITAN MASTER HUB V7 ⚡</h1>", unsafe_allow_html=True)
    
    if st.button("🚀 KÍCH HOẠT QUÉT DỮ LIỆU REAL-TIME"):
        st.cache_data.clear()
        st.rerun()

    df = load_data_clean()

    if not df.empty:
        core = TitanCore(df)
        last_num, freq_stats = core.get_summary()

        # Chỉ số nhanh
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng Kỳ", len(df))
        c2.metric("Kỳ Gần Nhất", last_num)
        c3.metric("Trạng Thái", "ĐANG CHẠY ✅")

        st.divider()

        # AI CHỐT SỐ
        st.subheader("🤖 AI GEMINI NHẬN ĐỊNH")
        with st.spinner('Đang soi cầu đa vị trí...'):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                history = df['numbers'].tail(15).tolist()
                prompt = f"Bạn là TITAN MASTER AI. Dữ liệu: {history}. Hãy phân tích nhịp bệt và nhịp nhảy. Chốt 1 cặp 2D duy nhất. Trả lời cực ngắn: NHẬN ĐỊNH: ... CHỐT SỐ: XX-YY CHIẾN THUẬT: ..."
                res = model.generate_content(prompt)
                st.markdown(f'<div class="prediction-card">{res.text}</div>', unsafe_allow_html=True)
            except:
                st.info("Dữ liệu đang đồng bộ, hãy nhấn quét lại sau 5 giây.")

        # Biểu đồ tần suất
        st.subheader("📊 Tần suất 30 kỳ gần nhất")
        chart_df = pd.DataFrame([{"Số": k, "Lượt": v} for k, v in freq_stats.items()])
        fig = px.bar(chart_df, x="Số", y="Lượt", color_discrete_sequence=['#00ff00'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📂 Xem 20 kỳ mới nhất"):
            st.table(df.tail(20))
    else:
        st.error("❌ KHÔNG TÌM THẤY SỐ: Anh hãy kiểm tra file Sheets xem có đúng dãy 5 số ở cột A không nhé!")

if __name__ == "__main__":
    main()
