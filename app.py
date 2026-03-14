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

SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA/export?format=csv"

st.set_page_config(page_title="TITAN MASTER HUB V7 - ULTIMATE", layout="wide") 

# --- GIAO DIỆN DARK VIP ---
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #00ff00 !important; font-size: 28px !important; }
    .stButton>button { background: linear-gradient(45deg, #00ff00, #008000) !important; color: #000 !important; font-weight: bold; border-radius: 10px; border: none; height: 3em; }
    .reportview-container .main { background: #050505; }
    .prediction-card { background: #111; padding: 25px; border-radius: 15px; border-left: 10px solid #00ff00; box-shadow: 0 4px 15px rgba(0,255,0,0.1); margin-bottom: 25px; }
    .status-tag { padding: 5px 15px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- HÀM LẤY DỮ LIỆU THÔNG MINH ---
@st.cache_data(ttl=60)
def load_data_v7():
    try:
        response = requests.get(SHEET_CSV_URL, timeout=15)
        df = pd.read_csv(StringIO(response.text), header=None)
        all_numbers = []
        for col in df.columns:
            vals = df[col].astype(str).str.strip().str.extract(r'^(\d{5})$')[0].dropna()
            all_numbers.extend(vals.tolist())
        if not all_numbers: return pd.DataFrame()
        return pd.DataFrame(all_numbers, columns=['numbers'])
    except:
        return pd.DataFrame()

# --- BỘ NÃO PHÂN TÍCH TITAN V7 ---
class TitanBrainV7:
    def __init__(self, df):
        self.df = df
        # Chuyển đổi thành ma trận số học (Mỗi cột là 1 vị trí từ A-E)
        self.matrix = np.array([[int(d) for d in str(n)] for n in df['numbers'].tail(300)])
        
    def analyze_positions(self):
        pos_stats = []
        for i in range(5):
            col_data = self.matrix[:, i]
            freq = np.bincount(col_data[-20:], minlength=10)
            most_freq = np.argmax(freq)
            pos_stats.append({"pos": i, "most_freq": most_freq, "freq_val": freq[most_freq]})
        return pos_stats

    def get_market_sentiment(self):
        # Kiểm tra độ "nhiễu" của thị trường qua 10 kỳ gần nhất
        last_10 = self.matrix[-10:]
        shuffled_score = sum(1 for row in last_10 if len(set(row)) < 4)
        if shuffled_score > 6: return "RỦI RO CAO (NHIỄU)", "red"
        if shuffled_score > 3: return "TRUNG BÌNH", "yellow"
        return "ỔN ĐỊNH (CẦU ĐẸP)", "green"

# --- GIAO DIỆN CHÍNH ---
def main():
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>⚡ TITAN MASTER HUB V7 - ULTIMATE ⚡</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #888;'>Cập nhật lúc: {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

    # Sidebar quản lý vốn
    with st.sidebar:
        st.header("💰 QUẢN LÝ VỐN")
        vốn = st.number_input("Tổng vốn đầu tư (VNĐ):", value=5000000, step=500000)
        st.write(f"🎯 Mục tiêu lãi (20%): **{vốn*0.2:,.0f}**")
        st.write(f"🛑 Cắt lỗ an toàn (15%): **{vốn*0.15:,.0f}**")
        st.divider()
        if st.button("🔄 LÀM MỚI DỮ LIỆU"):
            st.cache_data.clear()
            st.rerun()

    df = load_data_v7()

    if not df.empty:
        brain = TitanBrainV7(df)
        pos_stats = brain.analyze_positions()
        sentiment, s_color = brain.get_market_sentiment()

        # Thẻ chỉ số KPI
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng Kỳ Quét", f"{len(df)}")
        c2.metric("Số Kỳ Bệt", f"{sum(1 for i in range(len(df)-1) if df['numbers'].iloc[i] == df['numbers'].iloc[i+1])}")
        c3.metric("Số Nóng", f"{pos_stats[0]['most_freq']}")
        c4.metric("Thị Trường", sentiment)

        st.divider()

        # KHU VỰC AI CHỐT SỐ
        st.subheader("🤖 NHẬN ĐỊNH CHUYÊN GIA GEMINI AI")
        with st.spinner('AI đang quét nhịp cầu đa tầng...'):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                history = df['numbers'].tail(15).tolist()
                prompt = f"""
                Bạn là TITAN MASTER AI. Phân tích dữ liệu: {history}. 
                Thị trường hiện tại: {sentiment}.
                Hãy thực hiện:
                1. Soi cầu bệt và nhịp nhảy giữa các kỳ.
                2. Chốt 1 cặp 2D duy nhất có xác suất nổ cao nhất.
                3. Đưa ra chiến thuật vào tiền (Vd: Gấp thếp 1-2-4 hoặc Đều tay).
                Trả lời ngắn gọn, chuyên nghiệp. Định dạng: 
                - NHẬN ĐỊNH: ...
                - CHỐT SỐ: XX-YY
                - CHIẾN THUẬT: ...
                """
                res = model.generate_content(prompt)
                prediction = res.text
            except:
                prediction = "Hệ thống AI đang quá tải nhịp cầu. Gợi ý kỹ thuật: **38 - 83**"

        st.markdown(f'<div class="prediction-card">{prediction}</div>', unsafe_allow_html=True)

        # PHÂN TÍCH VỊ TRÍ (A-B-C-D-E)
        st.subheader("📊 Phân Tích Chi Tiết 5 Vị Trí (20 kỳ gần nhất)")
        tabs = st.tabs(["Vị trí A", "Vị trí B", "Vị trí C", "Vị trí D", "Vị trí E"])
        for i, tab in enumerate(tabs):
            with tab:
                col_data = brain.matrix[:, i]
                unique, counts = np.unique(col_data[-30:], return_counts=True)
                fig = px.pie(values=counts, names=unique, hole=.4, title=f"Tỷ lệ xuất hiện Vị trí {chr(65+i)}", color_discrete_sequence=px.colors.sequential.Greens_r)
                st.plotly_chart(fig, use_container_width=True)

        # BIỂU ĐỒ NHỊP CẦU
        st.divider()
        st.subheader("📈 Xu Hướng Nhịp Tổng (Trendline)")
        # Tính tổng các chữ số của mỗi kỳ để xem biến động Tài/Xỉu
        trend_data = [sum([int(d) for d in str(n)]) for n in df['numbers'].tail(50)]
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(y=trend_data, mode='lines+markers', line=dict(color='#00ff00', width=3)))
        fig_trend.update_layout(title="Biến động Tổng Điểm 50 kỳ", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig_trend, use_container_width=True)

        # DỮ LIỆU GỐC
        with st.expander("📂 Xem Nhật Ký Dữ Liệu Gốc"):
            st.dataframe(df.tail(50), use_container_width=True)
    else:
        st.error("❌ CHƯA CÓ DỮ LIỆU: Anh Đạt hãy kiểm tra lại Google Sheets. Hãy đảm bảo cột A chứa các dãy 5 số và đã chia sẻ quyền truy cập!")

if __name__ == "__main__":
    main()
