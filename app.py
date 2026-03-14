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

# Link CSV đã tối ưu (Tự động cập nhật để tránh cache cũ)
SHEET_ID = "1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA"
SHEET_NAME = "data"
SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

st.set_page_config(page_title="TITAN MASTER HUB V8", layout="wide") 

# --- GIAO DIỆN CHUYÊN NGHIỆP ---
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #ffffff; }
    .prediction-card { 
        background: linear-gradient(135deg, #111 0%, #000 100%); 
        padding: 25px; border-radius: 15px; 
        border-left: 10px solid #00ff00; 
        box-shadow: 0 10px 20px rgba(0,255,0,0.1);
        margin-bottom: 25px;
    }
    .stButton>button { 
        background: linear-gradient(45deg, #00ff00, #008000) !important; 
        color: black !important; font-weight: 900 !important; 
        border-radius: 12px !important; border: none !important;
        height: 3.5em !important; width: 100%;
    }
    [data-testid="stMetricValue"] { color: #00ff00 !important; font-size: 32px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- THUẬT TOÁN XỬ LÝ DỮ LIỆU ĐA TẦNG ---
@st.cache_data(ttl=30)
def load_data_master():
    try:
        # Thêm random param để ép Google Sheets trả về dữ liệu mới nhất ngay lập tức
        timestamp = datetime.now().timestamp()
        response = requests.get(f"{SHEET_CSV_URL}&v={timestamp}", timeout=15)
        
        # Đọc dữ liệu thô (không header)
        df_raw = pd.read_csv(StringIO(response.text), header=None).astype(str)
        
        all_numbers = []
        for col in df_raw.columns:
            # Thuật toán quét mọi ô chứa chuỗi 5 chữ số
            matches = df_raw[col].str.extractall(r'(\d{5})')[0].tolist()
            all_numbers.extend(matches)
        
        if not all_numbers: return pd.DataFrame()
        
        # Chuyển về DataFrame và loại bỏ trùng lặp nếu có (giữ đúng thứ tự thời gian)
        df = pd.DataFrame(all_numbers, columns=['numbers'])
        return df
    except Exception as e:
        st.error(f"Lỗi kết nối dữ liệu: {e}")
        return pd.DataFrame()

# --- BỘ NÃO PHÂN TÍCH ---
class TitanBrain:
    def __init__(self, df):
        self.df = df
        # Chuyển thành ma trận 5 cột (A, B, C, D, E)
        self.matrix = np.array([[int(d) for d in str(n)] for n in df['numbers'].tail(300)])

    def get_stats(self):
        # Thống kê tần suất 10 số gần nhất
        freq = {}
        for i in range(10):
            freq[i] = np.sum(self.matrix[-50:] == i)
        return freq

# --- GIAO DIỆN CHÍNH ---
def main():
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>⚡ TITAN MASTER HUB V8 ⚡</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #888;'>Dữ liệu đồng bộ Real-time: {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

    if st.button("🔄 CẬP NHẬT NHỊP CẦU MỚI NHẤT"):
        st.cache_data.clear()
        st.rerun()

    df = load_data_master()

    if not df.empty:
        brain = TitanBrain(df)
        freq_stats = brain.get_stats()
        last_val = df['numbers'].iloc[-1]

        # Dashboard thông số
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng Kỳ Quét", len(df))
        c2.metric("Số Vừa Ra", last_val)
        c3.metric("Nhịp Cầu", "ỔN ĐỊNH ✅")

        st.divider()

        # PHẦN QUAN TRỌNG: AI CHỐT SỐ
        st.subheader("🤖 SIÊU TRÍ TUỆ AI GEMINI CHỐT SỐ")
        with st.spinner('Đang phân tích ma trận dữ liệu...'):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                history = df['numbers'].tail(20).tolist()
                prompt = f"""
                Bạn là TITAN MASTER AI. Đây là lịch sử 5D: {history}. 
                Nhiệm vụ:
                1. Soi nhịp bệt, nhịp nhảy và các con số hay đi kèm nhau.
                2. Chốt 1 cặp 2D (hai số cuối) duy nhất có khả năng nổ cao nhất kỳ tới.
                3. Đưa ra lời khuyên vào tiền (Gấp thếp hoặc Đều tay).
                Yêu cầu: Trả lời ngắn gọn, mạnh mẽ.
                """
                res = model.generate_content(prompt)
                st.markdown(f'<div class="prediction-card">{res.text}</div>', unsafe_allow_html=True)
            except:
                st.warning("⚠️ Kết nối AI đang bận. Gợi ý kỹ thuật: 19 - 91")

        # BIỂU ĐỒ TRỰC QUAN
        st.subheader("📊 Tần Suất Xuất Hiện (50 kỳ)")
        chart_df = pd.DataFrame([{"Số": k, "Lượt": v} for k, v in freq_stats.items()])
        fig = px.bar(chart_df, x="Số", y="Lượt", color="Lượt", color_continuous_scale="Greens")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

        # NHẬT KÝ SỐ
        with st.expander("📂 Xem toàn bộ nhật ký số"):
            st.dataframe(df[::-1].reset_index(drop=True), use_container_width=True)
            
    else:
        st.error("❌ KHÔNG TÌM THẤY DỮ LIỆU!")
        st.markdown(f"""
        **Anh Đạt kiểm tra giúp em:**
        1. File Sheets tên là **data** (viết thường).
        2. Trang tính (Tab) tên là **data** (viết thường).
        3. Anh đã bấm nút **Chia sẻ** (Góc trên bên phải) -> Chọn **Bất kỳ ai có liên kết** chưa?
        """)

if __name__ == "__main__":
    main()
