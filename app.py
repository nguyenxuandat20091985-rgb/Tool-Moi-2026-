import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
import requests
from io import StringIO

# ================= CẤU HÌNH HỆ THỐNG =================
# Cập nhật API Key mới của anh Đạt
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
genai.configure(api_key=GEMINI_API_KEY) 

# Link Google Sheets của anh (Định dạng xuất CSV)
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA/export?format=csv"

st.set_page_config(page_title="TITAN AI PRO - VIP", layout="wide") 

# Tùy chỉnh giao diện bằng CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .prediction-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #000000 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #00ff00;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Hàm lấy dữ liệu từ Google Sheets
def load_data():
    try:
        response = requests.get(SHEET_CSV_URL, timeout=10)
        response.encoding = 'utf-8'
        df = pd.read_csv(StringIO(response.text))
        # Tự động tìm cột chứa 5 số
        for col in df.columns:
            if df[col].astype(str).str.contains(r'\d{5}').any():
                df = df.rename(columns={col: 'numbers'})
                break
        return df[['numbers']].dropna()
    except Exception as e:
        st.error(f"⚠️ Lỗi đồng bộ Sheets: {e}")
        return pd.DataFrame(columns=["numbers"])

# Hàm gọi Gemini nhận định chuyên sâu
def get_gemini_analysis(history_data, logic_results):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Sử dụng model Flash để tốc độ nhanh hơn
        prompt = f"""
        Bạn là hệ thống trí tuệ nhân tạo TITAN v6.
        Dữ liệu lịch sử 10 kỳ: {history_data}
        Kết quả thuật toán: {logic_results}
        
        Nhiệm vụ:
        1. Phân tích nhịp cầu (Bệt, Nhảy, hay Đảo).
        2. Loại bỏ các số có dấu hiệu 'Gan' quá lâu.
        3. Chốt 1 cặp duy nhất có xác suất > 85%.
        
        Trả lời theo định dạng:
        - Nhận định: [Ngắn gọn 1 câu]
        - Chốt số: XX - YY
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ Gemini đang bận phân tích nhịp cầu, hãy thử lại sau."

# ================= LOGIC TOÁN HỌC TITAN =================
class TitanEngine:
    def process_matrix(self, df):
        matrix = []
        for val in df['numbers'].astype(str).tail(150):
            nums = [int(d) for d in val if d.isdigit()]
            if len(nums) == 5: matrix.append(nums)
        return np.array(matrix)

    def run_stats(self, matrix):
        if len(matrix) < 10: return None
        stats = {}
        for i in range(10):
            # Tần suất trong 15 kỳ gần nhất
            freq = np.sum(matrix[-15:] == i)
            # Kỳ cuối cùng xuất hiện
            last_idx = np.where(np.any(matrix == i, axis=1))[0]
            gap = (len(matrix) - 1 - last_idx[-1]) if len(last_idx) > 0 else 99
            
            state = "ỔN ĐỊNH"
            if freq > 10: state = "CẦU NÓNG"
            elif gap > 8: state = "SỐ GAN"
            
            stats[i] = {"freq": int(freq), "gap": int(gap), "state": state}
        return stats

# ================= GIAO DIỆN CHÍNH =================
def main():
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>⚡ TITAN AI MASTER HUB ⚡</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Hệ thống dự đoán 5D Bet - Đồng bộ Real-time</p>", unsafe_allow_html=True)

    # Sidebar điều khiển
    st.sidebar.header("CÀI ĐẶT HỆ THỐNG")
    if st.sidebar.button("🔄 LÀM MỚI DỮ LIỆU"):
        st.cache_data.clear()
        st.rerun()

    data = load_data()
    
    if not data.empty:
        engine = TitanEngine()
        matrix = engine.process_matrix(data)
        
        if len(matrix) >= 10:
            stats = engine.run_stats(matrix)
            
            # Khu vực hiển thị kết quả chính
            st.subheader("🔮 KẾT QUẢ PHÂN TÍCH TỪ AI")
            
            history_str = str(data['numbers'].tail(10).tolist())
            with st.spinner('AI đang quét nhịp cầu...'):
                ai_advice = get_gemini_analysis(history_str, str(stats))
            
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: #00ff00; margin-top: 0;">🎯 LỜI KHUYÊN HỆ THỐNG</h3>
                    <p style="font-size: 1.2rem; color: #ffffff;">{ai_advice}</p>
                </div>
            """, unsafe_allow_html=True)

            # Thống kê chi tiết
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📊 Biểu đồ nhịp số (0-9)")
                chart_df = pd.DataFrame([
                    {"Số": k, "Tần suất (15 kỳ)": v['freq'], "Trạng thái": v['state']} 
                    for k, v in stats.items()
                ])
                fig = px.bar(chart_df, x='Số', y='Tần suất (15 kỳ)', color='Trạng thái',
                             color_discrete_map={"CẦU NÓNG": "#ff0000", "ỔN ĐỊNH": "#00ff00", "SỐ GAN": "#555555"})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📋 Dữ liệu mới nhất")
                st.dataframe(data.tail(10), use_container_width=True)
        else:
            st.warning("Dữ liệu Sheets cần ít nhất 10 kỳ để bắt đầu phân tích.")
    else:
        st.error("Không tìm thấy dữ liệu numbers. Hãy kiểm tra lại cột 'numbers' trong file Google Sheets.")

if __name__ == "__main__":
    main()
