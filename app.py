import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
import requests
from io import StringIO

# ================= CẤU HÌNH HỆ THỐNG =================
# API Key của anh Đạt
API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
genai.configure(api_key=API_KEY) 

# Link Google Sheets của anh (Đã chuyển sang định dạng xuất CSV để AI đọc nhanh)
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA/export?format=csv"

st.set_page_config(page_title="TITAN AI - DỮ LIỆU ĐỒNG BỘ", layout="wide") 

# Hàm lấy dữ liệu trực tiếp từ Google Sheets
def load_data_from_sheets():
    try:
        response = requests.get(SHEET_CSV_URL)
        response.encoding = 'utf-8'
        df = pd.read_csv(StringIO(response.text))
        # Giả định cột chứa số 5D tên là 'numbers' hoặc cột đầu tiên
        if 'numbers' not in df.columns:
            df.columns = ['numbers'] + list(df.columns[1:])
        return df
    except Exception as e:
        st.error(f"Lỗi kết nối dữ liệu Sheets: {e}")
        return pd.DataFrame(columns=["numbers"])

# Hàm gọi Gemini nhận định
def get_gemini_advice(history_str, ai_analysis):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Bạn là chuyên gia toán xác suất TITAN. 
        Dữ liệu gần đây: {history_str}
        Kết quả máy học: {ai_analysis}
        Yêu cầu: Loại bỏ số chập, ưu tiên nhịp cầu bệt.
        Hãy đưa ra 1 cặp số duy nhất (2 số khác nhau) hoặc khuyên 'KHÔNG ĐÁNH'.
        Trả lời cực ngắn: 'Cặp số: XX-YY' hoặc 'KHÔNG ĐÁNH'.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Gemini đang bảo trì nhịp cầu."

# ================= LỚP PHÂN TÍCH AI =================
class LotobetAI_V2:
    def clean_data(self, df):
        matrix = []
        # Lấy 100 dòng gần nhất để đảm bảo tốc độ
        valid_data = df['numbers'].astype(str).tail(100).values
        for val in valid_data:
            digits = [int(d) for d in val if d.isdigit()]
            if len(digits) == 5:
                matrix.append(digits)
        return np.array(matrix)

    def analyze_numbers(self, matrix):
        if len(matrix) < 5: return None
        analysis = {}
        for num in range(10):
            appears = np.where(np.any(matrix == num, axis=1))[0]
            count_10 = sum(1 for row in matrix[-10:] if num in row)
            count_3 = sum(1 for row in matrix[-3:] if num in row)
            
            if count_3 >= 2: state = "NÓNG/BỆT"
            elif 1 <= count_10 <= 3: state = "ỔN ĐỊNH"
            else: state = "YẾU/GAN"
            
            analysis[num] = {
                "state": state,
                "freq": count_10,
                "last_seen": (len(matrix) - 1 - appears[-1]) if len(appears) > 0 else 99
            }
        return analysis

    def get_predictions(self, analysis):
        if not analysis: return []
        sorted_nums = sorted(analysis.items(), key=lambda x: x[1]['freq'], reverse=True)
        top_7 = [x[0] for x in sorted_nums[:7]]
        
        candidates = []
        for i in range(len(top_7)):
            for j in range(i + 1, len(top_7)):
                n1, n2 = top_7[i], top_7[j]
                s1, s2 = analysis[n1], analysis[n2]
                score = 50
                if s1['state'] == "NÓNG/BỆT": score += 20
                if s2['state'] == "NÓNG/BỆT": score += 20
                if s1['state'] == "ỔN ĐỊNH": score += 10
                if s1['last_seen'] == 0 and s2['last_seen'] == 0: score -= 30 
                if score >= 70:
                    candidates.append({"pair": (n1, n2), "score": score})
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:2]

# ================= GIAO DIỆN CHÍNH =================
def main():
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>⚡ TITAN AI REAL-TIME v2</h1>", unsafe_allow_html=True)
    
    # Nút bấm đồng bộ
    if st.button("🔄 ĐỒNG BỘ DỮ LIỆU TỪ GOOGLE SHEETS"):
        st.session_state.data = load_data_from_sheets()
        st.success("Đã cập nhật dữ liệu mới nhất từ Sheets!")

    if 'data' not in st.session_state:
        st.session_state.data = load_data_from_sheets()

    df = st.session_state.data
    
    if not df.empty:
        st.info(f"Dữ liệu hiện có: {len(df)} kỳ quay.")
        ai = LotobetAI_V2()
        matrix = ai.clean_data(df)
        
        if len(matrix) >= 5:
            analysis = ai.analyze_numbers(matrix)
            preds = ai.get_predictions(analysis)

            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🤖 AI Dự Đoán")
                if not preds:
                    st.warning("Hệ thống khuyên: KHÔNG ĐÁNH (Nhịp cầu không đẹp)")
                else:
                    history_str = ", ".join(df['numbers'].tail(5).astype(str).tolist())
                    advice = get_gemini_advice(history_str, str(preds))
                    st.success(f"NHẬN ĐỊNH GEMINI: {advice}")
                    
                    for p in preds:
                        st.markdown(f"""
                        <div style="background: #111; padding: 20px; border-radius: 10px; border: 1px solid #00ff00; margin-bottom: 10px;">
                            <span style="font-size: 24px; color: #ffff00;">Cặp số vàng: {p['pair'][0]} - {p['pair'][1]}</span>
                            <br><span style="color: #00ff00;">Xác suất TITAN: {p['score']}%</span>
                        </div>
                        """, unsafe_allow_html=True)

            with col2:
                st.subheader("📈 Trạng thái")
                hot_count = sum(1 for v in analysis.values() if v['state'] == "NÓNG/BỆT")
                st.metric("Số lượng số NÓNG", hot_count)
                if hot_count > 6: st.error("Thị trường NHIỄU")
                else: st.write("Thị trường ỔN ĐỊNH")

            # Biểu đồ
            st.divider()
            chart_df = pd.DataFrame([{"Số": k, "Tần suất": v['freq'], "Trạng thái": v['state']} for k, v in analysis.items()])
            fig = px.bar(chart_df, x='Số', y='Tần suất', color='Trạng thái', title="Thống kê nhịp số 0-9")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dữ liệu trong Sheets không đủ 5 kỳ chuẩn (5 chữ số).")
    else:
        st.error("Không thể lấy dữ liệu. Vui lòng kiểm tra quyền chia sẻ link Google Sheets (Bất kỳ ai có liên kết đều có thể xem).")

if __name__ == "__main__":
    main()
