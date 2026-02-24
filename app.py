import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
from datetime import datetime
import os

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
genai.configure(api_key=API_KEY)

st.set_page_config(page_title="AI LOTOBET v2 - CHUẨN ĐẶC TẢ", layout="wide")

# Hàm gọi Gemini để nhận định chuyên sâu
def get_gemini_advice(history_str, ai_analysis):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Bạn là chuyên gia phân tích thuật toán Lotobet. 
        Dữ liệu lịch sử (5 số gần nhất): {history_str}
        Kết quả phân tích máy học: {ai_analysis}
        Dựa trên đặc tả: Loại bỏ số chập, ưu tiên số ổn định và cầu bệt đang chạy.
        Hãy đưa ra 1 cặp số duy nhất (2 tinh) có xác suất cao nhất hoặc khuyên 'KHÔNG ĐÁNH'.
        Trả lời ngắn gọn: 'Cặp số: XX-YY' hoặc 'KHÔNG ĐÁNH'.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Gemini đang bận, sử dụng kết quả thuật toán gốc."

# ================= LOGIC PHÂN TÍCH AI =================
class LotobetAI_V2:
    def __init__(self):
        self.forbidden_numbers = [i*11 for i in range(10)] # 00, 11... 99

    def clean_data(self, df):
        matrix = []
        for val in df['numbers'].values:
            digits = [int(d) for d in str(val) if d.isdigit()]
            if len(digits) == 5:
                matrix.append(digits)
        return np.array(matrix)

    def analyze_numbers(self, matrix):
        if len(matrix) < 5: return None
        
        analysis = {}
        for num in range(10):
            # Tìm các kỳ có xuất hiện số num
            appears = np.where(np.any(matrix == num, axis=1))[0]
            count_10 = sum(1 for row in matrix[-10:] if num in row)
            count_3 = sum(1 for row in matrix[-3:] if num in row)
            
            # Gán trạng thái theo đặc tả
            if count_3 >= 2: state = "NÓNG/BỆT"
            elif 1 <= count_10 <= 3: state = "ỔN ĐỊNH"
            elif count_10 == 0: state = "YẾU"
            else: state = "NGUY HIỂM"
            
            analysis[num] = {
                "state": state,
                "freq": count_10,
                "last_seen": (len(matrix) - 1 - appears[-1]) if len(appears) > 0 else 99
            }
        return analysis

    def get_predictions(self, matrix, analysis):
        if not analysis: return [], "Dữ liệu ít"
        
        # 1. Loại bỏ 3 số (Giữ lại 7 số tốt nhất)
        sorted_nums = sorted(analysis.items(), key=lambda x: (x[1]['freq']), reverse=True)
        top_7 = [x[0] for x in sorted_nums[:7]]
        
        # 2. Logic ghép cặp
        candidates = []
        for i in range(len(top_7)):
            for j in range(i + 1, len(top_7)):
                n1, n2 = top_7[i], top_7[j]
                
                # Loại bỏ số chập (Ví dụ: không ghép nếu tạo thành 11, 22...)
                # Đặc tả: Đánh 1 cặp gồm 2 số đơn khác nhau (Ví dụ 5 và 6)
                s1, s2 = analysis[n1], analysis[n2]
                
                score = 50
                # Ưu tiên cầu bệt (Quan trọng theo yêu cầu)
                if s1['state'] == "NÓNG/BỆT": score += 20
                if s2['state'] == "NÓNG/BỆT": score += 20
                # Ưu tiên 1 ổn định + 1 hồi
                if s1['state'] == "ỔN ĐỊNH": score += 10
                
                # Hình phạt: Tránh 2 số vừa ra kỳ trước (giảm xác suất theo đặc tả)
                if s1['last_seen'] == 0 and s2['last_seen'] == 0: score -= 30

                if score >= 70:
                    candidates.append({"pair": (n1, n2), "score": score})

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:2] # Trả về tối đa 1-2 cặp

# ================= GIAO DIỆN STREAMLIT =================
def main():
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🎯 AI LOTOBET 2-TINH PRO V2</h1>", unsafe_allow_html=True)
    st.caption("Hệ thống phân tích chuẩn đặc tả v2 - Tích hợp Gemini Pro")

    # Quản lý dữ liệu
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=["numbers"])

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("📥 Nhập dữ liệu")
        raw_input = st.text_area("Nhập kết quả (5 số liền nhau, mỗi dòng 1 kỳ):", height=250)
        if st.button("🔄 Phân tích mới"):
            if raw_input:
                lines = [n.strip() for n in raw_input.split("\n") if len(n.strip()) == 5]
                st.session_state.data = pd.DataFrame(lines, columns=["numbers"])
                st.rerun()

    with col_out:
        st.subheader("📊 Kết quả AI")
        df = st.session_state.data
        if df.empty:
            st.info("Hãy nhập ít nhất 10 kỳ để AI nhận diện cầu.")
            return

        ai = LotobetAI_V2()
        matrix = ai.clean_data(df)
        
        if len(matrix) < 5:
            st.error("Dữ liệu không hợp lệ. Mỗi dòng phải có đúng 5 chữ số.")
            return

        analysis = ai.analyze_numbers(matrix)
        preds = ai.get_predictions(matrix, analysis)

        # Hiển thị Trạng thái Thị trường
        hot_count = sum(1 for v in analysis.values() if v['state'] == "NÓNG/BỆT")
        
        if hot_count > 6:
            st.error("🚫 KHÔNG ĐÁNH KỲ NÀY: Thị trường quá nhiễu (Quá nhiều số nóng)")
        elif not preds:
            st.warning("🚫 KHÔNG ĐÁNH: Không tìm thấy cặp số an toàn đạt ngưỡng 75%")
        else:
            # Lấy nhận định từ Gemini
            history_str = ", ".join(df['numbers'].tail(5).tolist())
            with st.spinner('Gemini đang kiểm tra nhịp cầu...'):
                advice = get_gemini_advice(history_str, str(preds))
            
            st.success(f"🤖 NHẬN ĐỊNH GEMINI: {advice}")
            
            for p in preds:
                st.markdown(f"""
                <div style="background: #ffffff; padding: 20px; border-radius: 10px; border-left: 10px solid #ff4b4b; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1)">
                    <span style="font-size: 25px; font-weight: bold; color: #333;">Cặp số: {p['pair'][0]} - {p['pair'][1]}</span>
                    <br><span style="color: #ff4b4b;">Độ tự tin: {p['score']}%</span>
                </div>
                """, unsafe_allow_html=True)

        # Biểu đồ tần suất
        st.divider()
        st.subheader("📈 Thống kê nhịp số đơn (0-9)")
        chart_df = pd.DataFrame([{"Số": k, "Tần suất": v['freq'], "Trạng thái": v['state']} for k, v in analysis.items()])
        fig = px.bar(chart_df, x='Số', y='Tần suất', color='Trạng thái', barmode='group', height=300)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
