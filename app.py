import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIGURATION =================
st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide", page_icon="🎯")

# Kết nối Gemini AI
GEMINI_API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

DATA_FILE = "lotobet_v2_data.csv"

# ================= LOGIC AI CHUẨN V2 =================
class LotobetLogicV2:
    def __init__(self):
        self.min_confidence = 60
        self.states = {
            "HOT": "NÓNG (Ra dày)",
            "STABLE": "ỔN ĐỊNH (Đều)",
            "WEAK": "YẾU (Ít ra)",
            "DANGER": "NGUY HIỂM (Gãy/Lặp)"
        }

    def process_matrix(self, df):
        """Chuyển đổi dữ liệu thô sang ma trận số đơn 0-9"""
        matrix = []
        for val in df['numbers'].values:
            s_val = str(val).strip()
            if len(s_val) == 5:
                matrix.append([int(d) for d in s_val])
        return np.array(matrix)

    def analyze_numbers(self, matrix):
        """Phân tích 10 số đơn theo đặc tả bước 3, 4, 5, 6"""
        if len(matrix) < 5: return None
        
        analysis = {}
        total_draws = len(matrix)
        
        for num in range(10):
            # Tìm các kỳ xuất hiện (index)
            appears = np.where(np.any(matrix == num, axis=1))[0]
            count_10 = sum(1 for row in matrix[-10:] if num in row)
            count_3 = sum(1 for row in matrix[-3:] if num in row)
            
            # Tính khoảng cách (Gap)
            gaps = np.diff(appears) if len(appears) > 1 else []
            last_appearance = (total_draws - 1) - appears[-1] if len(appears) > 0 else 99
            
            # Định nhãn trạng thái (Bước 6)
            if count_3 >= 2: state = "HOT"
            elif 1 <= count_10 <= 3: state = "STABLE"
            elif last_appearance == 0: state = "DANGER" # Vừa ra kỳ trước
            else: state = "WEAK"

            # Nhận diện cầu (Bước 4)
            bridge = "BÌNH THƯỜNG"
            if len(gaps) >= 2 and all(g == 1 for g in gaps[-2:]): bridge = "BỆT (Bám sát)"
            elif len(gaps) >= 2 and gaps[-1] == gaps[-2]: bridge = "NHẢY ĐỀU"

            analysis[num] = {
                "num": num,
                "state": state,
                "bridge": bridge,
                "freq": count_10,
                "last_gap": last_appearance
            }
        return analysis

    def get_predictions(self, df):
        """Logic ghép cặp & Gọi Gemini kiểm chứng (Bước 7, 8)"""
        matrix = self.process_matrix(df)
        if len(matrix) < 8:
            return [], "THIẾU DỮ LIỆU", ["Cần ít nhất 8 kỳ để AI phân tích chính xác."]

        analysis = self.analyze_numbers(matrix)
        if not analysis: return [], "LỖI", ["Dữ liệu không hợp lệ."]

        # Lọc giữ lại 7 số tốt nhất (loại 3 số nhiễu nhất)
        sorted_nums = sorted(analysis.values(), key=lambda x: x['freq'], reverse=True)
        top_7 = [x['num'] for x in sorted_nums[:7]]

        scored_pairs = []
        for i in range(len(top_7)):
            for j in range(i + 1, len(top_7)):
                n1, n2 = top_7[i], top_7[j]
                # ❌ Loại số chập (Bước 1)
                if n1 == n2: continue
                
                s1, s2 = analysis[n1], analysis[n2]
                score = 65 # Base score

                # Cộng điểm theo đặc tả
                if s1['bridge'] == "BỆT (Bám sát)" or s2['bridge'] == "BỆT (Bám sát)": score += 15
                if s1['state'] == "STABLE" and s2['state'] == "STABLE": score += 10
                if s1['last_gap'] > 5: score += 5 # Cầu hồi

                # Trừ điểm (Bước 6)
                if s1['state'] == "HOT" and s2['state'] == "HOT": score -= 20
                if s1['state'] == "DANGER" or s2['state'] == "DANGER": score -= 15

                if score >= self.min_confidence:
                    scored_pairs.append({"pair": (n1, n2), "score": score})

        scored_pairs = sorted(scored_pairs, key=lambda x: x['score'], reverse=True)
        
        # Nếu không có cặp nào đạt 60% -> KHÔNG ĐÁNH (Bước 8)
        if not scored_pairs:
            return [], "SKIP", ["Không có cặp số nào đạt ngưỡng an toàn 60%."]

        return scored_pairs[:2], "PREDICT", []

# ================= GEMINI AI INTEGRATION =================
def ask_gemini_advice(history, suggestion):
    """Gửi dữ liệu cho Gemini để thẩm định cuối cùng"""
    prompt = f"""
    Bạn là chuyên gia xác suất Lotobet. 
    Lịch sử 10 kỳ gần nhất: {history}
    Thuật toán đề xuất cặp: {suggestion}
    Dựa trên đặc tả: Ưu tiên 1 số ổn định, 1 số nhảy nhịp, bám cầu bệt, loại số chập.
    Hãy trả lời ngắn gọn: Có nên đánh cặp này không? Tỷ lệ tin cậy bao nhiêu %?
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Gemini đang bận, hãy dựa vào điểm số thuật toán."

# ================= UI STREAMLIT =================
def main():
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎯 AI LOTOBET 2-TINH PRO v2</h1>", unsafe_allow_html=True)
    st.caption("Bản quyền 2026 - Hệ thống phân tích số đơn & bám cầu bệt")

    # Load Data
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=["numbers"])

    # Sidebar: Nhập liệu
    with st.sidebar:
        st.header("📥 Nhập dữ liệu")
        new_data = st.text_area("Nhập kết quả (5 số liền nhau, mỗi kỳ 1 dòng):", height=200)
        if st.button("💾 Cập nhật hệ thống"):
            if new_data:
                lines = [n.strip() for n in new_data.split("\n") if len(n.strip()) == 5]
                new_df = pd.DataFrame({"numbers": lines})
                df = pd.concat([df, new_df], ignore_index=True).tail(100)
                df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã cập nhật {len(lines)} kỳ!")
                st.rerun()
        
        if st.button("🗑 Xóa dữ liệu cũ"):
            if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
            st.rerun()

    # Main Dashboard
    if df.empty:
        st.info("Vui lòng nhập dữ liệu ở cột bên trái để bắt đầu.")
        return

    logic = LotobetLogicV2()
    
    col1, col2 = st.tabs(["📊 Phân tích & Dự đoán", "📈 Thống kê nhịp số"])

    with col1:
        preds, status, reasons = logic.get_predictions(df)
        
        if status == "SKIP":
            st.error("🚫 KHÔNG ĐÁNH KỲ NÀY")
            for r in reasons: st.write(f"- {r}")
        elif status == "PREDICT":
            st.success("✅ CẶP SỐ TIỀM NĂNG NHẤT")
            c1, c2 = st.columns(2)
            for i, p in enumerate(preds):
                with (c1 if i==0 else c2):
                    st.markdown(f"""
                    <div style="background:#262730; padding:20px; border-radius:10px; border-top: 4px solid #FF4B4B; text-align:center;">
                        <p style="color:gray; margin:0;">Cặp đề xuất {i+1}</p>
                        <h1 style="font-size: 50px; margin:10px 0;">{p['pair'][0]}{p['pair'][1]}</h1>
                        <p style="color:#FF4B4B; font-weight:bold;">Độ tin cậy: {p['score']}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Gemini Advice
            st.divider()
            with st.expander("🤖 Ý kiến từ Gemini AI"):
                with st.spinner("Đang hỏi ý kiến AI..."):
                    history_str = ", ".join(df['numbers'].tail(10).tolist())
                    suggestion = str([p['pair'] for p in preds])
                    advice = ask_gemini_advice(history_str, suggestion)
                    st.info(advice)

    with col2:
        matrix = logic.process_matrix(df)
        analysis = logic.analyze_numbers(matrix)
        if analysis:
            st.subheader("Trạng thái 10 số đơn")
            # Vẽ biểu đồ tần suất
            chart_df = pd.DataFrame([
                {"Số": k, "Tần suất (10 kỳ)": v['freq'], "Trạng thái": v['state']} 
                for k, v in analysis.items()
            ])
            fig = px.bar(chart_df, x='Số', y='Tần suất (10 kỳ)', color='Trạng thái',
                         title="Tần suất xuất hiện gần đây",
                         color_discrete_map={"HOT": "#FF4B4B", "STABLE": "#00CC96", "WEAK": "#636EFA", "DANGER": "#FFA15A"})
            st.plotly_chart(fig, use_container_width=True)
            
            # Bảng chi tiết
            st.table(pd.DataFrame(analysis).T[['num', 'state', 'bridge', 'last_gap']])

if __name__ == "__main__":
    main()
