import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIG & API =================
st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide")

# Cấu hình API Gemini
GEMINI_API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("Lỗi kết nối API Gemini. Hệ thống sẽ dùng thuật toán nội tại.")

DATA_FILE = "lotobet_history.csv"

# ================= CORE LOGIC AI v2 =================
class LotobetV2:
    def __init__(self):
        self.min_draws = 15
        self.labels = {"HOT": "NÓNG", "STABLE": "ỔN ĐỊNH", "WEAK": "YẾU", "RISKY": "NGUY HIỂM"}

    def clean_data(self, raw_text):
        """Xử lý dữ liệu đầu vào, loại bỏ lỗi"""
        lines = raw_text.split('\n')
        cleaned = []
        for line in lines:
            val = line.strip()
            if len(val) == 5 and val.isdigit():
                cleaned.append(val)
        return list(dict.fromkeys(cleaned)) # Loại bỏ kỳ trùng

    def analyze_numbers(self, df):
        """Phân tích số đơn theo đặc tả 4, 5, 6"""
        if len(df) < 5: return None
        
        # Chuyển thành ma trận số
        matrix = np.array([[int(d) for d in str(s)] for s in df['numbers'].values])
        analysis = {}
        
        for num in range(10):
            # 1. Tìm vị trí xuất hiện
            appears = np.where(np.any(matrix == num, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # 2. Tần suất theo giai đoạn (Đặc tả 5)
            last_3 = sum(1 for row in matrix[-3:] if num in row)
            last_5 = sum(1 for row in matrix[-5:] if num in row)
            last_10 = sum(1 for row in matrix[-10:] if num in row)
            
            # 3. Gán trạng thái (Đặc tả 6)
            current_gap = (len(matrix) - 1 - appears[-1]) if len(appears) > 0 else 99
            
            state = self.labels["STABLE"]
            if last_3 >= 2: state = self.labels["NGUY HIỂM"] # Vừa ra dồn
            elif last_5 >= 3: state = self.labels["NÓNG"]
            elif last_10 <= 1: state = self.labels["YẾU"]
            elif 3 <= current_gap <= 7: state = self.labels["STABLE"]
            
            # 4. Nhận diện cầu (Đặc tả 4)
            bridge = "BÌNH THƯỜNG"
            if len(gaps) >= 2 and gaps[-1] == gaps[-2] and gaps[-1] > 1:
                bridge = "CẦU NHẢY ✅"
            elif current_gap == 0:
                bridge = "CẦU LẶP ❌"
            elif 7 < current_gap < 12:
                bridge = "CẦU HỒI ✅"

            analysis[num] = {
                "state": state,
                "bridge": bridge,
                "freq_10": last_10,
                "current_gap": current_gap,
                "score": self.calculate_num_score(state, bridge, current_gap)
            }
        return analysis

    def calculate_num_score(self, state, bridge, gap):
        """Chấm điểm số đơn"""
        score = 50
        if state == self.labels["STABLE"]: score += 20
        if "✅" in bridge: score += 15
        if state == self.labels["NGUY HIỂM"]: score -= 30
        if state == self.labels["NÓNG"]: score -= 20
        if gap > 12: score -= 25 # Quá lâu (nhiễu)
        return score

    def get_prediction(self, df):
        """Logic ghép cặp & Không đánh (Đặc tả 7, 8)"""
        analysis = self.analyze_numbers(df)
        if not analysis: return None, "DỮ LIỆU CHƯA ĐỦ", []

        # Kiểm tra điều kiện KHÔNG ĐÁNH (Đặc tả 8)
        hot_count = sum(1 for v in analysis.values() if v['state'] in [self.labels["NÓNG"], self.labels["NGUY HIỂM"]])
        if hot_count >= 7 or len(df) < self.min_draws:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Thị trường quá nóng hoặc dữ liệu chưa đủ độ chín."]

        # Ghép cặp (Đặc tả 7)
        candidates = []
        for i in range(10):
            for j in range(i + 1, 10):
                s1, s2 = analysis[i], analysis[j]
                
                # Loại trừ theo đặc tả 6: Không ghép 2 nóng, 2 yếu, 2 nguy hiểm
                if s1['state'] == s2['state'] and s1['state'] in [self.labels["NÓNG"], self.labels["NGUY HIỂM"], self.labels["YẾU"]]:
                    continue
                
                total_score = (s1['score'] + s2['score']) / 2
                candidates.append({"pair": f"{i}{j}", "score": total_score})

        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if not candidates or candidates[0]['score'] < 60:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Không có cặp số nào đạt ngưỡng an toàn (60%)."]

        return candidates[0], "CÓ KẾT QUẢ", []

# ================= INTERFACE =================
def main():
    st.header("🎯 AI LOTOBET 2-TINH (BẢN CHUẨN v2)")
    engine = LotobetV2()

    # Quản lý dữ liệu
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["numbers"]).to_csv(DATA_FILE, index=False)

    df = pd.read_csv(DATA_FILE)

    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📥 Nhập dữ liệu"])

    with tab2:
        raw_input = st.text_area("Nhập kết quả (5 số viết liền, mỗi dòng 1 kỳ):", height=200)
        if st.button("Lưu & Phân tích"):
            cleaned = engine.clean_data(raw_input)
            new_df = pd.DataFrame(cleaned, columns=["numbers"])
            new_df.to_csv(DATA_FILE, index=False)
            st.success(f"Đã lưu {len(cleaned)} kỳ gần nhất!")
            st.rerun()

    with tab1:
        if len(df) < 5:
            st.warning("Vui lòng nhập thêm dữ liệu (Cần ít nhất 15 kỳ để chuẩn nhất).")
            return

        analysis = engine.analyze_numbers(df)
        pred, status, reasons = engine.get_prediction(df)

        # Hiển thị kết quả Dự đoán
        st.subheader("🔮 Kết quả soi cầu")
        if status == "KHÔNG ĐÁNH KỲ NÀY":
            st.error("🚫 KHÔNG ĐÁNH KỲ NÀY")
            for r in reasons: st.write(f"- {r}")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div style="background:#1E1E1E; padding:20px; border-radius:15px; text-align:center; border: 2px solid #FF4B4B;">
                    <h1 style="color:white; font-size:60px; margin:0;">{pred['pair']}</h1>
                    <p style="color:#FF4B4B; font-weight:bold;">ĐỘ TỰ TIN: {pred['score']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.info("Ưu tiên: 1 ổn định + 1 số đang hồi hoặc nhảy nhịp. Đánh cặp không cố định vị trí.")

        # Biểu đồ trạng thái (Đặc tả 6)
        
        st.divider()
        st.subheader("📊 Trạng thái chuỗi số đơn (0-9)")
        chart_data = pd.DataFrame([{"Số": k, "Tần suất (10 kỳ)": v['freq_10'], "Trạng thái": v['state']} for k, v in analysis.items()])
        fig = px.bar(chart_data, x='Số', y='Tần suất (10 kỳ)', color='Trạng thái', 
                     color_discrete_map={engine.labels["NÓNG"]: "red", engine.labels["STABLE"]: "green", engine.labels["YẾU"]: "gray", engine.labels["NGUY HIỂM"]: "orange"})
        st.plotly_chart(fig, use_container_width=True)

        # Gemini Tư vấn thêm
        if st.button("Hỏi Gemini về xu hướng này"):
            with st.spinner("Gemini đang phân tích nhịp cầu..."):
                prompt = f"Dữ liệu Lotobet 10 kỳ gần: {df['numbers'].tail(10).tolist()}. Phân tích nhịp số đơn và đưa ra lời khuyên quản lý vốn."
                try:
                    response = model.generate_content(prompt)
                    st.write(response.text)
                except:
                    st.warning("Gemini đang bận, anh hãy dựa vào kết quả Thuật toán bên trên.")

if __name__ == "__main__":
    main()
