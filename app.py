import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIG & API =================
st.set_page_config(page_title="AI LOTOBET 2-TINH PRO v2", layout="wide")

# Kết nối Gemini (Dùng API anh cung cấp)
genai.configure(api_key="AIzaSyAGl5dArirOAoRCRK2qHMcupWBcyt5ZmZU")
model = genai.GenerativeModel('gemini-1.5-flash')

DATA_FILE = "lotobet_history.csv"

# ================= CORE LOGIC AI =================
class LotobetStandardAI:
    def __init__(self):
        self.min_draws = 15
        self.labels = {
            "HOT": "🔥 NÓNG",
            "STABLE": "✅ ỔN ĐỊNH",
            "WEAK": "📉 YẾU",
            "RISKY": "⚠️ NGUY HIỂM"
        }

    def clean_matrix(self, df):
        """Chuyển dữ liệu thô thành ma trận số đơn chuẩn 5 cột"""
        matrix = []
        for val in df['numbers'].astype(str):
            nums = [int(d) for d in val if d.isdigit()]
            if len(nums) == 5:
                matrix.append(nums)
        return np.array(matrix)

    def analyze_numbers(self, matrix):
        """Bước 3: Phân tích từng số đơn 0-9"""
        analysis = {}
        total_len = len(matrix)
        
        for num in range(10):
            # Vị trí các kỳ xuất hiện
            appears = np.where(np.any(matrix == num, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # Tần suất trong các mốc thời gian
            recent_3 = sum(1 for row in matrix[-3:] if num in row)
            recent_5 = sum(1 for row in matrix[-5:] if num in row)
            recent_10 = sum(1 for row in matrix[-10:] if num in row)
            
            last_appear_idx = appears[-1] if len(appears) > 0 else -1
            dist_from_last = total_len - 1 - last_appear_idx

            # Phân loại trạng thái (Bước 6)
            if recent_3 >= 2: state = "RISKY"  # Vừa ra dồn
            elif recent_5 >= 3: state = "HOT"   # Ra dày
            elif 2 <= recent_10 <= 4: state = "STABLE" # Ra đều
            else: state = "WEAK"

            # Nhận diện loại cầu (Bước 4)
            bridge = "NORMAL"
            if len(gaps) >= 2:
                if gaps[-1] == 1 and gaps[-2] == 1: bridge = "BỆT"
                elif 2 <= gaps[-1] <= 3 and gaps[-1] == gaps[-2]: bridge = "NHẢY"
            
            analysis[num] = {
                "state": state,
                "bridge": bridge,
                "dist": dist_from_last,
                "freq_10": recent_10,
                "score": 0 # Sẽ tính sau
            }
        return analysis

    def get_predictions(self, df):
        """Logic ghép cặp và lọc số (Bước 7 & 8)"""
        if len(df) < self.min_draws:
            return None, "DỮ LIỆU THẤP", [f"Cần tối thiểu {self.min_draws} kỳ"]

        matrix = self.clean_matrix(df)
        if len(matrix) < 5: return None, "LỖI DỮ LIỆU", ["Định dạng số không chuẩn"]
        
        analysis = self.analyze_numbers(matrix)
        
        # Chấm điểm ưu tiên (Bước 5 & 7)
        scored_numbers = []
        for num, data in analysis.items():
            score = 50
            if data['state'] == "STABLE": score += 20
            if data['bridge'] == "NHẢY": score += 15
            if 5 <= data['dist'] <= 8: score += 20 # Cầu hồi tốt
            
            # Trừ điểm (Bước 5)
            if data['dist'] == 0: score -= 40 # Vừa ra kỳ trước
            if data['state'] == "RISKY": score -= 30
            if data['dist'] > 12: score -= 20 # Quá lâu (nhiễu)
            
            data['score'] = score
            scored_numbers.append((num, score))

        # Loại bỏ 3 số điểm thấp nhất, giữ lại 7 số (Yêu cầu của anh)
        scored_numbers.sort(key=lambda x: x[1], reverse=True)
        top_7 = [x[0] for x in scored_numbers[:7]]
        
        # Ghép cặp (Bước 1: Loại số chập)
        candidates = []
        for i in range(len(top_7)):
            for j in range(i + 1, len(top_7)):
                n1, n2 = top_7[i], top_7[j]
                
                # Logic loại trừ: Không ghép 2 số đều nóng/yếu
                s1, s2 = analysis[n1]['state'], analysis[n2]['state']
                if s1 == s2 and s1 in ["HOT", "RISKY", "WEAK"]: continue
                
                avg_score = (analysis[n1]['score'] + analysis[n2]['score']) / 2
                candidates.append(((n1, n2), avg_score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Logic KHÔNG ĐÁNH (Bước 8)
        reasons = []
        hot_count = sum(1 for v in analysis.values() if v['state'] in ["HOT", "RISKY"])
        if hot_count >= 6: reasons.append("Thị trường quá NÓNG (nhiều số ra dồn)")
        if not candidates or candidates[0][1] < 60: reasons.append("Độ tự tin dưới 60%")
        
        if reasons:
            return None, "KHÔNG ĐÁNH KỲ NÀY", reasons

        return candidates[:1], "PREDICT", [] # Trả về 1 cặp duy nhất tốt nhất

# ================= UI STREAMLIT =================
def main():
    st.markdown("<h1 style='text-align: center; color: #E74C3C;'>🎯 AI LOTOBET 2-TINH PRO v2</h1>", unsafe_allow_html=True)
    ai = LotobetStandardAI()

    # Quản lý dữ liệu
    if 'data' not in st.session_state:
        if os.path.exists(DATA_FILE):
            st.session_state.data = pd.read_csv(DATA_FILE)
        else:
            st.session_state.data = pd.DataFrame(columns=["time", "numbers"])

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("📥 Nhập dữ liệu")
        raw_input = st.text_area("Nhập kết quả (5 số liền nhau, mỗi kỳ 1 dòng):", height=200)
        if st.button("💾 Cập nhật hệ thống"):
            if raw_input:
                new_entries = [n.strip() for n in raw_input.split("\n") if len(n.strip()) == 5]
                if new_entries:
                    new_df = pd.DataFrame({"time": [datetime.now().strftime("%H:%M:%S")]*len(new_entries), "numbers": new_entries})
                    st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True).tail(100)
                    st.session_state.data.to_csv(DATA_FILE, index=False)
                    st.success(f"Đã lưu {len(new_entries)} kỳ mới")
                    st.rerun()

    with col_out:
        st.subheader("📊 Phân tích & Dự đoán")
        df = st.session_state.data
        if len(df) < 5:
            st.warning("Vui lòng nhập thêm dữ liệu để bắt đầu phân tích.")
            return

        # Gọi AI Phân tích
        preds, status, reasons = ai.get_predictions(df)

        if status == "KHÔNG ĐÁNH KỲ NÀY":
            st.error("🚫 **KHÔNG ĐÁNH KỲ NÀY**")
            for r in reasons: st.write(f"- {r}")
        elif status == "PREDICT":
            pair = preds[0][0]
            confidence = preds[0][1]
            
            # Hiển thị kết quả rực rỡ
            st.balloons()
            st.markdown(f"""
                <div style="background: #2ECC71; padding: 30px; border-radius: 15px; text-align: center;">
                    <h1 style="color: white; font-size: 50px; margin: 0;">{pair[0]}{pair[1]}</h1>
                    <p style="color: white; font-size: 20px;">Độ tự tin: {confidence:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

            # --- KẾT NỐI GEMINI ĐỂ NHẬN XÉT ---
            try:
                prompt = f"Kết quả Lotobet gần đây: {df['numbers'].tail(5).tolist()}. AI đề xuất cặp {pair[0]}{pair[1]} với độ tin cậy {confidence}%. Hãy đưa ra lời khuyên ngắn gọn cho người chơi bằng tiếng Việt."
                response = model.generate_content(prompt)
                st.info(f"🤖 **Trợ lý Gemini:** {response.text}")
            except:
                st.caption("Gemini đang bận, vui lòng kiểm tra API Key.")

        # Biểu đồ trạng thái số đơn
        st.divider()
        matrix = ai.clean_matrix(df)
        analysis = ai.analyze_numbers(matrix)
        chart_data = pd.DataFrame([{"Số": k, "Nhịp": v['dist'], "Trạng thái": v['state']} for k, v in analysis.items()])
        fig = px.bar(chart_data, x="Số", y="Nhịp", color="Trạng thái", title="Khoảng cách kỳ chưa ra (Nhịp hồi)")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
