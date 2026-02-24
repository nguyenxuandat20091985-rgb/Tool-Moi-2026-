import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIG & API =================
API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
genai.configure(api_key=API_KEY)

st.set_page_config(page_title="AI LOTOBET 2-TINH (BẢN CHUẨN v2)", layout="wide", page_icon="🎯")

DATA_FILE = "lotobet_history.csv"

# ================= ENGINE CHUẨN V2 =================
class LotobetEngineV2:
    def __init__(self):
        self.state_labels = {
            "HOT": "🔥 NÓNG",
            "STABLE": "✅ ỔN ĐỊNH",
            "WEAK": "❄️ YẾU",
            "RISKY": "⚠️ NGUY HIỂM"
        }

    def clean_data(self, raw_text):
        """Lọc dữ liệu: Chỉ lấy dòng đúng 5 số"""
        valid_rows = []
        lines = raw_text.split('\n')
        for line in lines:
            clean_line = "".join(filter(str.isdigit, line.strip()))
            if len(clean_line) == 5:
                valid_rows.append(clean_line)
        return valid_rows

    def analyze_numbers(self, df):
        """Phân tích 10 số đơn (0-9) theo ma trận"""
        if df.empty: return None
        
        # Chuyển list số thành ma trận numpy để tránh lỗi ValueError
        try:
            matrix = []
            for s in df['numbers'].tolist():
                matrix.append([int(d) for d in str(s)])
            matrix = np.array(matrix)
        except Exception:
            return None

        analysis = {}
        total_kỳ = len(matrix)
        
        for n in range(10):
            # 1. Tìm vị trí xuất hiện
            appears = np.where(np.any(matrix == n, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # 2. Tần suất 10 kỳ gần nhất
            recent_10 = matrix[-10:]
            count_10 = sum(1 for row in recent_10 if n in row)
            
            # 3. Phân loại trạng thái (Logic Mục 6)
            if count_10 >= 6: state = "HOT"
            elif count_10 >= 3: state = "STABLE"
            elif count_10 >= 1: state = "RISKY" if (len(appears) > 0 and (total_kỳ-1-appears[-1]) <= 1) else "WEAK"
            else: state = "WEAK"

            analysis[n] = {
                "state": state,
                "count_10": count_10,
                "last_appear": (total_kỳ - 1 - appears[-1]) if len(appears) > 0 else 99,
                "avg_gap": np.mean(gaps) if len(gaps) > 0 else 0
            }
        return analysis

    def get_prediction(self, analysis, df):
        """Logic ghép cặp (Mục 7 & 8)"""
        if not analysis: return None, "DỮ LIỆU LỖI", []

        reasons_to_skip = []
        
        # Kiểm tra điều kiện "Không đánh"
        hot_nums = [n for n, v in analysis.items() if v['state'] == "HOT"]
        if len(hot_nums) >= 6:
            reasons_to_skip.append("Toàn số quá nóng (Thị trường biến động mạnh)")

        # Lấy kỳ cuối để kiểm tra cầu lặp
        last_draw = [int(d) for d in str(df.iloc[-1]['numbers'])]
        
        # Ghép cặp (Loại số chập)
        candidates = []
        for i in range(10):
            for j in range(i + 1, 10): # j luôn > i -> Không bao giờ bị số chập (11, 22...)
                s1, s2 = analysis[i], analysis[j]
                
                # Rule Loại trừ (Mục 6)
                if s1['state'] == s2['state'] and s1['state'] in ["HOT", "RISKY", "WEAK"]:
                    continue
                
                # Tính điểm tự tin
                score = 50
                if s1['state'] == "STABLE" or s2['state'] == "STABLE": score += 20
                if s1['last_appear'] in range(5, 8): score += 15 # Cầu hồi tốt
                
                candidates.append({"pair": f"{i}{j}", "score": score})

        if not candidates or len(reasons_to_skip) > 0:
            return None, "KHÔNG ĐÁNH KỲ NÀY", reasons_to_skip

        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_pair = candidates[0]

        if best_pair['score'] < 60:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Độ tin cậy thấp hơn 60%"]

        return best_pair, "PREDICT", []

# ================= GEMINI ADVISOR =================
def ask_gemini(df_tail, prediction):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Bạn là chuyên gia phân tích Lotobet. 
        Dữ liệu 5 kỳ gần nhất: {df_tail}. 
        AI đề xuất cặp: {prediction}. 
        Dựa trên thuyết bóng số và nhịp cầu, hãy đưa ra nhận định ngắn gọn dưới 50 chữ về cặp này.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Gemini đang bận xử lý nhịp cầu..."

# ================= INTERFACE =================
def main():
    st.markdown("<h1 style='text-align: center; color: #E74C3C;'>🎯 AI LOTOBET 2-TINH CHUẨN v2</h1>", unsafe_allow_html=True)
    
    engine = LotobetEngineV2()
    
    # Load data
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=["time", "numbers"])

    menu = ["📊 Dự đoán & Thống kê", "📥 Nhập dữ liệu"]
    choice = st.sidebar.selectbox("MENU", menu)

    if choice == "📥 Nhập dữ liệu":
        st.subheader("📥 Cập nhật dữ liệu sạch")
        raw_data = st.text_area("Nhập 5 số viết liền (mỗi dòng 1 kỳ):", height=200)
        if st.button("Lọc & Lưu dữ liệu"):
            clean_list = engine.clean_data(raw_data)
            if clean_list:
                new_df = pd.DataFrame({"time": [datetime.now().strftime("%H:%M:%S")]*len(clean_list), "numbers": clean_list})
                df = pd.concat([df, new_df], ignore_index=True).tail(1000) # Giữ tối đa 1000 kỳ
                df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã lưu {len(clean_list)} kỳ hợp lệ!")
            else:
                st.error("Dữ liệu không hợp lệ (Phải là dãy 5 chữ số)")

    elif choice == "📊 Dự đoán & Thống kê":
        if len(df) < 15:
            st.warning("Cần tối nhất 15 kỳ để phân tích chính xác.")
            return

        analysis = engine.analyze_numbers(df)
        
        # Hiển thị trạng thái số (Mục 6)
        st.write("### 📈 Trạng thái dòng số đơn")
        cols = st.columns(5)
        for i in range(10):
            with cols[i % 5]:
                data = analysis[i]
                color = "red" if "NÓNG" in engine.state_labels[data['state']] else "black"
                st.markdown(f"**Số {i}**: <span style='color:{color}'>{engine.state_labels[data['state']]}</span>", unsafe_allow_html=True)

        st.divider()

        # Dự đoán
        best_pair, status, reasons = engine.get_prediction(analysis, df)
        
        if status == "SKIP":
            st.error("🚫 KHÔNG ĐÁNH KỲ NÀY")
            for r in reasons: st.write(f"- {r}")
        else:
            st.success(f"🚀 CẶP SỐ TIỀM NĂNG: {best_pair['pair']}")
            
            c1, c2 = st.columns(2)
            c1.metric("Độ tự tin AI", f"{best_pair['score']}%")
            
            # Kết nối Gemini
            with st.spinner("Gemini đang soi bóng số..."):
                recent_history = df['numbers'].tail(5).tolist()
                advice = ask_gemini(recent_history, best_pair['pair'])
                st.info(f"💡 **Nhận định chuyên gia:** {advice}")

        # Thống kê kỳ gần nhất
        with st.expander("Xem lịch sử 10 kỳ gần nhất"):
            st.table(df.tail(10))

if __name__ == "__main__":
    main()
