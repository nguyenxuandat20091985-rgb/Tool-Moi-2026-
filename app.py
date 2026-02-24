import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIG & API =================
st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide", page_icon="🎯")

# Cấu hình Gemini (Dùng API anh cung cấp)
try:
    genai.configure(api_key="AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE")
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("API Gemini đang bận hoặc sai key. App sẽ dùng Logic Offline.")

DATA_FILE = "lotobet_history.csv"

# ================= CORE AI ENGINE V2 =================
class LotobetV2:
    def __init__(self):
        self.states = {
            "NÓNG": "🔥 Nóng",
            "ỔN ĐỊNH": "✅ Ổn định", 
            "YẾU": "❄️ Yếu",
            "NGUY HIỂM": "⚠️ Nguy hiểm"
        }

    def clean_data(self, df):
        """Xử lý dữ liệu thô, loại bỏ dòng lỗi"""
        valid_matrix = []
        for val in df['numbers'].values:
            s_val = str(val).strip()
            if len(s_val) == 5 and s_val.isdigit():
                valid_matrix.append([int(d) for d in s_val])
        return np.array(valid_matrix)

    def analyze_numbers(self, matrix):
        """Phân tích từng số đơn từ 0-9 theo đặc tả"""
        if len(matrix) < 5: return None
        
        analysis = {}
        for num in range(10):
            # Tìm các kỳ xuất hiện (index)
            appears = np.where(np.any(matrix == num, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # 1. Tần suất xuất hiện
            recent_5 = sum(1 for row in matrix[-5:] if num in row)
            recent_10 = sum(1 for row in matrix[-10:] if num in row)
            last_appearance = (len(matrix) - 1 - appears[-1]) if len(appears) > 0 else 99
            
            # 2. Gán trạng thái (Theo mục 6 Đặc tả)
            state = "ỔN ĐỊNH"
            if recent_5 >= 4: state = "NGUY HIỂM" # Ra quá dày
            elif recent_5 >= 3: state = "NÓNG"
            elif recent_10 <= 1: state = "YẾU"
            
            # 3. Nhận diện cầu (Theo mục 4 Đặc tả)
            bridge = "BÌNH THƯỜNG"
            if len(gaps) >= 2 and gaps[-1] == gaps[-2] and gaps[-1] in [2, 3]:
                bridge = "CẦU NHẢY"
            elif last_appearance == 0:
                bridge = "CẦU LẶP"
            elif 5 <= last_appearance <= 8:
                bridge = "CẦU HỒI"
            elif len(gaps) >= 3 and np.all(gaps[-3:] == 1):
                bridge = "CẦU BỆT"

            analysis[num] = {
                "state": state,
                "bridge": bridge,
                "last_app": last_appearance,
                "freq_5": recent_5,
                "score": self.calculate_score(state, bridge, last_appearance)
            }
        return analysis

    def calculate_score(self, state, bridge, last_app):
        """Tính điểm tin cậy cho từng số đơn"""
        score = 50
        if bridge == "CẦU NHẢY": score += 20
        if bridge == "CẦU HỒI": score += 15
        if state == "ỔN ĐỊNH": score += 10
        if state == "NGUY HIỂM" or bridge == "CẦU LẶP": score -= 30
        if last_app > 10: score -= 20
        return score

    def get_final_prediction(self, analysis):
        """Logic ghép cặp & Lọc số chập (Mục 1, 7, 8)"""
        if not analysis: return None, "DỮ LIỆU NHIỄU"

        candidates = []
        # Chỉ lấy số có điểm tốt
        for num, data in analysis.items():
            if data['score'] >= 60:
                candidates.append(num)
        
        # Sắp xếp theo điểm
        candidates.sort(key=lambda x: analysis[x]['score'], reverse=True)
        
        # Logic KHÔNG ĐÁNH
        hot_count = sum(1 for d in analysis.values() if d['state'] == "NGUY HIỂM")
        if hot_count >= 6 or len(candidates) < 2:
            return None, "KHÔNG ĐÁNH KỲ NÀY (Cầu nhiễu/Quá nóng)"

        # Ghép cặp (TỐI ĐA 1 CẶP - Mục 7)
        # Loại số chập tự động vì ghép từ 2 số đơn khác nhau (i, j)
        best_pair = None
        max_conf = 0
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                n1, n2 = candidates[i], candidates[j]
                
                # Kiểm tra quy tắc loại trừ (Mục 6)
                s1, s2 = analysis[n1]['state'], analysis[n2]['state']
                if s1 == s2 and s1 in ["NGUY HIỂM", "NÓNG", "YẾU"]:
                    continue
                
                conf = (analysis[n1]['score'] + analysis[n2]['score']) / 2
                if conf > max_conf:
                    max_conf = conf
                    best_pair = (n1, n2)

        if max_conf < 60:
            return None, "KHÔNG ĐÁNH KỲ NÀY (Độ tự tin thấp)"
            
        return {"pair": best_pair, "conf": int(max_conf)}, "OK"

# ================= UI LAYOUT =================
def main():
    st.title("🎯 AI LOTOBET 2-TINH (BẢN CHUẨN v2)")
    engine = LotobetV2()
    
    # --- PHẦN NHẬP LIỆU ---
    with st.expander("📥 Nhập dữ liệu hệ thống", expanded=not os.path.exists(DATA_FILE)):
        raw_input = st.text_area("Nhập kết quả (5 số viết liền, mỗi dòng 1 kỳ):", height=150)
        if st.button("💾 Lưu và Phân tích"):
            lines = [line.strip() for line in raw_input.split('\n') if len(line.strip()) == 5 and line.strip().isdigit()]
            if lines:
                new_df = pd.DataFrame(lines, columns=['numbers'])
                new_df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã lưu {len(lines)} kỳ!")
                st.rerun()
            else:
                st.error("Dữ liệu không hợp lệ. Vui lòng nhập đúng 5 số mỗi dòng.")

    # --- PHẦN HIỂN THỊ KẾT QUẢ ---
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        matrix = engine.clean_data(df)
        
        if len(matrix) < 10:
            st.warning("Cần tối thiểu 10 kỳ để phân tích chính xác.")
            return

        analysis = engine.analyze_numbers(matrix)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Trạng thái số")
            # Hiển thị bảng trạng thái đơn giản
            status_df = pd.DataFrame([
                {"Số": i, "Trạng thái": analysis[i]['state'], "Cầu": analysis[i]['bridge']}
                for i in range(10)
            ])
            st.table(status_df)

        with col2:
            st.subheader("🔮 Dự đoán AI")
            pred, status = engine.get_final_prediction(analysis)
            
            if pred and status == "OK":
                conf = pred['conf']
                color = "green" if conf >= 75 else "orange"
                
                st.markdown(f"""
                <div style="text-align:center; padding:20px; border:2px solid {color}; border-radius:10px;">
                    <h1 style="color:{color}; font-size: 50px;">{pred['pair'][0]}{pred['pair'][1]}</h1>
                    <h3>Độ tự tin: {conf}%</h3>
                    <p>(Đánh cặp 2 số không cố định vị trí)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gọi Gemini nhận xét
                if st.button("🤖 Hỏi ý kiến Gemini về cặp này"):
                    with st.spinner("Gemini đang phân tích nhịp cầu..."):
                        prompt = f"Kết quả Lotobet 10 kỳ gần nhất: {matrix[-10:].tolist()}. AI đề xuất cặp {pred['pair']}. Dựa trên đặc tả cầu nhảy, cầu hồi, hãy nhận xét ngắn gọn về cặp này."
                        try:
                            response = model.generate_content(prompt)
                            st.info(response.text)
                        except:
                            st.error("Gemini đang bận, bạn hãy dựa vào Độ tự tin của AI.")
            else:
                st.error(f"🚫 {status}")

        # Hiển thị lịch sử gần đây
        with st.expander("🕒 Lịch sử 10 kỳ gần nhất"):
            st.write(df.tail(10))

if __name__ == "__main__":
    main()
