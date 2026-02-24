import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIG & API =================
st.set_page_config(page_title="AI LOTOBET V2 - CHUẨN", layout="wide")
API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE" # API của anh
DATA_FILE = "data_clean.csv"

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("Lỗi cấu hình Gemini API. Kiểm tra lại kết nối mạng.")

# ================= AI ENGINE (ĐÚNG ĐẶC TẢ) =================
class LotobetV2:
    def __init__(self):
        self.MIN_DRAWS = 15
        
    def clean_input(self, text):
        """Lọc dữ liệu rác, chỉ lấy dòng đúng 5 chữ số"""
        lines = text.split('\n')
        clean_data = []
        for line in lines:
            s = line.strip()
            if s.isdigit() and len(s) == 5:
                clean_data.append(s)
        return clean_data

    def analyze_numbers(self, df):
        """Phân tích số đơn 0-9 theo ma trận"""
        if df.empty: return None
        
        # Chuyển series thành list of lists an toàn
        raw_list = df['numbers'].astype(str).tolist()
        matrix = []
        for s in raw_list:
            if len(s) == 5:
                matrix.append([int(d) for d in s])
        
        if not matrix: return None
        matrix = np.array(matrix)
        
        analysis = {}
        for n in range(10):
            # Vị trí xuất hiện (kỳ)
            appears = np.where(np.any(matrix == n, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            recent_5 = matrix[-5:]
            freq_5 = sum(1 for row in recent_5 if n in row)
            
            # Gán trạng thái theo đặc tả
            if freq_5 >= 4: state = "NÓNG"
            elif freq_5 == 0: state = "YẾU"
            elif len(gaps) > 0 and gaps[-1] == 1: state = "NGUY HIỂM"
            else: state = "ỔN ĐỊNH"
            
            analysis[n] = {
                "freq": freq_5,
                "state": state,
                "last_idx": appears[-1] if len(appears) > 0 else -1
            }
        return analysis

    def get_prediction(self, analysis, df):
        """Logic ghép cặp & KHÔNG ĐÁNH"""
        if not analysis: return None, "DỮ LIỆU LỖI", []
        
        reasons_skip = []
        # 1. Kiểm tra cầu nóng toàn diện
        hot_count = sum(1 for v in analysis.values() if v['state'] == "NÓNG")
        if hot_count >= 6: reasons_skip.append("Toàn số quá NÓNG (Thị trường nhiễu)")
        
        # 2. Kiểm tra dữ liệu ít
        if len(df) < self.MIN_DRAWS: reasons_skip.append(f"Dữ liệu ít ({len(df)}/{self.MIN_DRAWS} kỳ)")

        if reasons_skip:
            return None, "KHÔNG ĐÁNH KỲ NÀY", reasons_skip

        # 3. Ghép cặp (Ưu tiên 1 cặp duy nhất)
        candidates = []
        for i in range(10):
            for j in range(i + 1, 10):
                s1, s2 = analysis[i], analysis[j]
                
                # Loại số chập (Đặc tả 1) - i đã khác j nên không bao giờ trùng
                # Điều kiện loại trừ (Đặc tả 6)
                invalid_states = ["NÓNG", "NGUY HIỂM", "YẾU"]
                if s1['state'] in invalid_states and s2['state'] in invalid_states:
                    continue
                
                # Điểm ưu tiên (Đặc tả 7)
                score = 50
                if s1['state'] == "ỔN ĐỊNH" and s2['state'] == "ỔN ĐỊNH": score += 30
                if s1['state'] == "ỔN ĐỊNH" and s2['state'] == "YẾU": score += 10
                
                candidates.append({"pair": f"{i}{j}", "score": score})

        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if not candidates or candidates[0]['score'] < 60:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Không có cặp đạt ngưỡng an toàn"]

        return candidates[0], "PREDICT", []

# ================= UI STREAMLIT =================
def main():
    st.markdown("<h1 style='text-align: center; color: #E74C3C;'>🎯 AI LOTOBET 2-TINH PRO V2</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Bản chuẩn đặc tả - Ưu tiên bảo toàn vốn</p>", unsafe_allow_html=True)

    engine = LotobetV2()
    
    # Load Data
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["numbers"]).to_csv(DATA_FILE, index=False)
    df = pd.read_csv(DATA_FILE)

    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📥 Nhập dữ liệu"])

    with tab2:
        st.subheader("📥 Cập nhật dữ liệu sạch")
        input_text = st.text_area("Nhập 5 số viết liền (mỗi dòng 1 kỳ):", height=200)
        col_btn1, col_btn2 = st.columns(2)
        
        if col_btn1.button("💾 Lưu & Làm sạch"):
            clean_list = engine.clean_input(input_text)
            if clean_list:
                new_df = pd.DataFrame({"numbers": clean_list})
                df = pd.concat([df, new_df], ignore_index=True).drop_duplicates()
                df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã lưu {len(clean_list)} kỳ hợp lệ!")
                st.rerun()
        
        if col_btn2.button("🗑 Xóa hết dữ liệu"):
            pd.DataFrame(columns=["numbers"]).to_csv(DATA_FILE, index=False)
            st.warning("Đã xóa toàn bộ lịch sử.")
            st.rerun()

    with tab1:
        if df.empty:
            st.info("Vui lòng sang tab Nhập liệu để bắt đầu.")
            return

        analysis = engine.analyze_numbers(df)
        
        # Dashboard nhanh
        c1, c2, c3 = st.columns(3)
        c1.metric("Kỳ đã nhập", len(df))
        if analysis:
            hot_s = sum(1 for v in analysis.values() if v['state'] == "NÓNG")
            c2.metric("Số đang NÓNG", hot_s)
            c3.metric("Độ nhiễu", "Cao" if hot_s > 5 else "Thấp")

        st.divider()

        # DỰ ĐOÁN CHÍNH
        prediction, status, reasons = engine.get_prediction(analysis, df)

        if status == "KHÔNG ĐÁNH KỲ NÀY":
            st.error("🚫 **DỪNG LẠI: KHÔNG ĐÁNH KỲ NÀY**")
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.success(f"✅ **CẶP SỐ ĐỀ XUẤT: {prediction['pair']}**")
            st.write(f"Độ tin cậy: **{prediction['score']}%**")
            
            # Gọi Gemini hỗ trợ phân tích tâm lý (Tránh API bận bằng cách tóm tắt)
            if st.button("🤖 Hỏi Gemini về nhịp cầu này"):
                with st.spinner("Đang kết nối trí tuệ nhân tạo..."):
                    try:
                        prompt = f"Dữ liệu lotobet 5 kỳ gần: {df['numbers'].tail(5).tolist()}. AI đề xuất cặp {prediction['pair']} với độ tin cậy {prediction['score']}%. Hãy nhận định ngắn gọn về nhịp cầu này dưới góc độ toán học xác suất."
                        response = model.generate_content(prompt)
                        st.info(response.text)
                    except:
                        st.warning("Gemini đang bận xử lý dữ liệu khác. Hãy thử lại sau 1 phút.")

        # Thống kê chi tiết
        with st.expander("📊 Xem chi tiết trạng thái 0-9"):
            if analysis:
                stat_df = pd.DataFrame([{"Số": k, "Trạng thái": v['state'], "Tần suất (5 kỳ)": v['freq']} for k, v in analysis.items()])
                st.table(stat_df)

if __name__ == "__main__":
    main()
