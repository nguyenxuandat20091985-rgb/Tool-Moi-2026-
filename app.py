import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
from datetime import datetime
import os

# ================= CONFIG & API =================
st.set_page_config(page_title="AI LOTOBET 2-TINH V2", layout="wide", page_icon="🎯")

# Kết nối Gemini của anh
API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
genai.configure(api_key=API_KEY)

DATA_FILE = "lotobet_history.csv"

# ================= CORE ENGINE (LOGIC V2) =================
class LotobetEngine:
    def __init__(self):
        self.banned_pairs = [f"{i}{i}" for i in range(10)] # 00, 11... 99

    def clean_data(self, df):
        """Xử lý lỗi TypeError/ValueError bằng cách làm sạch dữ liệu"""
        if df.empty: return pd.DataFrame()
        # Chỉ lấy những dòng có đúng 5 ký số
        df['numbers'] = df['numbers'].astype(str).str.replace(r'\D', '', regex=True)
        df = df[df['numbers'].str.len() == 5]
        return df

    def analyze_single_numbers(self, df):
        """Phân tích số đơn 0-9 theo đặc tả mục 3"""
        if len(df) < 5: return None
        
        # Chuyển thành matrix an toàn
        try:
            matrix = np.array([[int(d) for d in str(s)] for s in df['numbers'].values])
        except:
            return None

        analysis = {}
        for num in range(10):
            # Tìm các kỳ xuất hiện
            appears = np.where(np.any(matrix == num, axis=1))[0]
            last_idx = appears[-1] if len(appears) > 0 else -1
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # Tính tần suất gần (3, 5, 10 kỳ)
            freq_3 = np.sum(np.any(matrix[-3:] == num, axis=1))
            freq_10 = np.sum(np.any(matrix[-10:] == num, axis=1))
            
            # Gán nhãn trạng thái (Mục 6)
            state = "ỔN ĐỊNH"
            if freq_3 >= 2: state = "NGUY HIỂM" # Ra dồn
            elif freq_10 >= 5: state = "NÓNG"
            elif freq_10 <= 1: state = "YẾU"
            
            # Nhận diện cầu (Mục 4)
            bridge = "NORMAL"
            if len(gaps) >= 2 and gaps[-1] == gaps[-2] and gaps[-1] > 1: bridge = "NHẢY"
            if len(df) - 1 == last_idx: bridge = "LẶP"

            analysis[num] = {
                "state": state,
                "bridge": bridge,
                "freq_10": freq_10,
                "last_seen": len(df) - 1 - last_idx
            }
        return analysis

    def get_predictions(self, df, analysis):
        """Logic ghép cặp & KHÔNG ĐÁNH (Mục 7, 8)"""
        if not analysis: return [], "SKIP", "Dữ liệu không đủ"
        
        # Kiểm tra điều kiện "KHÔNG ĐÁNH"
        hot_counts = sum(1 for v in analysis.values() if v['state'] in ["NÓNG", "NGUY HIỂM"])
        if hot_counts > 6:
            return [], "SKIP", "Thị trường quá NÓNG (nhiễu), nguy cơ gãy cầu cao."

        potential_pairs = []
        for i in range(10):
            for j in range(i+1, 10):
                s1, s2 = analysis[i], analysis[j]
                
                # Loại trừ theo mục 6
                if s1['state'] == s2['state'] and s1['state'] in ["NÓNG", "NGUY HIỂM", "YẾU"]:
                    continue
                
                # Ưu tiên ghép (Mục 7)
                score = 50
                if (s1['state'] == "ỔN ĐỊNH" and s2['last_seen'] >= 5): score += 30 # Ổn định + Hồi
                if (s1['bridge'] == "NHẢY" and s2['state'] == "ỔN ĐỊNH"): score += 25
                
                # Giảm trọng số cầu Lặp (Mục 4C)
                if s1['bridge'] == "LẶP" or s2['bridge'] == "LẶP": score -= 20

                if score >= 75:
                    potential_pairs.append({"pair": f"{i}{j}", "score": score})

        potential_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        if not potential_pairs or potential_pairs[0]['score'] < 75:
            return [], "SKIP", "Không có cặp số đạt ngưỡng an toàn (≥75%)"
            
        return potential_pairs[:1], "PREDICT", "" # Tối đa 1 cặp duy nhất (Mục 7)

# ================= GEMINI AI INTEGRATION =================
def ask_gemini(history_str, recommendation):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Bạn là chuyên gia xác suất Lotobet. Dữ liệu 10 kỳ gần nhất: {history_str}.
        Thuật toán toán học đề xuất: {recommendation}.
        Dựa trên Đặc tả Logic v2: 
        1. Tuyệt đối không cho số chập.
        2. Nếu thấy dấu hiệu 'nhà cái lừa cầu' hoặc dữ liệu nhiễu, hãy trả về 'KHÔNG ĐÁNH'.
        3. Phân tích ngắn gọn tối đa 50 từ.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Gemini đang bận xử lý dữ liệu..."

# ================= MAIN APP =================
def main():
    engine = LotobetEngine()
    
    st.title("🎯 AI LOTOBET 2-TINH PRO (BẢN CHUẨN v2)")
    st.caption("Nguyên Xuân Đạt - Hệ thống phân tích chính xác cao")

    # Load dữ liệu
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=["time", "numbers"])

    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📥 Nhập dữ liệu"])

    with tab2:
        st.subheader("📥 Cập nhật dữ liệu sạch")
        raw_input = st.text_area("Nhập 5 số viết liền (mỗi dòng 1 kỳ):", height=200)
        col_btn1, col_btn2 = st.columns(2)
        
        if col_btn1.button("💾 Lưu dữ liệu"):
            lines = [n.strip() for n in raw_input.split("\n") if len(n.strip()) == 5]
            if lines:
                new_data = pd.DataFrame({"time": [datetime.now().strftime("%H:%M:%S")] * len(lines), "numbers": lines})
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã lưu {len(lines)} kỳ mới!")
                st.rerun()
        
        if col_btn2.button("🗑 Xóa lịch sử"):
            if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
            st.rerun()

    with tab1:
        df = engine.clean_data(df)
        if len(df) < 10:
            st.warning(f"Cần thêm dữ liệu (Hiện có: {len(df)}/10 kỳ).")
            return

        # Phân tích
        analysis = engine.analyze_single_numbers(df)
        preds, status, reason = engine.get_predictions(df, analysis)

        # Hiển thị Dashboard
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Tổng số kỳ", len(df))
        col_m2.metric("Số đang NÓNG", sum(1 for v in analysis.values() if v['state'] == "NÓNG"))
        col_m3.metric("Số đang YẾU", sum(1 for v in analysis.values() if v['state'] == "YẾU"))

        st.divider()

        if status == "SKIP":
            st.error(f"🚫 KHÔNG ĐÁNH KỲ NÀY")
            st.info(f"Lý do: {reason}")
        else:
            res = preds[0]
            st.success(f"✅ AI ĐỀ XUẤT CẶP 2 TINH: {res['pair']}")
            st.subheader(f"Độ tự tin: {res['score']}%")
            
            # Gọi Gemini AI
            with st.spinner('Gemini AI đang thẩm định...'):
                history_str = ", ".join(df['numbers'].tail(10).tolist())
                gemini_review = ask_gemini(history_str, res['pair'])
                st.info(f"🤖 Trợ lý Gemini thẩm định: {gemini_review}")

        # Biểu đồ trạng thái số đơn
        st.subheader("📊 Trạng thái số đơn (0-9)")
        chart_df = pd.DataFrame([{"Số": k, "Tần suất (10 kỳ)": v['freq_10'], "Trạng thái": v['state']} for k, v in analysis.items()])
        fig = px.bar(chart_df, x="Số", y="Tần suất (10 kỳ)", color="Trạng thái", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
