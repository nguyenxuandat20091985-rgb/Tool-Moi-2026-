import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
from datetime import datetime
import os

# ================= CONFIG & API GEMINI =================
API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
genai.configure(api_key=API_KEY)

st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide")

# Hàm gọi Gemini để phân tích sâu (Bóng số & Tâm lý nhà cái)
def ask_gemini_pro(history_str, stats_summary):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Bạn là chuyên gia xác suất Lotobet. Dữ liệu 5 kỳ gần nhất: {history_str}.
        Thống kê trạng thái: {stats_summary}.
        Dựa trên quy luật bóng số (0-5, 1-6, 2-7, 3-8, 4-9) và nhịp cầu, 
        hãy cho biết 1 cặp 2-tinh (2 số khác nhau) có khả năng về cao nhất. 
        Chỉ trả về 1 dòng duy nhất gồm cặp số và lý do ngắn gọn.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Gemini đang bận, sử dụng thuật toán cục bộ..."

# ================= CORE ENGINE v2 =================
class LotoEngineV2:
    def __init__(self):
        self.DATA_FILE = "lotobet_v2.csv"

    def clean_data(self, raw_text):
        """Lọc dữ liệu bẩn triệt để"""
        cleaned = []
        lines = raw_text.split('\n')
        for line in lines:
            nums = "".join(filter(str.isdigit, line.strip()))
            if len(nums) == 5:
                cleaned.append(nums)
        return list(dict.fromkeys(cleaned)) # Loại bỏ trùng lặp

    def analyze_numbers(self, df):
        """Phân tích số đơn theo đặc tả v2"""
        # Chuyển đổi an toàn sang matrix, bỏ qua dòng lỗi
        matrix = []
        for val in df['numbers'].values:
            try:
                row = [int(d) for d in str(val)]
                if len(row) == 5: matrix.append(row)
            except: continue
            
        matrix = np.array(matrix)
        if len(matrix) == 0: return None
        
        stats = {}
        for n in range(10):
            # Tìm các kỳ có sự xuất hiện của n
            appears = np.where(np.any(matrix == n, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # Phân loại theo đặc tả
            recent_5 = matrix[-5:]
            count_recent = sum(1 for row in recent_5 if n in row)
            
            last_idx = appears[-1] if len(appears) > 0 else -1
            dist_last = len(matrix) - 1 - last_idx
            
            # Gán trạng thái
            if count_recent >= 3: state = "NÓNG"
            elif dist_last == 0: state = "NGUY HIỂM"
            elif 3 <= dist_last <= 7: state = "ỔN ĐỊNH"
            else: state = "YẾU"
            
            stats[n] = {"state": state, "gap": dist_last, "freq": count_recent}
        return stats

    def get_prediction(self, stats, df):
        """Logic ghép cặp v2 - Tối đa 1 cặp hoặc KHÔNG ĐÁNH"""
        if not stats: return None, "DỮ LIỆU LỖI", []
        
        # 1. Lọc số chập (Đã mặc định vì phân tích số đơn 0-9)
        # 2. Tiêu chí KHÔNG ĐÁNH
        hot_nums = [n for n, v in stats.items() if v['state'] == "NÓNG"]
        risky_nums = [n for n, v in stats.items() if v['state'] == "NGUY HIỂM"]
        
        if len(hot_nums) >= 6 or len(risky_nums) >= 4:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Thị trường quá nhiễu, nhà cái đang đảo cầu."]

        # 3. Ưu tiên ghép: Ổn định + Hồi (Yếu bắt đầu quay lại)
        stable = [n for n, v in stats.items() if v['state'] == "ỔN ĐỊNH"]
        weak = [n for n, v in stats.items() if v['state'] == "YẾU" and 8 <= v['gap'] <= 12]
        
        candidates = []
        if stable and weak:
            candidates.append(((stable[0], weak[0]), 85, "1 Ổn định + 1 Hồi nhịp"))
        elif len(stable) >= 2:
            candidates.append(((stable[0], stable[1]), 78, "Song thủ Ổn định"))
            
        if not candidates:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Không tìm thấy nhịp cầu an toàn."]
        
        # Chỉ trả về duy nhất 1 cặp tốt nhất
        best = candidates[0]
        return best[0], "PREDICT", [best[2]]

# ================= GUI STREAMLIT =================
def main():
    st.header("🎯 AI LOTOBET 2-TINH (BẢN CHUẨN v2)")
    engine = LotoEngineV2()
    
    if "data_df" not in st.session_state:
        if os.path.exists(engine.DATA_FILE):
            st.session_state.data_df = pd.read_csv(engine.DATA_FILE)
        else:
            st.session_state.data_df = pd.DataFrame(columns=["numbers"])

    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📥 Nhập dữ liệu"])

    with tab2:
        st.subheader("📥 Cập nhật dữ liệu sạch")
        raw_text = st.text_area("Nhập 5 số viết liền (mỗi dòng 1 kỳ):", height=200)
        if st.button("💾 Lọc & Lưu dữ liệu"):
            cleaned_list = engine.clean_data(raw_text)
            if cleaned_list:
                new_df = pd.DataFrame({"numbers": cleaned_list})
                st.session_state.data_df = pd.concat([st.session_state.data_df, new_df]).drop_duplicates().tail(1000)
                st.session_state.data_df.to_csv(engine.DATA_FILE, index=False)
                st.success(f"Đã cập nhật {len(cleaned_list)} kỳ sạch!")
            else:
                st.error("Dữ liệu không đúng định dạng 5 chữ số.")

    with tab1:
        df = st.session_state.data_df
        if len(df) < 15:
            st.warning("Cần tối thiểu 15 kỳ để phân tích chính xác.")
            return

        st.info(f"📈 Đang phân tích trên {len(df)} kỳ gần nhất.")
        stats = engine.analyze_numbers(df)
        
        if stats:
            # Giao diện hiển thị trạng thái số
            
            cols = st.columns(10)
            for i in range(10):
                color = "red" if stats[i]['state'] == "NÓNG" else "green" if stats[i]['state'] == "ỔN ĐỊNH" else "gray"
                cols[i].markdown(f"<div style='text-align:center; color:{color}'><b>{i}</b><br><small>{stats[i]['state']}</small></div>", unsafe_allow_html=True)

            st.divider()
            
            # Dự đoán
            pair, status, reasons = engine.get_prediction(stats, df)
            
            if status == "SKIP":
                st.error(f"🚫 {status}")
                for r in reasons: st.write(f"- {r}")
            else:
                st.success("✅ CẶP SỐ ĐỀ XUẤT")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"<h1 style='color:#FF4B4B;'>{pair[0]}{pair[1]}</h1>", unsafe_allow_html=True)
                with c2:
                    st.write(f"**Lý do:** {reasons[0]}")
                
                # Kết nối Gemini
                with st.expander("🤖 Ý kiến chuyên gia Gemini AI (Phân tích bóng số)"):
                    history_str = ", ".join(df['numbers'].tail(5).astype(str).tolist())
                    stats_str = str([(k, v['state']) for k, v in stats.items()])
                    gemini_advice = ask_gemini_pro(history_str, stats_str)
                    st.write(gemini_advice)

            # Biểu đồ tần suất
            st.divider()
            chart_df = pd.DataFrame([{"Số": k, "Tần suất (5 kỳ)": v['freq']} for k, v in stats.items()])
            fig = px.bar(chart_df, x='Số', y='Tần suất (5 kỳ)', title="Biểu đồ tần suất xuất hiện gần đây")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
