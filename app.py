import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import os
import plotly.express as px

# ================= CONFIG & API =================
# Dán API Key trực tiếp vào đây
GEMINI_API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="LOTOBET AI SIÊU CẤP v3", layout="wide")

# Giao diện Dark Mode chuyên nghiệp
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ================= CORE LOGIC =================
class LotobetUltimateAI:
    def __init__(self):
        self.data_file = "lotobet_db.csv"
        
    def clean_data(self, df):
        """Xử lý dữ liệu đầu vào sạch 100%"""
        clean_matrix = []
        for val in df['numbers'].astype(str):
            nums = [int(d) for d in val.strip() if d.isdigit()]
            if len(nums) == 5:
                clean_matrix.append(nums)
        return np.array(clean_matrix)

    def get_stats(self, matrix):
        """Phân tích số đơn theo đặc tả v2"""
        if len(matrix) == 0: return {}
        stats = {}
        for n in range(10):
            # Tần suất trong 5, 10 kỳ gần nhất
            f5 = sum(1 for row in matrix[-5:] if n in row)
            f10 = sum(1 for row in matrix[-10:] if n in row)
            
            # Tìm kỳ cuối cùng xuất hiện
            last_idx = -1
            for i in range(len(matrix)-1, -1, -1):
                if n in matrix[i]:
                    last_idx = len(matrix) - 1 - i
                    break
            
            # Gán trạng thái
            state = "ỔN ĐỊNH"
            if f5 >= 4: state = "NÓNG"
            elif f5 <= 1: state = "YẾU"
            if last_idx == 0: state = "VỪA RA"
            
            stats[n] = {"f5": f5, "f10": f10, "last": last_idx, "state": state}
        return stats

    def ask_gemini(self, history_str, stats_str):
        """Kết nối Gemini để ra quyết định cuối cùng"""
        prompt = f"""
        Bạn là chuyên gia phân tích Lotobet. Dựa trên đặc tả logic v2:
        1. 2 Tinh: Không chọn số chập (00,11...). Chọn 1 cặp (2 số đơn).
        2. Nhận biết số BỆT và theo bệt nếu nó đang ra đều.
        3. Loại 3 số xấu, giữ 7 số tốt, từ 7 số chọn ra 1 cặp duy nhất.
        4. Nếu cầu nhiễu hoặc quá nóng, trả về 'KHÔNG ĐÁNH'.
        
        Dữ liệu lịch sử: {history_str}
        Thống kê số đơn: {stats_str}
        
        Trả về định dạng JSON: {{"pair": "XY", "confidence": %, "reason": "..."}} hoặc {{"pair": "NONE"}}
        """
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"ERROR: {str(e)}"

# ================= UI APP =================
def main():
    st.title("🎯 AI LOTOBET 2-TINH SIÊU CẤP v3")
    st.caption("Hệ thống kết hợp Thuật toán Xác suất & Trí tuệ nhân tạo Gemini")

    ai = LotobetUltimateAI()
    
    # Khởi tạo file nếu chưa có
    if not os.path.exists(ai.data_file):
        pd.DataFrame(columns=["numbers"]).to_csv(ai.data_file, index=False)

    col_input, col_view = st.columns([1, 2])

    with col_input:
        st.subheader("📥 Nhập dữ liệu")
        raw_input = st.text_area("Nhập 5 số viết liền (mỗi dòng 1 kỳ):", height=200)
        if st.button("💾 CẬP NHẬT HỆ THỐNG"):
            if raw_input:
                lines = [n.strip() for n in raw_input.split("\n") if len(n.strip()) == 5]
                new_df = pd.DataFrame(lines, columns=["numbers"])
                new_df.to_csv(ai.data_file, mode='a', header=False, index=False)
                st.success(f"Đã nạp {len(lines)} kỳ!")
                st.rerun()
        
        if st.button("🗑️ XÓA DỮ LIỆU CŨ"):
            pd.DataFrame(columns=["numbers"]).to_csv(ai.data_file, index=False)
            st.rerun()

    with col_view:
        df = pd.read_csv(ai.data_file)
        if len(df) < 5:
            st.warning("⚠️ Cần tối thiểu 5 kỳ dữ liệu để AI bắt đầu phân tích.")
            return

        matrix = ai.clean_data(df)
        stats = ai.get_stats(matrix)

        st.subheader("📊 Phân tích & Dự đoán")
        
        # Dashboard nhanh
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng số kỳ", len(matrix))
        c2.metric("Số đang bệt", sum(1 for v in stats.values() if v['f5'] >= 4))
        c3.metric("Trạng thái", "ĐANG CÓ CẦU" if len(matrix) > 10 else "DỮ LIỆU ÍT")

        # --- GỌI GEMINI ---
        with st.spinner("🤖 Gemini đang soi cầu..."):
            history_str = ", ".join(df['numbers'].tail(10).tolist())
            stats_str = str(stats)
            res = ai.ask_gemini(history_str, stats_str)

        st.divider()

        # Hiển thị kết quả AI
        if "NONE" in res or "KHÔNG ĐÁNH" in res:
            st.error("🚫 KHÔNG ĐÁNH KỲ NÀY (Cầu đang nhiễu hoặc không an toàn)")
        elif "ERROR" in res:
            st.warning("⚠️ Gemini đang bận, dùng thuật toán dự phòng...")
            # Thuật toán dự phòng (Simple Logic)
            best_nums = sorted(stats.items(), key=lambda x: x[1]['f5'], reverse=True)[:2]
            p1, p2 = best_nums[0][0], best_nums[1][0]
            st.success(f"💎 CẶP SỐ DỰ PHÒNG: {p1}{p2} (Độ tin cậy: 65%)")
        else:
            try:
                # Tìm chuỗi JSON trong phản hồi của Gemini
                import json
                start = res.find('{')
                end = res.rfind('}') + 1
                data = json.loads(res[start:end])
                
                pair = data.get("pair", "NONE")
                conf = data.get("confidence", 0)
                reason = data.get("reason", "")

                if pair != "NONE" and conf >= 60:
                    st.balloons()
                    st.markdown(f"""
                    <div style="background:#1e2130; padding:30px; border-radius:15px; border:2px solid #00ff00; text-align:center">
                        <h1 style="color:#00ff00; font-size:4em; margin:0">{pair}</h1>
                        <h3 style="color:white">ĐỘ TIN CẬY: {conf}%</h3>
                        <p style="color:#aaa">Lý do: {reason}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("🚫 AI KHÔNG TÌM THẤY CẶP SỐ ĐẠT NGƯỠNG AN TOÀN")
            except:
                st.error("❌ Lỗi xử lý dữ liệu AI. Hãy thử nhấn Cập nhật lại.")

        # --- BIỂU ĐỒ ---
        st.divider()
        st.subheader("📈 Thống kê nhịp số đơn")
        chart_df = pd.DataFrame([{"Số": k, "Tần suất (5 kỳ)": v['f5'], "Trạng thái": v['state']} for k, v in stats.items()])
        fig = px.bar(chart_df, x='Số', y='Tần suất (5 kỳ)', color='Trạng thái', barmode='group', height=300)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
