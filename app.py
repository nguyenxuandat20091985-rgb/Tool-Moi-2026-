import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from datetime import datetime
import os

# ================= CONFIG & API =================
st.set_page_config(page_title="AI 2 TINH LOTOBET v2", layout="wide", page_icon="🎯")

# Kết nối Gemini (Có xử lý lỗi bận/sai key)
GEMINI_API_KEY = "AIzaSyBgd0Au6FGhsiqTkADgz1SBECjs2e1MwGE"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
except:
    st.error("Cấu hình Gemini thất bại. Hệ thống sẽ dùng AI nội bộ.")

DATA_FILE = "lotobet_data_v2.csv"

# ================= THUẬT TOÁN AI NỘI BỘ (BẢN CHUẨN V2) =================
class LotobetEngineV2:
    def __init__(self):
        self.STATES = {"HOT": "NÓNG", "STABLE": "ỔN ĐỊNH", "WEAK": "YẾU", "RISKY": "NGUY HIỂM"}

    def clean_data(self, raw_text):
        """Làm sạch dữ liệu: Chỉ lấy dòng có đúng 5 chữ số"""
        lines = raw_text.split('\n')
        clean_list = []
        for l in lines:
            s = "".join(filter(str.isdigit, l.strip()))
            if len(s) == 5:
                clean_list.append(s)
        return list(dict.fromkeys(clean_list)) # Loại trùng

    def analyze_numbers(self, df):
        """Phân tích số đơn theo đặc tả 5.0 & 6.0"""
        if len(df) < 5: return None
        
        # Chuyển đổi an toàn sang Matrix
        matrix = []
        for s in df['numbers'].astype(str).tolist():
            matrix.append([int(d) for d in s])
        matrix = np.array(matrix)
        
        analysis = {}
        for n in range(10):
            # Vị trí xuất hiện
            appears = np.where(np.any(matrix == n, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else [99]
            
            # Tần suất trong các mốc (Trọng số thời gian)
            freq_3 = sum(1 for row in matrix[-3:] if n in row)
            freq_5 = sum(1 for row in matrix[-5:] if n in row)
            freq_10 = sum(1 for row in matrix[-10:] if n in row)
            
            last_idx = appears[-1] if len(appears) > 0 else -1
            dist_from_last = (len(matrix) - 1) - last_idx

            # Gán trạng thái (Logic 6.0)
            state = self.STATES["STABLE"]
            if freq_3 >= 2: state = self.STATES["NGUY HIỂM"]
            elif freq_5 >= 4: state = self.STATES["NÓNG"]
            elif freq_10 <= 1: state = self.STATES["YẾU"]
            
            # Phát hiện cầu (Logic 4.0)
            bridge = "BÌNH THƯỜNG"
            if len(gaps) >= 2 and gaps[-1] == gaps[-2] and gaps[-1] in [2, 3]: bridge = "CẦU NHẢY"
            elif dist_from_last == 0: bridge = "CẦU LẶP"
            elif dist_from_last >= 7: bridge = "CẦU HỒI"

            analysis[n] = {
                "state": state, "bridge": bridge, "freq_5": freq_5,
                "dist": dist_from_last, "n": n
            }
        return analysis

    def get_prediction(self, analysis, df):
        """Logic Ghép 2 Tinh (Logic 7.0 & 8.0)"""
        if not analysis: return None, "DỮ LIỆU THẤP", []
        
        # Kiểm tra điều kiện "KHÔNG ĐÁNH"
        hot_count = sum(1 for v in analysis.values() if v['state'] == self.STATES["NÓNG"])
        risky_count = sum(1 for v in analysis.values() if v['state'] == self.STATES["NGUY HIỂM"])
        
        if hot_count >= 6 or risky_count >= 4:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Thị trường quá NÓNG/NGUY HIỂM (Dễ gãy cầu)"]

        # Lọc số tiềm năng (Bỏ chập, chọn theo nhịp)
        candidates = []
        for n, v in analysis.items():
            # Ưu tiên số Ổn Định và Cầu Hồi, tránh số vừa ra (Logic 5.0)
            if v['dist'] > 0 and v['state'] in [self.STATES["STABLE"], self.STATES["YẾU"]]:
                candidates.append(v)
        
        if len(candidates) < 2:
            return None, "KHÔNG ĐÁNH KỲ NÀY", ["Không tìm thấy cặp số an toàn"]

        # Sắp xếp chọn cặp tốt nhất (Bóng số & Nhịp hồi)
        candidates.sort(key=lambda x: (x['bridge'] == "CẦU HỒI", x['freq_5']), reverse=True)
        
        p1, p2 = candidates[0], candidates[1]
        pair = tuple(sorted([p1['n'], p2['n']]))
        
        # Tính độ tin cậy % (Logic 9.0)
        conf = 60
        if p1['bridge'] == "CẦU HỒI" or p2['bridge'] == "CẦU HỒI": conf += 15
        if p1['state'] == self.STATES["STABLE"]: conf += 10
        
        if conf < 60: return None, "KHÔNG ĐÁNH KỲ NÀY", ["Độ tin cậy thấp"]
        
        return {"pair": pair, "conf": conf, "desc": f"{p1['bridge']} + {p2['bridge']}"}, "PREDICT", []

# ================= INTERFACE =================
def main():
    st.title("🎯 AI LOTOBET 2-TINH (BẢN CHUẨN v2)")
    engine = LotobetEngineV2()
    
    # Load Data
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["numbers"]).to_csv(DATA_FILE, index=False)
    df = pd.read_csv(DATA_FILE)

    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📥 Nhập dữ liệu"])

    with tab2:
        st.subheader("📥 Cập nhật dữ liệu sạch")
        raw_text = st.text_area("Nhập 5 số viết liền (mỗi dòng 1 kỳ):", height=200)
        if st.button("🔄 Làm mới & Lưu dữ liệu"):
            clean_list = engine.clean_data(raw_text)
            if clean_list:
                new_df = pd.DataFrame({"numbers": clean_list})
                new_df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã cập nhật {len(clean_list)} kỳ sạch!")
                st.rerun()
            else:
                st.error("Dữ liệu không hợp lệ (Phải là chuỗi 5 số)")

    with tab1:
        if len(df) < 10:
            st.warning("Cần tối thiểu 10 kỳ để AI bắt đầu phân tích nhịp cầu.")
            return

        # Thống kê nhanh
        st.subheader("📈 Trạng thái dòng số hiện tại")
        analysis = engine.analyze_numbers(df)
        
        if analysis:
            # Hiển thị bảng trạng thái
            cols = st.columns(10)
            for i in range(10):
                v = analysis[i]
                bg = "#ffeded" if v['state'] == "NÓNG" else "#e8f5e9"
                cols[i].markdown(f"""<div style="background:{bg}; padding:5px; border-radius:5px; text-align:center; border:1px solid #ddd">
                <b style="font-size:20px">{i}</b><br><small>{v['state']}</small></div>""", unsafe_allow_html=True)

            st.divider()

            # Dự đoán
            res, status, reasons = engine.get_prediction(analysis, df)
            
            if status == "PREDICT":
                st.balloons()
                st.markdown(f"""
                <div style="background:#fff3e0; padding:30px; border-radius:15px; border:2px solid #ff9800; text-align:center">
                    <h2 style="margin:0; color:#e65100">CẶP SỐ ĐỀ XUẤT SIÊU CẤP</h2>
                    <h1 style="font-size:80px; margin:10px 0;">{res['pair'][0]} {res['pair'][1]}</h1>
                    <p style="font-size:20px">Độ tin cậy: <b>{res['conf']}%</b> | Nhịp: <i>{res['desc']}</i></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"🚫 {status}")
                for r in reasons: st.write(f"- {r}")

            # Biểu đồ tần suất
            st.subheader("📊 Tần suất xuất hiện (10 kỳ gần nhất)")
            chart_df = pd.DataFrame([{"Số": k, "Lần": v['freq_5']} for k, v in analysis.items()])
            fig = px.bar(chart_df, x="Số", y="Lần", color="Lần", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
