import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# ================= CONFIG =================
st.set_page_config(page_title="AI LOTOBET 2-TINH v2", layout="wide", page_icon="🎯")

# Giao diện Dark/Light mode tối ưu
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: white; padding: 15px; border-radius: 12px; border: 1px solid #d1d5db; }
    .prediction-box { padding: 25px; border-radius: 15px; background: #ffffff; border-left: 8px solid #ff4b4b; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "lotobet_v2_history.csv"

# ================= CORE AI LOGIC v2 =================
class LotobetAIv2:
    def __init__(self):
        self.labels = {
            "HOT": "🔥 NÓNG",
            "STABLE": "🟢 ỔN ĐỊNH",
            "WEAK": "⚪ YẾU",
            "RISK": "⚠️ NGUY HIỂM"
        }

    def clean_data(self, df):
        """Xử lý dữ liệu thô, loại bỏ dòng lỗi"""
        matrix = []
        for val in df['numbers'].values:
            digits = [int(d) for d in str(val).strip() if d.isdigit()]
            if len(digits) == 5:
                matrix.append(digits)
        return np.array(matrix)

    def analyze_numbers(self, matrix):
        """Phân tích 10 số đơn (0-9) theo Đặc tả v2"""
        if len(matrix) < 5: return None
        
        analysis = {}
        total_draws = len(matrix)
        
        for num in range(10):
            # Vị trí xuất hiện (kỳ)
            appears = np.where(np.any(matrix == num, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # Thống kê tần suất
            recent_3 = sum(1 for row in matrix[-3:] if num in row)
            recent_5 = sum(1 for row in matrix[-5:] if num in row)
            recent_10 = sum(1 for row in matrix[-10:] if num in row)
            
            last_seen = total_draws - 1 - appears[-1] if len(appears) > 0 else 99
            
            # --- PHÂN LOẠI TRẠNG THÁI (Đặc tả mục 6) ---
            if recent_3 >= 2 or (len(gaps) > 0 and gaps[-1] == 1):
                state = "RISK" # Vừa ra hoặc ra dồn
            elif recent_10 >= 4:
                state = "HOT" # Ra dày
            elif 1 <= recent_10 <= 3 and last_seen <= 7:
                state = "STABLE" # Ra đều, có nhịp
            else:
                state = "WEAK" # Ít xuất hiện

            analysis[num] = {
                "state": state,
                "last_seen": last_seen,
                "recent_5": recent_5,
                "avg_gap": np.mean(gaps) if len(gaps) > 0 else 99
            }
        return analysis

    def get_predictions(self, matrix):
        """Logic ghép cặp & Lọc KHÔNG ĐÁNH (Đặc tả mục 7 & 8)"""
        analysis = self.analyze_numbers(matrix)
        if not analysis: return [], "THIẾU DỮ LIỆU", []

        reasons_to_skip = []
        
        # Kiểm tra điều kiện KHÔNG ĐÁNH
        hot_count = sum(1 for v in analysis.values() if v['state'] in ["HOT", "RISK"])
        if hot_count >= 7: 
            reasons_to_skip.append("Toàn số quá NÓNG/NGUY HIỂM (Cầu nhiễu)")
        
        last_draw = matrix[-1]
        repeats = sum(1 for n in last_draw if analysis[n]['state'] == "RISK")
        if repeats >= 3:
            reasons_to_skip.append("Quá nhiều số vừa ra kỳ trước (Cầu bệt ảo)")

        if reasons_to_skip:
            return [], "KHÔNG ĐÁNH KỲ NÀY", reasons_to_skip

        # Ghép cặp (Loại số chập)
        candidates = []
        for i in range(10):
            for j in range(i + 1, 10): # i+1 đảm bảo i != j (Loại số chập 11, 22...)
                s1, s2 = analysis[i], analysis[j]
                
                # Logic Ưu tiên: Ổn định + Hồi hoặc Nhảy + Ổn định
                score = 0
                if s1['state'] == "STABLE" and s2['state'] == "STABLE": score = 85
                elif (s1['state'] == "STABLE" and 5 <= s2['last_seen'] <= 8): score = 78 # Hồi
                elif (s1['state'] == "STABLE" and s2['state'] == "WEAK"): score = 65
                
                # Loại trừ theo đặc tả (Mục 6)
                invalid_states = ["HOT", "RISK", "WEAK"]
                if s1['state'] in ["HOT", "RISK"] and s2['state'] in ["HOT", "RISK"]: score = 0
                if s1['state'] == "WEAK" and s2['state'] == "WEAK": score = 0

                if score >= 60:
                    candidates.append({"pair": f"{i}{j}", "score": score})

        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if not candidates:
            return [], "KHÔNG ĐÁNH KỲ NÀY", ["Không có cặp đạt ngưỡng an toàn"]
            
        return candidates[:1], "PREDICT", [] # Trả về tối đa 1 cặp tốt nhất theo đặc tả

# ================= UI RENDER =================
def main():
    ai = LotobetAIv2()
    
    # Load data
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=["time", "numbers"])

    tab1, tab2 = st.tabs(["📊 Phân tích & Dự đoán", "📥 Nhập dữ liệu"])

    with tab2:
        st.subheader("📥 Cập nhật kết quả Lotobet")
        raw_input = st.text_area("Nhập kết quả (5 số viết liền, mỗi dòng 1 kỳ):", height=200, placeholder="12345\n67890\n...")
        if st.button("💾 Lưu dữ liệu"):
            lines = [l.strip() for l in raw_input.split("\n") if len(l.strip()) == 5]
            if lines:
                new_df = pd.DataFrame([{"time": datetime.now().strftime("%H:%M:%S"), "numbers": l} for l in lines])
                df = pd.concat([df, new_df], ignore_index=True).tail(100) # Giữ 100 kỳ gần nhất
                df.to_csv(DATA_FILE, index=False)
                st.success(f"Đã lưu {len(lines)} kỳ!")
                st.rerun()
            else:
                st.error("Dữ liệu không đúng định dạng (5 chữ số)!")

    with tab1:
        if len(df) < 10:
            st.warning("⚠️ Cần tối thiểu 10 kỳ để AI bắt đầu phân tích.")
            return

        matrix = ai.clean_data(df)
        analysis = ai.analyze_numbers(matrix)
        preds, status, reasons = ai.get_predictions(matrix)

        # Hiển thị kết quả dự đoán
        st.subheader("🎯 Dự đoán kỳ kế tiếp")
        if status == "KHÔNG ĐÁNH KỲ NÀY":
            st.error("🚫 **KHÔNG ĐÁNH KỲ NÀY**")
            for r in reasons: st.write(f"- {r}")
        else:
            for p in preds:
                st.markdown(f"""
                <div class="prediction-box">
                    <span style="color: #6b7280;">CẶP SỐ ĐỀ XUẤT:</span>
                    <h1 style="font-size: 80px; margin: 0; color: #ff4b4b;">{p['pair']}</h1>
                    <span style="font-weight: bold;">Độ tự tin: {p['score']}%</span>
                </div>
                """, unsafe_allow_html=True)

        # Hiển thị bảng trạng thái số đơn
        st.divider()
        st.subheader("📊 Trạng thái số đơn (0-9)")
        cols = st.columns(5)
        for i in range(10):
            with cols[i % 5]:
                data = analysis[i]
                color = "red" if "NÓNG" in ai.labels[data['state']] or "NGUY" in ai.labels[data['state']] else "green"
                st.markdown(f"""
                <div style="padding:10px; border:1px solid #ddd; border-radius:8px; text-align:center; background:white;">
                    <b style="font-size:20px;">{i}</b><br>
                    <span style="color:{color}; font-size:12px;">{ai.labels[data['state']]}</span><br>
                    <small>Gần nhất: {data['last_seen']} kỳ</small>
                </div>
                """, unsafe_allow_html=True)

        # Biểu đồ tần suất
        st.divider()
        freq_df = pd.DataFrame([{"Số": i, "Tần suất (10 kỳ)": analysis[i]['recent_5']*2} for i in range(10)])
        fig = px.bar(freq_df, x='Số', y='Tần suất (10 kỳ)', title="Biểu đồ mật độ xuất hiện", color='Tần suất (10 kỳ)')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
