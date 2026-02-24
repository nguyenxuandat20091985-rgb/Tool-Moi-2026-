import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from datetime import datetime
import os
import json

# ================= CONFIG PRO =================
st.set_page_config(page_title="LOTOBET AI PRO v3", layout="wide", page_icon="🎯")

# Tối ưu giao diện
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    div[data-testid="stExpander"] { background-color: white !important; }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "lotobet_data.csv"

# ================= CORE AI ENGINE =================
class LotobetProAI:
    def __init__(self):
        # Đồng bộ hóa nhãn trạng thái để tránh lỗi so sánh
        self.STATE_HOT = "NÓNG"
        self.STATE_STABLE = "ỔN ĐỊNH"
        self.STATE_COLD = "LẠNH"
        self.STATE_REPEAT = "LẶP"

    def analyze_deep(self, df):
        """Phân tích dữ liệu với xử lý lỗi chặt chẽ"""
        if df.empty or len(df) < 5:
            return None
        
        try:
            # Chuyển đổi dữ liệu số thành ma trận
            matrix = []
            for val in df['numbers'].values:
                nums = [int(d) for d in str(val).strip() if d.isdigit()]
                if len(nums) == 5:
                    matrix.append(nums)
            
            if not matrix: return None
            matrix = np.array(matrix)
            results = {}

            for num in range(10):
                # Tìm các kỳ xuất hiện số num
                appears = np.where(np.any(matrix == num, axis=1))[0]
                gaps = np.diff(appears) if len(appears) > 1 else []
                
                # Lấy 10 kỳ gần nhất
                recent_window = matrix[-10:] if len(matrix) >= 10 else matrix
                freq_recent = sum(1 for row in recent_window if num in row)
                
                # Phân loại trạng thái (Đồng bộ với UI)
                if freq_recent >= 5: state = self.STATE_HOT
                elif freq_recent <= 1: state = self.STATE_COLD
                elif len(gaps) > 0 and gaps[-1] == 1: state = self.STATE_REPEAT
                else: state = self.STATE_STABLE

                results[num] = {
                    "freq": freq_recent,
                    "state": state,
                    "last_gap": int(gaps[-1]) if len(gaps) > 0 else 99,
                    "avg_gap": float(np.mean(gaps)) if len(gaps) > 0 else 0.0
                }
            return results
        except Exception as e:
            st.error(f"Lỗi phân tích: {e}")
            return None

    def predict(self, df, analysis):
        """Dự đoán cặp số tiềm năng"""
        if not analysis: return [], "THIẾU DỮ LIỆU", []
        
        # Logic KHÔNG ĐÁNH (Bảo vệ vốn)
        hot_count = sum(1 for v in analysis.values() if v['state'] == self.STATE_HOT)
        if hot_count >= 7:
            return [], "SKIP", ["Thị trường quá nóng (Nhiều số ra dồn dập), rủi ro cao!"]

        scored_pairs = []
        for i in range(10):
            for j in range(i + 1, 10):
                s1, s2 = analysis[i], analysis[j]
                score = 50 # Điểm cơ sở
                
                # Cộng điểm theo chiến thuật
                if s1['state'] == self.STATE_STABLE and s2['state'] == self.STATE_STABLE: score += 30
                if s1['state'] == self.STATE_COLD and s2['state'] == self.STATE_STABLE: score += 20
                if s1['state'] == self.STATE_HOT and s2['state'] == self.STATE_HOT: score -= 40
                
                if score >= 65:
                    scored_pairs.append({
                        "pair": (i, j),
                        "score": min(98, score),
                        "desc": f"{s1['state']} + {s2['state']}"
                    })
        
        scored_pairs.sort(key=lambda x: x['score'], reverse=True)
        return scored_pairs[:2], "PREDICT", []

# ================= DATA HELPERS =================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "numbers"])

def save_data(new_numbers):
    df = load_data()
    now = datetime.now().strftime("%H:%M:%S")
    new_df = pd.DataFrame([{"time": now, "numbers": n} for n in new_numbers if len(n)==5])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# ================= MAIN APP =================
def main():
    st.title("🎯 AI LOTOBET 2-TINH PRO v3")
    ai = LotobetProAI()
    df = load_data()

    tab1, tab2 = st.tabs(["📊 Dự đoán & Thống kê", "📥 Nhập liệu hệ thống"])

    with tab2:
        st.subheader("📥 Cập nhật kết quả mới")
        raw_input = st.text_area("Nhập dãy 5 số (Mỗi kỳ 1 dòng)", height=150, placeholder="12345\n67890...")
        if st.button("💾 Lưu kết quả"):
            if raw_input:
                new_list = [n.strip() for n in raw_input.split("\n") if n.strip()]
                save_data(new_list)
                st.success(f"Đã lưu {len(new_list)} kỳ!")
                st.rerun()

    with tab1:
        if df.empty or len(df) < 5:
            st.info("Chưa đủ dữ liệu. Vui lòng nhập ít nhất 5 kỳ ở tab Nhập liệu.")
            return

        analysis = ai.analyze_deep(df)
        
        if analysis:
            # Fix lỗi AttributeError: Kiểm tra analysis trước khi tính sum
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng kỳ", len(df))
            
            # Đếm số trạng thái an toàn
            hot_numbers = sum(1 for v in analysis.values() if v['state'] == ai.STATE_HOT)
            c2.metric("Số đang NÓNG", hot_numbers, delta_color="inverse")
            c3.metric("Trạng thái", "Ổn định" if hot_numbers < 7 else "Rủi ro")

            st.divider()

            # --- KHU VỰC DỰ ĐOÁN ---
            preds, status, reasons = ai.predict(df, analysis)
            
            if status == "SKIP":
                st.warning(f"⚠️ **KHÔNG NÊN VÀO TIỀN:** {reasons[0]}")
            elif preds:
                st.subheader("🔮 Cặp số đề xuất (Tối ưu nhất)")
                cols = st.columns(len(preds))
                for i, p in enumerate(preds):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="background: white; padding: 20px; border-radius: 15px; border-left: 5px solid #ff4b4b; box-shadow: 2px 2px 10px rgba(0,0,0,0.1)">
                            <h2 style="margin:0; color:#1f1f1f;">{p['pair'][0]}{p['pair'][1]}</h2>
                            <p style="margin:0; color:gray;">Độ tin cậy: <b>{p['score']}%</b></p>
                            <p style="margin:0; font-size: 0.8em;">Cơ sở: {p['desc']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # --- BIỂU ĐỒ ---
            st.divider()
            st.subheader("📈 Phân tích nhịp cầu 0-9")
            chart_data = pd.DataFrame([
                {"Số": k, "Tần suất": v['freq'], "Trạng thái": v['state']} 
                for k, v in analysis.items()
            ])
            fig = px.bar(chart_data, x='Số', y='Tần suất', color='Trạng thái',
                         color_discrete_map={ai.STATE_HOT: "#ef553b", ai.STATE_STABLE: "#00cc96", 
                                           ai.STATE_COLD: "#636efa", ai.STATE_REPEAT: "#ab63fa"})
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
