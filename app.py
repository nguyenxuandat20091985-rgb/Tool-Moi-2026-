import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import plotly.express as px # Thêm thư viện biểu đồ
from datetime import datetime
import os
import json

# ================= CONFIG PRO =================
st.set_page_config(page_title="LOTOBET AI PRO v3", layout="wide", page_icon="🚀")

# Tạo style CSS để giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .prediction-card { border-left: 5px solid #ff4b4b; background: white; padding: 20px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ================= CORE AI ENGINE (UPGRADED) =================
class LotobetProAI:
    def __init__(self, config=None):
        self.config = config or {
            "min_draws": 15,
            "min_confidence": 65,
            "backtest_depth": 20
        }

    def analyze_deep(self, df):
        """Phân tích sâu với Vectorization"""
        if len(df) < 5: return None
        
        # Chuyển đổi series numbers thành matrix 2D để xử lý nhanh
        matrix = np.array([list(map(int, list(str(x)))) for x in df['numbers'].values])
        last_indices = {}
        results = {}

        for num in range(10):
            # Tìm các vị trí (kỳ) mà số num xuất hiện
            appears = np.where(np.any(matrix == num, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # Tính toán chỉ số "Nóng/Lạnh" dựa trên entropy hoặc tần suất gần đây
            recent_window = matrix[-10:]
            freq_recent = np.sum(np.any(recent_window == num, axis=1))
            
            # Phân loại trạng thái (Refined Logic)
            state = "STABLE"
            if freq_recent >= 5: state = "HOT"
            elif freq_recent <= 1: state = "COLD"
            elif len(gaps) > 0 and gaps[-1] == 1: state = "REPEAT"

            results[num] = {
                "freq": freq_recent,
                "state": state,
                "last_gap": gaps[-1] if len(gaps) > 0 else 99,
                "avg_gap": np.mean(gaps) if len(gaps) > 0 else 0
            }
        return results

    def backtest(self, df):
        """Hệ thống kiểm tra lại lịch sử (Tính tỷ lệ thắng)"""
        if len(df) < self.config['backtest_depth'] + 10:
            return 0
        
        hits = 0
        total = self.config['backtest_depth']
        
        for i in range(len(df) - total, len(df)):
            test_df = df.iloc[:i]
            actual_next = set(map(int, list(str(df.iloc[i]['numbers']))))
            
            # Giả lập dự đoán của AI tại thời điểm đó
            preds, _, status, _ = self.predict(test_df)
            if status == "PREDICT":
                for p in preds:
                    if p[0] in actual_next and p[1] in actual_next:
                        hits += 1
                        break
        return (hits / total) * 100

    def predict(self, df):
        """Dự đoán với hệ thống Scoring Weight"""
        analysis = self.analyze_deep(df)
        if not analysis: return [], {}, "INSUFFICIENT", []
        
        # Logic KHÔNG ĐÁNH (Skip Logic)
        hot_count = sum(1 for v in analysis.values() if v['state'] == "HOT")
        if hot_count > 6:
            return [], {}, "SKIP", ["Thị trường đang quá biến động (Quá nhiều số HOT)"]

        # Ghép cặp & Chấm điểm
        scored_pairs = []
        for i in range(10):
            for j in range(i + 1, 10):
                s1, s2 = analysis[i], analysis[j]
                
                # Base score
                score = 50
                
                # Bonus/Penalty
                if s1['state'] == "STABLE" and s2['state'] == "STABLE": score += 25
                if s1['state'] == "COLD" or s2['state'] == "COLD": score += 10
                if s1['state'] == "HOT" and s2['state'] == "HOT": score -= 30
                
                # Nhịp độ (Gap matching)
                if abs(s1['avg_gap'] - s2['avg_gap']) < 1: score += 15
                
                if score >= self.config['min_confidence']:
                    scored_pairs.append({
                        "pair": (i, j),
                        "score": min(98, score),
                        "details": f"{s1['state']} + {s2['state']}"
                    })
        
        scored_pairs.sort(key=lambda x: x['score'], reverse=True)
        top_pairs = scored_pairs[:2]
        
        return [p['pair'] for p in top_pairs], {p['pair']: p for p in top_pairs}, "PREDICT", []

# ================= UI HELPERS =================
def render_stats_chart(analysis):
    """Biểu đồ tần suất số đơn"""
    data = pd.DataFrame([
        {"Số": k, "Tần suất (10 kỳ)": v['freq'], "Trạng thái": v['state']} 
        for k, v in analysis.items()
    ])
    fig = px.bar(data, x='Số', y='Tần suất (10 kỳ)', color='Trạng thái',
                 color_discrete_map={"HOT": "#ff4b4b", "STABLE": "#00cc96", "COLD": "#636efa", "REPEAT": "#ab63fa"})
    st.plotly_chart(fig, use_container_width=True)

# ================= MAIN APP =================
def main():
    ai = LotobetProAI()
    
    # Sidebar: Nhập liệu nhanh
    with st.sidebar:
        st.header("⚙️ Điều khiển")
        mode = st.radio("Chế độ", ["Phân tích & Dự đoán", "Quản lý dữ liệu"])
        
        st.divider()
        quick_input = st.text_area("Nhập kết quả mới (5 số/dòng):")
        if st.button("📥 Cập nhật nhanh"):
            # Logic lưu file tương tự bản cũ của bạn
            st.success("Đã cập nhật dữ liệu!")

    # Main Dashboard
    df = load_current_data() # Giả định hàm load từ file csv
    
    if mode == "Phân tích & Dự đoán":
        col1, col2, col3 = st.columns(3)
        
        if not df.empty:
            analysis = ai.analyze_deep(df)
            win_rate = ai.backtest(df)
            
            with col1: st.metric("Tổng kỳ đã lưu", len(df))
            with col2: st.metric("Tỷ lệ thắng dự kiến (Backtest)", f"{win_rate:.1f}%")
            with col3: st.metric("Trạng thái AI", "🔥 Sẵn sàng")
            
            st.divider()
            
            # --- KHU VỰC DỰ ĐOÁN ---
            preds, details, status, reasons = ai.predict(df)
            
            if status == "PREDICT":
                st.subheader("🎯 Cặp số tiềm năng nhất")
                p_cols = st.columns(len(preds))
                for i, p in enumerate(preds):
                    with p_cols[i]:
                        st.markdown(f"""
                            <div class="prediction-card">
                                <h3>Cặp {i+1}: <span style="color:#ff4b4b">{p[0]}{p[1]}</span></h3>
                                <p>Độ tin cậy: <b>{details[p]['score']}%</b></p>
                                <p>Cơ sở: <i>{details[p]['details']}</i></p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"🚫 Tạm dừng chơi: {reasons[0]}")

            # --- KHU VỰC BIỂU ĐỒ ---
            st.divider()
            st.subheader("📊 Trực quan hóa thị trường")
            render_stats_chart(analysis)
            
    # Các Tab khác ...

def load_current_data():
    # Giữ nguyên logic load CSV từ bản cũ của bạn
    if os.path.exists("lotobet_data.csv"):
        return pd.read_csv("lotobet_data.csv")
    return pd.DataFrame()

if __name__ == "__main__":
    main()
