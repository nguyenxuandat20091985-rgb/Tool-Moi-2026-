import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
import os
import json
from datetime import datetime

# ================= CONFIG PRO =================
st.set_page_config(page_title="LOTOBET AI PRO v3.1", layout="wide", page_icon="🚀")

# Giao diện CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background: white; padding: 15px; border-radius: 10px; border: 1px solid #ddd; }
    .prediction-card { 
        background: #ffffff; padding: 20px; border-radius: 12px; 
        border-left: 8px solid #FF4B4B; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "lotobet_data.csv"

# ================= CORE AI ENGINE (FIXED & TUNED) =================
class LotobetProAI:
    def __init__(self):
        self.config = {"min_draws": 15, "min_confidence": 65}

    def analyze_deep(self, df):
        """Phân tích dữ liệu lịch sử"""
        if len(df) < 5: return None
        
        # Chuyển đổi dữ liệu sang mảng số
        try:
            matrix = np.array([list(map(int, list(str(x)))) for x in df['numbers'].values])
        except:
            return None
            
        results = {}
        for num in range(10):
            # Tìm vị trí xuất hiện (đảo ngược để tính từ kỳ mới nhất)
            appears = np.where(np.any(matrix == num, axis=1))[0]
            gaps = np.diff(appears) if len(appears) > 1 else []
            
            # Tần suất 10 kỳ gần nhất
            recent_window = matrix[-10:]
            freq_recent = np.sum(np.any(recent_window == num, axis=1))
            
            # Phân loại trạng thái (Dùng Tiếng Việt để đồng bộ giao diện)
            state = "ỔN ĐỊNH"
            if freq_recent >= 5: state = "NÓNG"
            elif freq_recent <= 1: state = "LẠNH"
            elif len(gaps) > 0 and gaps[-1] == 1: state = "BỆT"

            results[num] = {
                "freq": int(freq_recent),
                "state": state,
                "last_gap": int(len(df) - 1 - appears[-1]) if len(appears) > 0 else 99,
                "avg_gap": float(np.mean(gaps)) if len(gaps) > 0 else 0
            }
        return results

    def predict(self, df):
        """Dự đoán cặp số"""
        analysis = self.analyze_deep(df)
        if not analysis: return [], {}, "INSUFFICIENT", []
        
        # Logic KHÔNG ĐÁNH
        hot_count = sum(1 for v in analysis.values() if v['state'] == "NÓNG")
        if hot_count > 6:
            return [], {}, "SKIP", ["Thị trường loạn (Quá nhiều số NÓNG)"]

        scored_pairs = []
        for i in range(10):
            for j in range(i + 1, 10):
                s1, s2 = analysis[i], analysis[j]
                score = 50
                
                # Trọng số thuật toán
                if s1['state'] == "ỔN ĐỊNH" and s2['state'] == "ỔN ĐỊNH": score += 25
                if s1['state'] == "LẠNH" or s2['state'] == "LẠNH": score += 10
                if s1['state'] == "NÓNG" and s2['state'] == "NÓNG": score -= 35
                if abs(s1['avg_gap'] - s2['avg_gap']) < 1.2: score += 15
                
                if score >= self.config['min_confidence']:
                    scored_pairs.append({
                        "pair": (i, j),
                        "score": min(95, score),
                        "details": f"{s1['state']} + {s2['state']}"
                    })
        
        scored_pairs.sort(key=lambda x: x['score'], reverse=True)
        return [p['pair'] for p in scored_pairs[:2]], {p['pair']: p for p in scored_pairs[:2]}, "PREDICT", []

# ================= DATA HELPERS =================
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["numbers"])

def save_data(new_numbers):
    df = load_data()
    valid_rows = []
    for n in new_numbers:
        if len(str(n).strip()) == 5:
            valid_rows.append({"numbers": str(n).strip()})
    
    if valid_rows:
        new_df = pd.concat([df, pd.DataFrame(valid_rows)], ignore_index=True)
        new_df.to_csv(DATA_FILE, index=False)
        return len(valid_rows)
    return 0

# ================= MAIN APP =================
def main():
    st.title("🎯 AI LOTOBET 2-TINH PRO v3.1")
    ai = LotobetProAI()
    df = load_data()

    tab1, tab2 = st.tabs(["📊 Dự đoán & Thống kê", "📥 Nhập liệu hệ thống"])

    with tab1:
        if len(df) < 5:
            st.info("👋 Chào anh! Vui lòng nhập ít nhất 5 kỳ ở Tab 'Nhập liệu' để em bắt đầu phân tích.")
        else:
            analysis = ai.analyze_deep(df)
            
            # Header metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số kỳ", len(df))
            # FIX LỖI DÒNG 115:
            nong_count = sum(1 for v in analysis.values() if v['state'] == "NÓNG")
            c2.metric("Số đang NÓNG", nong_count)
            c3.metric("Trạng thái AI", "Sẵn sàng" if nong_count < 7 else "Rủi ro")

            st.divider()

            # Khu vực dự đoán
            preds, details, status, reasons = ai.predict(df)
            if status == "PREDICT" and preds:
                st.subheader("🚀 Cặp số AI đề xuất (2-Tỉnh)")
                pc1, pc2 = st.columns(2)
                for idx, p in enumerate(preds):
                    with (pc1 if idx == 0 else pc2):
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2 style='margin:0; color:#FF4B4B;'>{p[0]}{p[1]}</h2>
                            <p style='margin:5px 0;'>Độ tin cậy: <b>{details[p]['score']}%</b></p>
                            <p style='font-size:0.8em; color:gray;'>Cơ sở: {details[p]['details']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            elif status == "SKIP":
                st.warning(f"⚠️ Cảnh báo: {reasons[0]}")
            else:
                st.info("Chưa tìm thấy cầu đẹp, anh vui lòng đợi thêm vài kỳ.")

            # Biểu đồ Plotly
            st.divider()
            st.subheader("📈 Biểu đồ xu hướng số đơn")
            chart_data = pd.DataFrame([
                {"Số": k, "Tần suất (10 kỳ)": v['freq'], "Trạng thái": v['state']} 
                for k, v in analysis.items()
            ])
            fig = px.bar(chart_data, x="Số", y="Tần suất (10 kỳ)", color="Trạng thái",
                         color_discrete_map={"NÓNG": "#FF4B4B", "ỔN ĐỊNH": "#00CC96", "LẠNH": "#636EFA", "BỆT": "#AB63FA"})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📥 Nhập kết quả mới")
        txt = st.text_area("Dán danh sách kết quả (Ví dụ: 12345, mỗi dòng 1 kỳ)", height=200)
        if st.button("💾 Lưu dữ liệu"):
            lines = [l.strip() for l in txt.split("\n") if l.strip()]
            added = save_data(lines)
            if added > 0:
                st.success(f"✅ Đã thêm {added} kỳ thành công!")
                st.rerun()
            else:
                st.error("Dữ liệu không đúng định dạng (phải là dãy 5 số).")
        
        if st.button("🗑 Xóa hết dữ liệu làm lại"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
                st.rerun()

if __name__ == "__main__":
    main()
