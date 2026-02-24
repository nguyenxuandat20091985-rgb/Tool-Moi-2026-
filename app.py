import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
import os
from datetime import datetime

# ================= CONFIG & STYLE =================
st.set_page_config(page_title="LOTOBET AI PRO 2026", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; border: 1px solid #d1d5db; }
    .prediction-box { padding: 20px; border-radius: 15px; border: 2px solid #ff4b4b; background-color: #ffffff; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ================= CORE ENGINE =================
class LotobetEngine:
    def __init__(self):
        self.data_file = "lotobet_data.csv"
        
    def analyze_patterns(self, df):
        """Thuật toán phân tích nhịp cầu chuyên sâu"""
        if len(df) < 5: return None
        
        # Chuyển dữ liệu sang dạng matrix để xử lý nhanh (Numpy)
        try:
            matrix = np.array([list(map(int, list(str(x)))) for x in df['numbers'].values])
        except:
            return None

        results = {}
        for num in range(10):
            # 1. Tìm các kỳ xuất hiện
            appears = np.where(np.any(matrix == num, axis=1))[0]
            
            # 2. Tính khoảng cách (Gaps)
            gaps = np.diff(appears) if len(appears) > 1 else [99]
            last_appearance = (len(df) - 1) - appears[-1] if len(appears) > 0 else 99
            
            # 3. Tần suất 10 kỳ gần nhất
            recent_freq = np.sum(np.any(matrix[-10:] == num, axis=1))
            
            # 4. Phân loại trạng thái chuẩn
            if recent_freq >= 4: state = "NÓNG"
            elif last_appearance == 0: state = "VỪA RA"
            elif last_appearance > 8: state = "LẠNH"
            else: state = "ỔN ĐỊNH"

            results[num] = {
                "freq": recent_freq,
                "state": state,
                "gap": int(gaps[-1]) if len(gaps) > 0 else 99,
                "avg_gap": float(np.mean(gaps)) if len(gaps) > 0 else 0
            }
        return results

    def predict_strategy(self, df):
        """Chiến thuật ghép cặp thông minh"""
        analysis = self.analyze_patterns(df)
        if not analysis: return [], "THIẾU DỮ LIỆU"

        # Kiểm tra điều kiện "KHÔNG ĐÁNH"
        hot_nums = [n for n, v in analysis.items() if v['state'] == "NÓNG"]
        if len(hot_nums) > 5:
            return [], "SKIP: THỊ TRƯỜNG BIẾN ĐỘNG (NHIỀU SỐ NÓNG)"

        scored_pairs = []
        # Duyệt ghép cặp (Loại bỏ số chập như 11, 22...)
        for i in range(10):
            for j in range(i + 1, 10):
                score = 60 # Điểm gốc
                
                # Chiến thuật 1: Ổn định + Lạnh (Cầu hồi)
                if analysis[i]['state'] == "ỔN ĐỊNH" and analysis[j]['state'] == "LẠNH": score += 25
                # Chiến thuật 2: Hai số có nhịp trung bình khớp nhau
                if abs(analysis[i]['avg_gap'] - analysis[j]['avg_gap']) < 0.5: score += 15
                # Hình phạt: Tránh ghép 2 số đang quá nóng (Dễ gãy)
                if analysis[i]['state'] == "NÓNG" and analysis[j]['state'] == "NÓNG": score -= 40
                
                scored_pairs.append({
                    "pair": f"{i}{j}",
                    "score": score,
                    "desc": f"{analysis[i]['state']} + {analysis[j]['state']}"
                })

        scored_pairs.sort(key=lambda x: x['score'], reverse=True)
        return scored_pairs[:2], "SUCCESS"

# ================= INTERFACE =================
def main():
    st.title("🎯 AI LOTOBET 2-TINH PRO v3")
    engine = LotobetEngine()

    # Load Data
    if os.path.exists(engine.data_file):
        df = pd.read_csv(engine.data_file)
    else:
        df = pd.DataFrame(columns=["time", "numbers"])

    tab1, tab2 = st.tabs(["📊 Dự đoán & Thống kê", "📥 Nhập liệu hệ thống"])

    with tab1:
        if len(df) < 10:
            st.info("👋 Chào anh! Hãy nhập ít nhất 10 kỳ để AI bắt đầu phân tích nhịp.")
        else:
            analysis = engine.analyze_patterns(df)
            preds, status = engine.predict_strategy(df)

            # Dashboard chỉ số nhanh
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số kỳ", len(df))
            c2.metric("Số đang NÓNG", sum(1 for v in analysis.values() if v['state'] == "NÓNG"))
            c3.metric("Trạng thái cầu", "Ổn định" if "SKIP" not in status else "Nguy hiểm")

            st.divider()

            # Hiển thị dự đoán
            if "SKIP" in status:
                st.warning(f"⚠️ Lời khuyên AI: {status}")
            else:
                st.subheader("🤖 Cặp số đề xuất (Ưu tiên cao)")
                p1, p2 = st.columns(2)
                for idx, item in enumerate(preds):
                    with (p1 if idx == 0 else p2):
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h1 style="color: #ff4b4b; font-size: 50px;">{item['pair']}</h1>
                            <p>Độ tin cậy: <b>{item['score']}%</b></p>
                            <small>Cơ sở: {item['desc']}</small>
                        </div>
                        """, unsafe_allow_html=True)

            # Biểu đồ tần suất
            st.divider()
            st.subheader("📈 Biểu đồ nhịp số (10 kỳ gần nhất)")
            chart_data = pd.DataFrame([{"Số": k, "Tần suất": v['freq'], "Trạng thái": v['state']} for k, v in analysis.items()])
            fig = px.bar(chart_data, x="Số", y="Tần suất", color="Trạng thái", 
                         color_discrete_map={"NÓNG": "#ef4444", "ỔN ĐỊNH": "#10b981", "LẠNH": "#3b82f6", "VỪA RA": "#f59e0b"})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📥 Thêm dữ liệu mới")
        raw_input = st.text_area("Dán kết quả (Mỗi kỳ 1 dòng, chỉ lấy 5 số cuối):", height=150)
        if st.button("💾 Lưu kết quả"):
            if raw_input:
                lines = raw_input.strip().split('\n')
                new_data = []
                for l in lines:
                    val = l.strip()[-5:] # Lấy 5 số cuối nếu người dùng dán cả chuỗi dài
                    if val.isdigit() and len(val) == 5:
                        new_data.append({"time": datetime.now().strftime("%H:%M:%S"), "numbers": val})
                
                if new_data:
                    new_df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
                    new_df.to_csv(engine.data_file, index=False)
                    st.success(f"✅ Đã thêm {len(new_data)} kỳ mới!")
                    st.rerun()

if __name__ == "__main__":
    main()
